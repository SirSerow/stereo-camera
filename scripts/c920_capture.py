#!/usr/bin/env python3
"""Capture validated chessboard calibration images from an auto-discovered C920 (Linux).

Requirements: pip install opencv-python; sudo apt install v4l-utils
Usage: python c920_capture.py calibration/left --focus 40 --exposure 200
    python c920_capture.py calibration/left --focus 40 --auto-exposure
"""

import argparse
import json
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

PATTERN_SIZE = (9, 6)  # Internal corners: columns, rows
SQUARE_MM = 25.0


def v4l(device, *args):
    result = subprocess.run(
        ["v4l2-ctl", "--device", str(device), *args],
        capture_output=True,
        text=True,
    )
    if result.returncode:
        raise RuntimeError(
            f"v4l2-ctl {' '.join(args)} failed for {device}: "
            f"{result.stderr.strip() or result.stdout.strip()}"
        )
    return result.stdout


def cameras():
    """Return C920 V4L2 *capture* nodes (not metadata nodes)."""
    found = []
    for dev in sorted(Path("/dev").glob("video[0-9]*"), key=lambda p: int(p.name[5:])):
        try:
            info = v4l(dev, "--info")
        except RuntimeError:
            continue

        card_match = re.search(r"^\s*Card type\s*:\s*(.+)$", info, re.M)
        bus_match = re.search(r"^\s*Bus info\s*:\s*(.+)$", info, re.M)
        card = card_match.group(1).strip() if card_match else ""
        bus = bus_match.group(1).strip() if bus_match else ""

        # The same C920 often creates two /dev/videoX nodes, one for metadata.
        device_caps = info.partition("Device Caps")[2]
        if re.search(r"\bc920(?:s|e|x)?\b", card, re.I) and re.search(
            r"\bVideo Capture\b", device_caps
        ):
            found.append({"device": str(dev), "card": card, "bus": bus})

    # Bus identity is more stable than the changing video node numbers.
    return sorted(found, key=lambda c: (c["bus"], c["device"]))


def available_controls(device):
    output = v4l(device, "--list-ctrls")
    return {
        match.group(1): match.group(2)
        for line in output.splitlines()
        if (
            match := re.match(
                r"^\s*([a-zA-Z][a-zA-Z0-9_]*)\s+0x[0-9a-fA-F]+\s+\([^)]+\)\s*:\s*(.*)$",
                line,
            )
        )
    }


def get_ctrl(device, name):

    output = v4l(device, f"--get-ctrl={name}")

    match = re.search(rf"\b{re.escape(name)}\s*:\s*(-?\d+)", output)

    if not match:
        raise RuntimeError(f"Cannot parse {name} from: {output.strip()}")

    return int(match.group(1))


def set_checked(device, controls, name, value):
    if name not in controls:
        raise RuntimeError(f"Required V4L2 control {name!r} is not supported")

    # Check permitted interval/step before setting, for a useful error message.
    description = controls[name]
    limits = {
        k: int(v) for k, v in re.findall(r"\b(min|max|step)=(-?\d+)", description)
    }
    if "min" in limits and "max" in limits:
        if not (limits["min"] <= value <= limits["max"]):
            raise ValueError(
                f'{name}={value} outside [{limits["min"]}, {limits["max"]}]'
            )
        step = limits.get("step", 1)
        if step > 1 and (value - limits["min"]) % step:
            raise ValueError(
                f'{name} must increase in steps of {step} from {limits["min"]}'
            )

    v4l(device, f"--set-ctrl={name}={value}")
    actual = get_ctrl(device, name)
    if actual != value:
        raise RuntimeError(f"{name}: requested {value}, read back {actual}")
    print(f"  OK  {name} = {actual}")


def configure_controls(device, args):
    supported = available_controls(device)
    expected = {}

    auto_focus = next(
        (n for n in ("focus_automatic_continuous", "focus_auto") if n in supported),
        None,
    )
    auto_exposure = next(
        (n for n in ("auto_exposure", "exposure_auto") if n in supported), None
    )
    auto_gain = next(
        (n for n in ("gain_automatic", "autogain") if n in supported), None
    )

    if auto_focus is None or auto_exposure is None:
        raise RuntimeError("Cannot find the expected C920 focus/exposure auto controls")
    if args.auto_gain and auto_gain is None:
        raise RuntimeError("This camera does not expose an automatic gain control")

    # Set focus mode and the selected exposure mode before related controls.
    exposure_mode = 3 if args.auto_exposure else 1
    settings = [
        (auto_focus, 0),
        ("focus_absolute", args.focus),
        (auto_exposure, exposure_mode),  # C920: 1 = manual, 3 = auto
    ]

    # Dynamic frame rate can cause the C920 to reset manual exposure when
    # streaming starts. Keep the requested frame rate fixed when supported.
    if "exposure_dynamic_framerate" in supported:
        settings.append(("exposure_dynamic_framerate", 0))
    if "exposure_auto_priority" in supported:
        settings.append(("exposure_auto_priority", 0))
    if args.auto_gain:
        settings.append((auto_gain, 1))
    elif args.gain is not None:
        if auto_gain is not None:
            settings.append((auto_gain, 0))
        settings.append(("gain", args.gain))
    if args.mains_hz is not None:
        settings.append(("power_line_frequency", {50: 1, 60: 2}[args.mains_hz]))

    # Reset digital zoom to its minimum (normally 100 = 1x on a C920).
    if "zoom_absolute" in supported:
        match = re.search(r"\bmin=(-?\d+)", supported["zoom_absolute"])
        if match:
            settings.append(("zoom_absolute", int(match.group(1))))

    # Apply manual exposure last because related controls may alter it.
    if not args.auto_exposure:
        settings.append(("exposure_time_absolute", args.exposure))  # 100 us/unit

    print("Applying controls and checking read-back:")
    for name, value in settings:
        set_checked(device, supported, name, value)
        expected[name] = value
    return expected


def video_readback(device):
    fmt = v4l(device, "--get-fmt-video")
    parm = v4l(device, "--get-parm")
    size = re.search(r"Width/Height\s*:\s*(\d+)\s*/\s*(\d+)", fmt)
    pixel = re.search(r"Pixel Format\s*:\s*'([^']+)'", fmt)
    fps = re.search(r"Frames per second\s*:\s*([\d.]+)", parm)
    if not (size and pixel and fps):
        raise RuntimeError("Unable to verify V4L2 format/FPS:\n" + fmt + "\n" + parm)
    return {
        "width": int(size.group(1)),
        "height": int(size.group(2)),
        "fourcc": pixel.group(1),
        "fps": float(fps.group(1)),
    }


def verify(device, args, expected):
    """Raise if the camera deviates from requested settings."""
    current = video_readback(device)
    wanted = {
        "width": args.width,
        "height": args.height,
        "fourcc": "MJPG",
        "fps": float(args.fps),
    }
    for key, value in wanted.items():
        actual = current[key]
        if key == "fps":
            valid = abs(actual - value) < 0.15
        else:
            valid = actual == value
        if not valid:
            raise RuntimeError(f"{key}: requested {value}, actual {actual}")

    for name, value in expected.items():

        actual = get_ctrl(device, name)

        if actual != value:
            raise RuntimeError(f"{name}: requested {value}, actual {actual}")

    return current


def draw_preview(frame, count, message):
    scale = min(960 / frame.shape[1], 540 / frame.shape[0], 1.0)
    preview = cv2.resize(frame, None, fx=scale, fy=scale)
    # flip
    preview = cv2.flip(preview, 1)
    # convert to grayscale
    preview = cv2.cvtColor(preview, cv2.COLOR_BGR2GRAY)
    h, w = preview.shape[:2]
    for i in (1, 2):
        cv2.line(preview, (w * i // 3, 0), (w * i // 3, h), (140, 140, 140), 1)
        cv2.line(preview, (0, h * i // 3), (w, h * i // 3), (140, 140, 140), 1)
    # cv2.rectangle(preview, (0, 0), (w, 76), (0, 0, 0), -1)
    # cv2.putText(
    #     preview,
    #     f"S: save   Q: quit   Saved: {count}",
    #     (12, 28),
    #     cv2.FONT_HERSHEY_SIMPLEX,
    #     0.75,
    #     (255, 255, 255),
    #     2,
    # )
    # cv2.putText(
    #     preview,
    #     message[:90],
    #     (12, 60),
    #     cv2.FONT_HERSHEY_SIMPLEX,
    #     0.56,
    #     (60, 240, 200),
    #     1,
    # )
    return preview


def compute_laplacian(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)  # Apply Laplacian filter
    return np.var(laplacian)  # Compute variance of Laplacian


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "output",
        nargs="?",
        type=Path,
        help="Output directory for images + settings.json",
    )
    parser.add_argument(
        "--list-cameras",
        action="store_true",
        help="List detected C920 capture devices and exit",
    )
    parser.add_argument(
        "--camera-index",
        type=int,
        default=0,
        help="Index in detected C920 list (default: 0)",
    )
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument(
        "--focus", type=int, default=40, help="Manual focus value (default: 40)"
    )
    parser.add_argument(
        "--exposure",
        type=int,
        default=200,
        help="Manual exposure in 100-us units; 200 = 20 ms (default)",
    )
    parser.add_argument(
        "--auto-exposure",
        action="store_true",
        help="Let the camera adjust exposure automatically instead of using --exposure",
    )
    gain_group = parser.add_mutually_exclusive_group()
    gain_group.add_argument(
        "--gain",
        type=int,
        default=None,
        help="Optional fixed gain; omit to keep current gain",
    )
    gain_group.add_argument(
        "--auto-gain",
        action="store_true",
        help="Let the camera adjust gain automatically",
    )
    parser.add_argument(
        "--mains-hz",
        type=int,
        choices=(50, 60),
        default=None,
        help="Optional anti-flicker control",
    )
    parser.add_argument(
        "--square-mm",
        type=float,
        default=SQUARE_MM,
        help="Chessboard square size for metadata (default: 20 mm)",
    )
    args = parser.parse_args()

    if not shutil.which("v4l2-ctl"):
        parser.error("v4l2-ctl is required. Install: sudo apt install v4l-utils")

    found = cameras()
    if not found:
        parser.error(
            "No Logitech C920 capture nodes found. Check USB connection and permissions."
        )

    print("Detected C920 cameras:")
    for index, cam in enumerate(found):
        print(f'  [{index}] {cam["device"]}: {cam["card"]} ({cam["bus"]})')
    if args.list_cameras:
        return
    if args.output is None:
        parser.error("Missing required output directory, e.g. calibration/left")
    if not 0 <= args.camera_index < len(found):
        parser.error("Invalid --camera-index")
    if args.square_mm <= 0:
        parser.error("--square-mm must be positive")

    camera = found[args.camera_index]
    device = camera["device"]
    print(f'\nUsing {device} ({camera["card"]})')
    if len(found) > 1 and args.camera_index == 0:
        print("Note: multiple C920s found; use --camera-index to choose explicitly.")

    cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise RuntimeError(
            f"Cannot open {device}; check if another app owns the camera"
        )

    try:
        print("Requesting MJPG video format...")
        for prop, value in (
            (cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG")),
            (cv2.CAP_PROP_FRAME_WIDTH, args.width),
            (cv2.CAP_PROP_FRAME_HEIGHT, args.height),
            (cv2.CAP_PROP_FPS, args.fps),
        ):
            if not cap.set(prop, value):
                print(f"  Warning: OpenCV reported set({prop}, {value}) unsuccessful")

        expected_controls = configure_controls(device, args)
        # Force a real stream before verifying negotiated format and controls.
        for _ in range(5):
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError("Could not read a frame from camera")
        actual_video = verify(device, args, expected_controls)

        if frame.shape[:2] != (args.height, args.width):
            raise RuntimeError(f"Frame shape mismatch: {frame.shape}")

        print("\nVERIFIED: ", actual_video, "\nControls:", expected_controls)
        output_dir = args.output.expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        existing = [
            int(m.group(1))
            for p in output_dir.glob("image_*.png")
            if (m := re.fullmatch(r"image_(\d+)\.png", p.name))
        ]
        index = max(existing, default=0) + 1

        manifest = output_dir / "settings.json"

        if manifest.exists():
            prior = json.loads(manifest.read_text(encoding="utf-8"))
            for key, new in (
                ("device_bus", camera["bus"]),
                ("video", actual_video),
                ("controls", expected_controls),
                ("pattern_size", list(PATTERN_SIZE)),
                ("square_size_mm", args.square_mm),
            ):
                if prior.get(key) != new:
                    raise RuntimeError(
                        f"{manifest} already exists with different {key}; "
                        "use a new output directory to avoid mixing calibration sessions"
                    )
        else:
            manifest.write_text(
                json.dumps(
                    {
                        "camera": camera["card"],
                        "device_bus": camera["bus"],
                        "device_at_capture": device,
                        "video": actual_video,
                        "controls": expected_controls,
                        "pattern_size": list(PATTERN_SIZE),
                        "square_size_mm": args.square_mm,
                        "started_at_utc": datetime.now(timezone.utc).isoformat(),
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )

        print(
            f"\nSaving valid {PATTERN_SIZE[0]}x{PATTERN_SIZE[1]} corner views to: {output_dir}"
        )
        print("Press S to detect corners and save. Q to quit.")
        message = (
            "Move/tilt the board between captures; keep all 54 inner corners visible."
        )

        while True:
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError("Camera stopped delivering frames")
            preview = draw_preview(frame, index - 1, message)
            cv2.imshow("C920 chessboard calibration capture", preview)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break
            if key not in (ord("s"), ord("S")):
                continue

            # Check again just before every saved image, since controls can reset.
            # verify(device, args, expected_controls)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            detected, corners = cv2.findChessboardCornersSB(
                gray, PATTERN_SIZE, flags=cv2.CALIB_CB_NORMALIZE_IMAGE
            )
            if not detected:
                message = "Not saved: full 9x6 inner-corner pattern was not detected."
                print(message)
                continue

            filename = output_dir / f"image_{index:04d}.png"
            if not cv2.imwrite(str(filename), frame):
                raise RuntimeError(f"Could not save {filename}")
            message = f"Saved {filename.name}; change board position/tilt."
            print(message)
            # Briefly show the detected corners on the preview (not saved image).
            marked = frame.copy()
            cv2.drawChessboardCorners(marked, PATTERN_SIZE, corners, True)
            cv2.imshow(
                "C920 chessboard calibration capture",
                draw_preview(marked, index, message),
            )
            cv2.waitKey(250)
            index += 1
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
