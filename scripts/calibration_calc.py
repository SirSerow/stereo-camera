import numpy as np
import cv2 as cv
import glob

import argparse
import logging
import os
import json

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def run_calibration(folder_name: str):

    calibration_info = {}

    # termination criteria
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((6 * 9, 3), np.float32)
    objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.

    images = glob.glob(f"calibration/{folder_name}/*.png")

    for fname in images:
        img = cv.imread(fname)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret, corners = cv.findChessboardCorners(gray, (9, 6), None)

        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)

            corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)

            # Draw and display the corners
            cv.drawChessboardCorners(img, (9, 6), corners2, ret)
            cv.imshow("img", img)
            cv.waitKey(500)

    cv.destroyAllWindows()

    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None
    )

    mtx_serializable = (
        mtx.tolist()
    )  # Convert the camera matrix to a serializable format
    dist_serializable = (
        dist.tolist()
    )  # Convert the distortion coefficients to a serializable format

    calibration_info["camera_matrix"] = mtx_serializable
    calibration_info["distortion_coefficients"] = dist_serializable
    # calibration_info["rotation_vectors"] = [r.tolist() for r in rvecs]
    # calibration_info["translation_vectors"] = [t.tolist() for t in tvecs]

    logger.info("Camera matrix:\n%s", mtx)
    logger.info("Distortion coefficients:\n%s", dist)

    img = cv.imread(f"calibration/{folder_name}/image_0001.png")
    h, w = img.shape[:2]
    newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

    # undistort
    dst = cv.undistort(img, mtx, dist, None, newcameramtx)

    # crop the image
    x, y, w, h = roi
    dst = dst[y : y + h, x : x + w]
    cv.imwrite(f"calibration/{folder_name}/calibresult-{folder_name}.png", dst)

    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv.norm(
            imgpoints[i].reshape(-1, 2),
            imgpoints2.reshape(-1, 2),
            cv.NORM_L2,
        ) / np.sqrt(len(imgpoints2))
        mean_error += error

    logger.info("total error: %s", mean_error / len(objpoints))
    calibration_info["reprojection_error"] = mean_error / len(objpoints)

    # Save calibration info to a JSON file
    with open(f"calibration/{folder_name}/calibration_info.json", "w") as f:
        json.dump(calibration_info, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Camera calibration")
    parser.add_argument(
        "folder_name", type=str, help="Folder containing calibration images"
    )
    args = parser.parse_args()

    # check if the folder exists
    if not os.path.exists(f"calibration/{args.folder_name}"):
        logger.error("Folder calibration/%s does not exist", args.folder_name)
        exit(1)

    run_calibration(args.folder_name)
