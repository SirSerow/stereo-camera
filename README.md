# Stereo Camera Project

The goal of this project is building a stereo camera with two Logitech C920 cameras, testing this build and comparing the accuracy with real production stereo cameras. Initially the disparity and depth map computation will be implemented. Following that the image feature detection and point cloud generation will be added.

## Project Steps

1. Basic application architecture layout.
2. Camera calibration.
3. Validation of disparity and depth computation.
4. Porting the project to a small single-board computer.


## Application Architecture

The application consists of several processes running in parallel and exchanging the data between each other using queues. Following processes have to be implemented:

- TopController: the top process (entry point). On startup this process reads the config.yml file, sets up the logger which is passed to other processes, sets up the data queues and starts all the processes according to the configuration file.

- StereoCameraReader: this process reads the camera images from two calibrated cameras. This could be split into two separate processes, but it was decided to unite both camera into a single process to guarantee the simultaneous frame acquisition for both cameras. The output of this process is an array of two images (RGB).

- DisparityComputer: this process takes the array of two images from StereoCameraReader and computes the disparity map for one of the cameras. The output of this process is the computed disparity map.

- DepthMapCompute: this process takes the disparity map and the precomputed projection matrices and calculated the depth map for all the points of the image. The output of this process is computed depth map.

- VideoWriter: this process writes out the image processing result (original image, disparity map or depth map depending on which processes are running), saving it in a mp4 video format. The VideoWriter can be configured to output the video in various sizes, or not output anything at all. If imshow mode is requested in the configuration, than the OpenCV imshow window is used to actively show the output on the screen (does not work in headless mode).


## Camera Calibration

### Camera description

#### Logitech C920

A standard Full-HD web-camera. The camera is connected through USB and using the default web-camera communication protocol so no special driver installation or dedicated SDK is needed for frame acquisition. The parameters of web camera are described ina table below.

| Parameter | Value |
|---|---|
| Model | Logitech C920 HD Pro Webcam |
| Maximum resolution | 1920 × 1080 |
| Maximum frame rate | 30 FPS at 1080p |
| Sensor resolution | 3 MP |
| Focal length | 3.67 mm |
| Horizontal FOV | 70.42° |
| Vertical FOV | 43.3° |
| Diagonal FOV | 78° |
| Focus | Autofocus / Manual |
| Interface | USB 2.0 |
| Video standard | UVC |

### Stereo Setup

The stereo camera setup was implemented using a double mount shoe, ensuring that both cameras are mounted in a fixed relative position to each other. The distance between cameras can be adjusted by moving the mount shoe, but the recalibration is required incase of ajustment.



