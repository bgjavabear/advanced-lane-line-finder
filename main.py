import glob
import cv2
from image_processing.calibration import calibrate_camera
from image_processing.image_processing import find_line_edges
import numpy as np

# Compute the camera calibration matrix and distortion coefficients given a set of chessboard images

chessboard_images = glob.glob('data/main/camera_cal/*.jpg')
nx = 9  # chessboard corners in x direction
ny = 6  # chessboard corners in y direction

calibration_result = calibrate_camera(chessboard_images, (nx, ny))
ret = calibration_result["ret"]
mtx = calibration_result["mtx"]
dist = calibration_result["dist"]
rvecs = calibration_result["rvecs"]
tvecs = calibration_result["tvecs"]

test_images = glob.glob('data/main/test_images/*.jpg')

for filename in test_images:
    img = cv2.imread(filename)
    # Apply a distortion correction to raw images
    undistorted_img = cv2.undistort(img, mtx, dist, None, mtx)
    line_edges = find_line_edges(undistorted_img)
