import cv2
import glob
import numpy as np

from image_processing.image_processing import find_line_edges
from image_processing.calibration import calibrate_camera


def pipeline(img):
    chessboard_images = glob.glob('../data/main/camera_cal/*.jpg')
    nx = 9  # chessboard corners in x direction
    ny = 6  # chessboard corners in y direction
    calibration_result = calibrate_camera(chessboard_images, (nx, ny))
    mtx = calibration_result["mtx"]
    dist = calibration_result["dist"]

    undistorted_img = cv2.undistort(img, mtx, dist, None, mtx)
    binary = find_line_edges(undistorted_img)
    line_edges = np.dstack((binary, binary, binary)) * 255
    return line_edges
