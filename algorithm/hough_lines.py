import cv2
import glob
import numpy as np

from image_processing.image_processing import find_line_edges
from image_processing.transform import hough_transformation
from image_processing.calibration import calibrate_camera


def pipeline(img):
    out = np.array(img, copy=True)

    chessboard_images = glob.glob('../data/main/camera_cal/*.jpg')
    nx = 9  # chessboard corners in x direction
    ny = 6  # chessboard corners in y direction
    calibration_result = calibrate_camera(chessboard_images, (nx, ny))
    mtx = calibration_result["mtx"]
    dist = calibration_result["dist"]

    undistorted_img = cv2.undistort(img, mtx, dist, None, mtx)
    binary = find_line_edges(undistorted_img)
    l_line, r_line = hough_transformation(binary)

    cv2.line(out, (l_line.get_x1(), l_line.get_y1()), (l_line.get_x2(), l_line.get_y2()), (250, 0, 0), 10)
    cv2.line(out, (r_line.get_x1(), r_line.get_y1()), (r_line.get_x2(), r_line.get_y2()), (250, 0, 0), 10)
    return out
