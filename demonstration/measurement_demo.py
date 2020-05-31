import glob

import cv2
import numpy as np
import matplotlib.pyplot as plt
from image_processing.calibration import calibrate_camera
from image_processing.image_processing import find_line_edges
from image_processing.transform import perspective_transform
from image_processing.sliding_window import fit_polynomial_meters, fit_polynomial
from measurement.curvature import measure_curvature_meters, measure_curvature_pixels
from measurement.curvature import get_px_to_meters_coef

chessboard_images = glob.glob('../data/main/camera_cal/*.jpg')
nx = 9  # chessboard corners in x direction
ny = 6  # chessboard corners in y direction
calibration_result = calibrate_camera(chessboard_images, (nx, ny))
mtx = calibration_result["mtx"]
dist = calibration_result["dist"]
img = cv2.imread('../data/main/test_images/test4.jpg')
undistorted_img = cv2.undistort(img, mtx, dist, None, mtx)
img_rgb = cv2.cvtColor(undistorted_img, cv2.COLOR_BGR2RGB)
binary = find_line_edges(undistorted_img, yellow_thresh=(155, 255), white_thresh=((190, 190, 190), (255, 255, 255)))
lines_img = np.dstack((binary, binary, binary)) * 255

warped, M, src, dst = perspective_transform(binary)

ym_per_pix, xm_per_pix = get_px_to_meters_coef(warped.shape)

left_fit, right_fit = fit_polynomial_meters(warped, ym_per_pix, xm_per_pix)
left_fitx, right_fitx, ploty, left_fit_px, right_fit_px, out_img = fit_polynomial(warped)

left_r_px, right_r_px = measure_curvature_pixels(warped.shape[0], left_fit_px, right_fit_px)
left_r, right_r = measure_curvature_meters(warped.shape[0], left_fit, right_fit, ym_per_pix)

print(f'Left R = {left_r}. Pixels = {left_r_px}')
print(f'Right R = {right_r}. Pixels = {right_r_px}')
