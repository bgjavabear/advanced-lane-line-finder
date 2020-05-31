import glob
import cv2
import os
import numpy as np
from image_processing.calibration import calibrate_camera
from image_processing.image_processing import find_line_edges
from image_processing.transform import perspective_transform, reverse_transform
from image_processing.sliding_window import fit_polynomial, fit_polynomial_meters
from measurement.curvature import measure_curvature_meters, get_px_to_meters_coef, measure_relative_position

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

image_input_path = 'data/main/test_images/'
image_output_path = 'data/main/output/images/'

test_images = os.listdir(image_input_path)

if not os.path.exists(image_output_path):
    os.mkdir(image_output_path)

for filename in test_images:
    image_path = os.path.join(image_input_path, filename)
    img = cv2.imread(image_path)
    # Apply a distortion correction to raw images
    undistorted_img = cv2.undistort(img, mtx, dist, None, mtx)
    line_edges = find_line_edges(undistorted_img, yellow_thresh=(140, 255),
                                 white_thresh=((200, 200, 200), (255, 255, 255)),
                                 gradient_yellow_threshold=(40, 255), gradient_white_thresh=(40, 255))

    # Bird-view transform
    warped, M, Minv, src, dst = perspective_transform(line_edges)

    # sliding window algorithm
    left_fitx, right_fitx, ploty, out_img = fit_polynomial(warped)

    # Calculate curvature and relative position of a car
    ym_per_pix, xm_per_pix = get_px_to_meters_coef(warped.shape)
    left_fit, right_fit = fit_polynomial_meters(warped, ym_per_pix, xm_per_pix)
    left_curverad, right_curverad = measure_curvature_meters(ploty, left_fit, right_fit, ym_per_pix)
    R = (left_curverad + right_curverad) / 2
    D = measure_relative_position(warped, xm_per_pix)

    # reverse warp
    output = reverse_transform(undistorted_img, img, warped, Minv, left_fitx, right_fitx, ploty)
    cv2.putText(output,
                f'Radius of Curvature = {R:.2f}(m).Vehicle is {np.absolute(D):.2f}m {"left" if D < 0 else "right"} of '
                f'center',
                (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255))
    cv2.imwrite(os.path.join(image_output_path, filename), output)
