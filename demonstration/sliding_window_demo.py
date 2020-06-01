import glob

import cv2
import numpy as np
import matplotlib.pyplot as plt
from image_processing.calibration import calibrate_camera
from image_processing.image_processing import find_line_edges
from image_processing.transform import perspective_transform
from image_processing.sliding_window import fit_polynomial

chessboard_images = glob.glob('../data/main/camera_cal/*.jpg')
nx = 9  # chessboard corners in x direction
ny = 6  # chessboard corners in y direction
calibration_result = calibrate_camera(chessboard_images, (nx, ny))
mtx = calibration_result["mtx"]
dist = calibration_result["dist"]
img = cv2.imread('../data/main/test_images/straight_lines1.jpg')
undistorted_img = cv2.undistort(img, mtx, dist, None, mtx)
img_rgb = cv2.cvtColor(undistorted_img, cv2.COLOR_BGR2RGB)
binary = find_line_edges(undistorted_img)
lines_img = np.dstack((binary, binary, binary)) * 255

warped, M, Minv, src, dst = perspective_transform(binary)

left_fitx, right_fitx, ploty, out_img = fit_polynomial(warped)

f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(24, 9))

ax1.imshow(img_rgb)
ax1.set_title('Original Image', fontsize=50)

ax2.imshow(lines_img)
ax2.set_title('Binary', fontsize=50)

ax3.plot(left_fitx, ploty, color='yellow')
ax3.plot(right_fitx, ploty, color='yellow')
ax3.imshow(out_img)

ax4.imshow(warped, cmap='gray')
ax4.set_title('Warped', fontsize=50)
plt.show()
