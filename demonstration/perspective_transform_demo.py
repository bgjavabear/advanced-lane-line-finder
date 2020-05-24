import glob

import cv2
import numpy as np
import matplotlib.pyplot as plt
from image_processing.calibration import calibrate_camera
from image_processing.image_processing import find_line_edges
from image_processing.transform import perspective_transform

chessboard_images = glob.glob('../data/main/camera_cal/*.jpg')
nx = 9  # chessboard corners in x direction
ny = 6  # chessboard corners in y direction
calibration_result = calibrate_camera(chessboard_images, (nx, ny))
mtx = calibration_result["mtx"]
dist = calibration_result["dist"]
img = cv2.imread('../data/main/test_images/test3.jpg')
undistorted_img = cv2.undistort(img, mtx, dist, None, mtx)
img_rgb = cv2.cvtColor(undistorted_img, cv2.COLOR_BGR2RGB)
binary = find_line_edges(undistorted_img, yellow_thresh=(155, 255), white_thresh=((190, 190, 190), (255, 255, 255)))
lines_img = np.dstack((binary, binary, binary)) * 255

warped, M = perspective_transform(binary)

warped_lines_img = np.dstack((warped, warped, warped)) * 255

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
f.tight_layout()
ax1.imshow(lines_img)
ax1.set_title('Original Image', fontsize=50)
ax2.imshow(warped_lines_img)
ax2.set_title('Warped', fontsize=50)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
plt.show()
