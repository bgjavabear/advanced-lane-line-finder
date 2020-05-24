import glob

import cv2
import matplotlib.pyplot as plt
import numpy as np

from image_processing.calibration import calibrate_camera
from image_processing.transform import hough_transformation
from image_processing.image_processing import find_line_edges

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

left_line, right_line = hough_transformation(binary)

cv2.line(img_rgb, (left_line.x1, left_line.y1), (left_line.x2, left_line.y2), (250, 0, 0), 10)
cv2.line(img_rgb, (right_line.x1, right_line.y1), (right_line.x2, right_line.y2), (250, 0, 0), 10)

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
f.tight_layout()
ax1.imshow(img_rgb)
ax1.set_title('Original Image', fontsize=50)
ax2.imshow(lines_img)
ax2.set_title('Threshold', fontsize=50)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
plt.show()
