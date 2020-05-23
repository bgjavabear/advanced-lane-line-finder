import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt

from image_processing.transform import perspective_transform
from image_processing.calibration import calibrate_camera
from image_processing.image_processing import find_line_edges

chessboard_images = glob.glob('../data/main/camera_cal/*.jpg')
nx = 9  # chessboard corners in x direction
ny = 6  # chessboard corners in y direction
calibration_result = calibrate_camera(chessboard_images, (nx, ny))
mtx = calibration_result["mtx"]
dist = calibration_result["dist"]
img = cv2.imread('../data/main/test_images/test2.jpg')
undistorted_img = cv2.undistort(img, mtx, dist, None, mtx)
img_rgb = cv2.cvtColor(undistorted_img, cv2.COLOR_BGR2RGB)
binary = find_line_edges(undistorted_img, yellow_thresh=(155, 255), white_thresh=((230, 230, 230), (255, 255, 255)))

warped, M = perspective_transform(binary, 'test2.jpg')

line_edges = np.dstack((binary, binary, binary)) * 255
f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 9))
f.tight_layout()
ax1.imshow(img_rgb)
ax1.set_title('Original Image', fontsize=50)
ax2.imshow(line_edges)
ax2.set_title('Threshold', fontsize=50)
ax3.imshow(warped, cmap='gray')
ax3.set_title('Warped', fontsize=50)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
plt.show()
