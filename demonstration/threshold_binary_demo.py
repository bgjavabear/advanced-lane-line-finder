import glob

import cv2
import matplotlib.pyplot as plt
import numpy as np

from image_processing.calibration import calibrate_camera
from image_processing.threshold import channel_threshold

# Compute the camera calibration matrix and distortion coefficients given a set of chessboard images

chessboard_images = glob.glob('../data/main/camera_cal/*.jpg')
nx = 9  # chessboard corners in x direction
ny = 6  # chessboard corners in y direction

calibration_result = calibrate_camera(chessboard_images, (nx, ny))
ret = calibration_result["ret"]
mtx = calibration_result["mtx"]
dist = calibration_result["dist"]
rvecs = calibration_result["rvecs"]
tvecs = calibration_result["tvecs"]

img = cv2.imread('../data/main/test_images/test5.jpg')
# Apply a distortion correction to raw images
undistorted_img = cv2.undistort(img, mtx, dist, None, mtx)

hls = cv2.cvtColor(undistorted_img, cv2.COLOR_BGR2HLS)
S = hls[:, :, 2]
binary = channel_threshold(S, (170, 255))
binary = np.dstack((binary, binary, binary)) * 255
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
f.tight_layout()
ax1.imshow(img_rgb)
ax1.set_title('Original Image', fontsize=50)
ax2.imshow(binary)
ax2.set_title('Thresholded S', fontsize=50)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
plt.show()
