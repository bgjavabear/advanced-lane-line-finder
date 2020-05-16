import glob
import cv2
from image_processing.calibration import calibrate_camera
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

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

img = mpimg.imread('../data/main/test_images/test4.jpg')
# Apply a distortion correction to raw images
undistorted_img = cv2.undistort(img, mtx, dist, None, mtx)

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
f.tight_layout()
ax1.imshow(img)
ax1.set_title('Original Image', fontsize=50)
ax2.imshow(undistorted_img)
ax2.set_title('Undistorted Image', fontsize=50)
plt.show()
