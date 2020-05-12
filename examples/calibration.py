import numpy as np
import cv2
import matplotlib.pyplot as plt
import glob


def cal_undistort(img, objpoints, imgpoints):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    return cv2.undistort(img, mtx, dist, None, mtx)


images = glob.glob('../data/calibration/*.jpg')

objpoints = []  # 3D points in real world space
imgpoints = []  # 2D points in image plane

objp = np.zeros((6 * 8, 3), np.float32)
objp[:, :2] = np.mgrid[0:8, 0:6].T.reshape(-1, 2)  # x,y coordinates

for filename in images:
    img = cv2.imread(filename)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (8, 6), None)

    if ret is True:
        imgpoints.append(corners)
        objpoints.append(objp)

# mtx - 3d to 2d transformation matrix
# dist - distortion coefficients
# rvecs - rotation vector
# tvecs - translation vector

img = cv2.imread('../data/calibration/GOPR0032.jpg')
undistorted = cal_undistort(img, objpoints, imgpoints)

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
f.tight_layout()
ax1.imshow(img)
ax1.set_title('Original Image', fontsize=50)
ax2.imshow(undistorted)
ax2.set_title('Undistorted Image', fontsize=50)
plt.show()
