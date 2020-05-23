import glob

import cv2
import matplotlib.pyplot as plt

from image_processing.calibration import calibrate_camera


def display_all_channels_for_color_spaces(imgg):
    HLS = cv2.cvtColor(imgg, cv2.COLOR_BGR2HLS)
    RGB = cv2.cvtColor(imgg, cv2.COLOR_BGR2RGB)
    HSV = cv2.cvtColor(imgg, cv2.COLOR_BGR2HSV)
    LAB = cv2.cvtColor(imgg, cv2.COLOR_BGR2LAB)

    hls_h = HLS[:, :, 0]
    hls_l = HLS[:, :, 1]
    hls_s = HLS[:, :, 2]

    rgb_r = RGB[:, :, 0]
    rgb_g = RGB[:, :, 1]
    rgb_b = RGB[:, :, 2]

    hsv_h = HSV[:, :, 0]
    hsv_s = HSV[:, :, 1]
    hsv_v = HSV[:, :, 2]

    lab_l = LAB[:, :, 0]
    lab_a = LAB[:, :, 1]
    lab_b = LAB[:, :, 2]

    plt.imshow(RGB)

    f, ((ax1, ax2, ax3), (ax4, ax5, ax6), (ax7, ax8, ax9), (ax10, ax11, ax12)) = plt.subplots(4, 3)

    # RGB
    ax1.imshow(rgb_r)
    ax1.set_title('RGB(R channel)')
    ax2.imshow(rgb_g)
    ax2.set_title('RGB(G channel)')
    ax3.imshow(rgb_b)
    ax3.set_title('RGB(B channel)')

    # HLS
    ax4.imshow(hls_h)
    ax4.set_title('HLS(H channel)')
    ax5.imshow(hls_l)
    ax5.set_title('HLS(L channel)')
    ax6.imshow(hls_s)
    ax6.set_title('HLS(S channel)')

    # HSV
    ax7.imshow(hsv_h)
    ax7.set_title('HSV(H channel)')
    ax8.imshow(hsv_s)
    ax8.set_title('HSV(S channel)')
    ax9.imshow(hsv_v)
    ax9.set_title('HSV(V channel)')

    # LAB
    ax10.imshow(lab_l)
    ax10.set_title('LAB(L channel)')
    ax11.imshow(lab_a)
    ax11.set_title('LAB(A channel)')
    ax12.imshow(lab_b)
    ax12.set_title('LAB(B channel)')

    for ax in f.get_axes():
        ax.label_outer()

    plt.show()


# Compute the camera calibration matrix and distortion coefficients given a set of chessboard images

chessboard_images = glob.glob('../data/main/camera_cal/*.jpg')
nx = 9  # chessboard corners in x direction
ny = 6  # chessboard corners in y direction

calibration_result = calibrate_camera(chessboard_images, (nx, ny))
mtx = calibration_result["mtx"]
dist = calibration_result["dist"]

img = cv2.imread('../data/main/test_images/test5.jpg')
# Apply a distortion correction to raw images
undistorted_img = cv2.undistort(img, mtx, dist, None, mtx)

# let's try to use RGB,HLS, HSV,LAB color spaces and individual channels to identify the best options
display_all_channels_for_color_spaces(undistorted_img)
# I will stick to LAB color space (B channel). It gives good results for all test images I work with
