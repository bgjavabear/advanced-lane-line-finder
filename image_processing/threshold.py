import cv2
import numpy as np


def abs_sobel_thresh(S, orient='x', sobel_kernel=3, thresh=(0, 255)):
    # Take the derivative in x or y given orient = 'x' or 'y'
    if orient == 'y':
        sobel = cv2.Sobel(S, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    else:
        sobel = cv2.Sobel(S, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    # Take the absolute value of the derivative or gradient
    abs_sobel = np.absolute(sobel)
    # Scale to 8-bit (0 - 255) then convert to type = np.uint8
    scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))
    # Create a mask of 1's where the scaled gradient magnitude is > thresh_min and < thresh_max
    sbinary = np.zeros_like(scaled_sobel)
    sbinary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    return sbinary


def mag_thresh(S, sobel_kernel=3, thresh=(0, 255)):
    # Take the gradient in x and y separately
    sobelx = cv2.Sobel(S, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(S, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Calculate the magnitude
    magn = np.sqrt(np.power(sobelx, 2) + np.power(sobely, 2))
    # Scale to 8-bit (0 - 255) and convert to type = np.uint8
    scaled_magn = np.uint8(255 * magn / np.max(magn))
    # Create a binary mask where mag thresholds are met
    binary = np.zeros_like(scaled_magn)
    binary[(scaled_magn >= thresh[0]) & (scaled_magn <= thresh[1])] = 1
    return binary


def dir_threshold(S, sobel_kernel=3, thresh=(0, np.pi / 2)):
    # Take the gradient in x and y separately
    sobelx = cv2.Sobel(S, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(S, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Take the absolute value of the x and y gradients
    abs_sobelx = np.absolute(sobelx)
    abs_sobely = np.absolute(sobely)
    # calculate the direction of the gradient
    theta = np.arctan2(abs_sobely, abs_sobelx)
    # Create a binary mask where direction thresholds are met
    binary = np.zeros_like(theta)
    binary[(theta >= thresh[0]) & (theta <= thresh[1])] = 1
    return binary


def color_threshold(img, thresh_min=(0, 0, 0), thresh_max=(255, 255, 255)):
    img1 = img[:, :, 0]
    img2 = img[:, :, 1]
    img3 = img[:, :, 2]
    binary = np.zeros_like(img1)
    binary[((img1 >= thresh_min[0]) & (img1 <= thresh_max[0])) |
           ((img2 >= thresh_min[1]) & (img2 <= thresh_max[1])) |
           ((img3 >= thresh_min[2]) & (img3 <= thresh_max[2]))] = 1
    return binary


def channel_threshold(channel, thresh=(0, 255)):
    binary = np.zeros_like(channel)
    binary[(channel >= thresh[0]) & (channel <= thresh[1])] = 1
    return binary
