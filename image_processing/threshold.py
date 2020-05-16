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


def channel_threshold(channel, thresh=(0, 255)):
    binary = np.zeros_like(channel)
    binary[(channel >= thresh[0]) & (channel <= thresh[1])] = 1
    return binary


def find_line_edges(img, saturation_threshold=(0, 255), sobel_threshold=(0, 255), direction_threshold=(0, np.pi / 2)):
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    S = hls[:, :, 2]
    # saturation threshold
    saturation_binary = channel_threshold(S, thresh=saturation_threshold)
    # sobel threshold
    sobel_binary = abs_sobel_thresh(S, thresh=sobel_threshold)
    # direction threshold
    direction_binary = dir_threshold(S, thresh=direction_threshold)
    combined = np.zeros_like(S)
    combined[((saturation_binary == 1) & (sobel_binary == 1) & (direction_binary == 1))] = 1
    return combined
