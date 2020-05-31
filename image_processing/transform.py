import math

import cv2
import numpy as np

from utils.geometric_utils import Line, calculate_x_by_angle


def reverse_transform(undist, image, warped, Minv, left_fitx, right_fitx, ploty):
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (image.shape[1], image.shape[0]))

    # Combine the reult with the original image
    result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)
    return result


def perspective_transform(source):
    left_line, right_line = hough_transformation(source)
    src = np.float32([[left_line.x1, left_line.y1], [left_line.x2, left_line.y2], [right_line.x2, right_line.y2],
                      [right_line.x1, right_line.y1]])
    dst = np.float32([[200, 700], [200, 300], [1000, 300], [1000, 700]])
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    warped = cv2.warpPerspective(source, M, source.shape[::-1], flags=cv2.INTER_LINEAR)
    return warped, M, Minv, src, dst


def hough_transformation(source, rho=1, theta=np.pi / 180, threshold=20, min_line_length=0, max_line_gap=40):
    source_shape = source.shape
    lines = cv2.HoughLinesP(source, rho, theta, threshold, np.array([]), min_line_length, max_line_gap)

    right_x_arr = np.array([]), np.array([])
    left_x_arr = np.array([]), np.array([])
    left_line_angle_arr = np.array([])
    right_line_angle_arr = np.array([])
    angles = np.array([])

    # divide lines into 2 categories: left line and right line.
    # Extend those lines to the bottom of the image
    # find starting points for left and right lines at the bottom of the image
    # find average values for starting points for left and right lines
    # find average values for angles for those lines

    for line in lines:
        for x1, y1, x2, y2 in line:
            rad = math.atan2(y2 - y1, x2 - x1)
            angle = rad * 180 / math.pi
            angles = np.append(angles, angle)
            if angle < -10:
                left_line = Line(x1, y1, x2, y2)
                y = source_shape[0]
                x = left_line.calculate_x(y)
                left_x_arr = np.append(left_x_arr, x)
                left_line_angle_arr = np.append(left_line_angle_arr, angle)
            elif angle > 10:
                right_line = Line(x1, y1, x2, y2)
                y = source_shape[0]
                x = right_line.calculate_x(y)
                right_x_arr = np.append(right_x_arr, x)
                right_line_angle_arr = np.append(right_line_angle_arr, angle)

    x1_right = np.average(right_x_arr)
    x1_left = np.average(left_x_arr)

    left_line_angle = np.average(left_line_angle_arr)
    right_line_angle = np.average(right_line_angle_arr)

    y2_right = source_shape[0] * .65
    y2_left = source_shape[0] * .65

    x2_right = calculate_x_by_angle(x1_right, source_shape[0], y2_right, right_line_angle)
    x2_left = calculate_x_by_angle(x1_left, source_shape[0], y2_left, left_line_angle)

    left_line = Line(x1_left, source_shape[0], x2_left, y2_left)
    right_line = Line(x1_right, source_shape[0], x2_right, y2_right)

    return left_line, right_line
