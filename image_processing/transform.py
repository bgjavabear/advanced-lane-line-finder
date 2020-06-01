import math

import cv2
import numpy as np

from utils.geometric_utils import Line, get_line


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

    # Combine the result with the original image
    result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)
    return result


def perspective_transform(source):
    left_line, right_line = hough_transformation(source)
    src = np.float32([[int(left_line.x1), int(left_line.y1)], [int(left_line.x2), int(left_line.y2)],
                      [int(right_line.x2), int(right_line.y2)], [int(right_line.x1), int(right_line.y1)]])
    dst = np.float32([[200, 700], [200, 300], [1000, 300], [1000, 700]])
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    warped = cv2.warpPerspective(source, M, source.shape[::-1], flags=cv2.INTER_LINEAR)
    return warped, M, Minv, src, dst


def hough_transformation(source, rho=1, theta=np.pi / 180, threshold=40, min_line_length=40, max_line_gap=10):
    source_shape = source.shape
    lines = cv2.HoughLinesP(source, rho, theta, threshold, np.array([]), min_line_length, max_line_gap)

    # divide lines into 2 categories: left line and right line.
    # Extend those lines to the bottom of the image
    # find starting points for left and right lines at the bottom of the image
    # find average values for starting points for left and right lines
    # find average values for angles for those lines

    left_m = []
    left_b = []

    right_m = []
    right_b = []

    for line in lines:
        for x1, y1, x2, y2 in line:
            rad = math.atan2(y2 - y1, x2 - x1)
            angle = rad * 180 / math.pi
            if angle < -20:
                left_line = Line(x1, y1, x2, y2)
                left_m.append(left_line.m)
                left_b.append(left_line.b)
            elif angle > 20:
                right_line = Line(x1, y1, x2, y2)
                right_m.append(right_line.m)
                right_b.append(right_line.b)

    left_average_line = get_line(source_shape[0], source_shape[0] * .75, np.average(left_m), np.average(left_b))
    right_average_line = get_line(source_shape[0], source_shape[0] * .75, np.average(right_m), np.average(right_b))

    return left_average_line, right_average_line
