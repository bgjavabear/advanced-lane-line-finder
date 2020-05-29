import cv2

from image_processing.mask import get_vertices, apply_mask
from image_processing.threshold import color_threshold, channel_threshold


def find_line_edges(img, yellow_thresh=(0, 255), white_thresh=((0, 0, 0), (255, 255, 255)),
                    gradient_white_thresh=(0, 255), gradient_yellow_threshold=(0, 255)):
    # threshold yellow color
    LAB = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    lab_b = LAB[:, :, 2]
    yellow_binary = channel_threshold(lab_b, yellow_thresh)
    # threshold white color
    RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    white_binary = color_threshold(RGB, thresh_min=white_thresh[0], thresh_max=white_thresh[1])

    # gradient threshold
    HLS = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    # gradient white threshold
    hls_l = HLS[:, :, 1]
    blur_white = cv2.GaussianBlur(hls_l, (3, 3), 1)
    canny_white = cv2.Canny(blur_white, gradient_white_thresh[0], gradient_white_thresh[1])
    # gradient yellow threshold
    hls_s = HLS[:, :, 2]
    blur_yellow = cv2.GaussianBlur(hls_s, (3, 3), 1)
    canny_yellow = cv2.Canny(blur_yellow, gradient_yellow_threshold[0], gradient_yellow_threshold[1])

    # apply mask
    vertices = get_vertices(white_binary.shape, 1, 0.1, 0.4)
    masked = apply_mask((white_binary & canny_white) | (yellow_binary & canny_yellow), vertices)

    return masked
