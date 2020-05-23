import cv2
from image_processing.threshold import color_threshold, channel_threshold
from image_processing.mask import get_vertices, apply_mask


def find_line_edges(img, yellow_thresh=(0, 255), white_thresh=((0, 0, 0), (255, 255, 255))):
    # threshold yellow color
    LAB = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    lab_b = LAB[:, :, 2]
    yellow_binary = channel_threshold(lab_b, yellow_thresh)
    # threshold white color
    RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    white_binary = color_threshold(RGB, thresh_min=white_thresh[0], thresh_max=white_thresh[1])
    vertices = get_vertices(white_binary.shape, 1, 0.1, 0.4)
    masked = apply_mask(white_binary | yellow_binary, vertices)
    return masked
