import cv2

from image_processing.mask import get_vertices, apply_mask


def find_line_edges(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    yellow = cv2.inRange(hsv, (20, 100, 100), (30, 255, 255))
    white = cv2.inRange(hsv, (0, 0, 214), (172, 111, 255))

    vertices = get_vertices(yellow.shape, 1, 0.1, 0.4)
    masked = apply_mask((yellow | white), vertices)
    return masked
