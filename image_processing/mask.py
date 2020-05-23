import numpy as np
import cv2


def get_vertices(shape, lower_width_percentage, upper_width_percentage, height_percentage):
    y_max = shape[0]
    x_max = shape[1]

    height = int(y_max * (1 - height_percentage))
    lower_width = int(x_max * lower_width_percentage)
    upper_width = int(x_max * upper_width_percentage)
    x_lower_delta = (x_max - lower_width) // 2
    x_upper_delta = (lower_width - upper_width) // 2 + x_lower_delta

    polygon_vertex1 = (x_lower_delta, y_max)
    polygon_vertex2 = (x_upper_delta, height)
    polygon_vertex3 = (x_max - x_upper_delta, height)
    polygon_vertex4 = (x_max - x_lower_delta, y_max)
    return np.array([[polygon_vertex1, polygon_vertex2, polygon_vertex3, polygon_vertex4]], dtype=np.int32)


def apply_mask(source, vertices):
    mask = np.zeros_like(source)
    cv2.fillPoly(mask, vertices, 1)
    return cv2.bitwise_and(source, mask)
