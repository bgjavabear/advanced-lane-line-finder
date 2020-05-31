import numpy as np
from image_processing.lane_histogram import hist


def measure_curvature_pixels(ploty, left_fit, right_fit):
    y_max = np.max(ploty)
    left_curverad = ((1 + (2 * left_fit[0] * y_max + left_fit[1]) ** 2) ** 1.5) / np.absolute(2 * left_fit[0])
    right_curverad = ((1 + (2 * right_fit[0] * y_max + right_fit[1]) ** 2) ** 1.5) / np.absolute(2 * right_fit[0])
    return left_curverad, right_curverad


def measure_curvature_meters(ploty, left_fit, right_fit, ym_per_px):
    y_max = np.max(ploty)

    left_curverad = ((1 + (2 * left_fit[0] * y_max * ym_per_px + left_fit[1]) ** 2) ** 1.5) / np.absolute(
        2 * left_fit[0])
    right_curverad = ((1 + (2 * right_fit[0] * y_max * ym_per_px + right_fit[1]) ** 2) ** 1.5) / np.absolute(
        2 * right_fit[0])

    return left_curverad, right_curverad


def get_px_to_meters_coef(shape):
    ym_per_pix = 30 / shape[0]
    xm_per_pix = 3.7 / shape[1]
    return ym_per_pix, xm_per_pix
