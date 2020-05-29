import numpy as np


def hist(source):
    bottom_half = source[source.shape[0] // 2:, :]

    histogram = np.sum(bottom_half, axis=0)
    return histogram
