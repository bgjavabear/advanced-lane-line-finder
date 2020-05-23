import cv2
import numpy as np


def perspective_transform(source, filename):
    # TODO Need to create an algorithm for picking source and destination points. Manual is not an option
    if filename == 'straight_lines1.jpg' or filename == 'straight_lines2.jpg':
        src = np.float32([[263, 680], [453, 548], [837, 548], [1040, 680]])
    elif filename == 'test1.jpg':
        src = np.float32([[300, 680], [507, 530], [840, 530], [1090, 680]])
    elif filename == 'test2.jpg':
        src = np.float32([[461, 573], [569, 464], [701, 464], [928, 573]])
    else:
        raise Exception('The method supports only a limited set of images')
    dst = np.float32([[100, 100], [100, 400], [400, 400], [400, 100]])
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(source, M, source.shape, flags=cv2.INTER_LINEAR)
    return warped, M
