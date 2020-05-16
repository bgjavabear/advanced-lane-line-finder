import numpy as np
import cv2
from utils.caching import Cache


@Cache(destination_folder='data/main/output/calibration', filename='calibration.pickle', enabled=True)
def calibrate_camera(chessboard_images, num_of_corners, image_size=None):
    num_of_images = len(chessboard_images)
    nx = num_of_corners[0]
    ny = num_of_corners[1]

    if num_of_images == 0:
        raise Exception('No images are loaded for calibration. Provide at least 20 images of a chessboard under '
                        'different angles')
    if num_of_images < 20:
        print(f'{num_of_images} images are loaded. Consider using at least 20 images')
    else:
        print(f'{num_of_images} images are loaded')

    objpoints = []  # 3D points in real world space
    imgpoints = []  # 2D points in image plane

    objpoints_per_image = np.zeros((nx * ny, 3), np.float32)
    objpoints_per_image[:, :2] = np.mgrid[0:nx, 0: ny].T.reshape(-1, 2)

    for filename in chessboard_images:
        img = cv2.imread(filename)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # cv2.imread returns BGR image
        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
        if ret is True:
            imgpoints.append(corners)
            objpoints.append(objpoints_per_image)

    if image_size is not None:
        size = image_size
    else:
        img = cv2.imread(chessboard_images[0])
        size = (img.shape[1], img.shape[0])

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, size, None, None)

    return {"ret": ret, "mtx": mtx, "dist": dist, "rvecs": rvecs, "tvecs": tvecs}
