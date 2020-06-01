import cv2
import numpy as np
from image_processing.threshold import abs_sobel_thresh, dir_threshold, mag_thresh

# Load in image
image = cv2.imread('data/main/test_images/test5.jpg')

# Create a window
cv2.namedWindow('image')

# create trackbars for color change
cv2.createTrackbar('HMin', 'image', 0, 179, lambda x: None)
cv2.createTrackbar('SMin', 'image', 0, 255, lambda x: None)
cv2.createTrackbar('VMin', 'image', 0, 255, lambda x: None)
cv2.createTrackbar('SobelMin', 'image', 0, 255, lambda x: None)
cv2.createTrackbar('DirectionMin', 'image', 0, 255, lambda x: None)
cv2.createTrackbar('MagnitudeMin', 'image', 0, 255, lambda x: None)

cv2.createTrackbar('HMax', 'image', 0, 179, lambda x: None)
cv2.createTrackbar('SMax', 'image', 0, 255, lambda x: None)
cv2.createTrackbar('VMax', 'image', 0, 255, lambda x: None)
cv2.createTrackbar('SobelMax', 'image', 0, 255, lambda x: None)
cv2.createTrackbar('DirectionMax', 'image', 0, 255, lambda x: None)
cv2.createTrackbar('MagnitudeMax', 'image', 0, 255, lambda x: None)

cv2.setTrackbarPos('HMax', 'image', 179)
cv2.setTrackbarPos('SMax', 'image', 255)
cv2.setTrackbarPos('VMax', 'image', 255)
cv2.setTrackbarPos('SobelMax', 'image', 255)
cv2.setTrackbarPos('DirectionMax', 'image', 255)
cv2.setTrackbarPos('MagnitudeMax', 'image', 255)

wait_time = 33

while True:

    hMin = cv2.getTrackbarPos('HMin', 'image')
    sMin = cv2.getTrackbarPos('SMin', 'image')
    vMin = cv2.getTrackbarPos('VMin', 'image')
    sobelMin = cv2.getTrackbarPos('SobelMin', 'image')
    directionMin = cv2.getTrackbarPos('DirectionMin', 'image')
    magnitudeMin = cv2.getTrackbarPos('MagnitudeMin', 'image')

    hMax = cv2.getTrackbarPos('HMax', 'image')
    sMax = cv2.getTrackbarPos('SMax', 'image')
    vMax = cv2.getTrackbarPos('VMax', 'image')
    sobelMax = cv2.getTrackbarPos('SobelMax', 'image')
    directionMax = cv2.getTrackbarPos('DirectionMax', 'image')
    magnitudeMax = cv2.getTrackbarPos('MagnitudeMax', 'image')

    # Set minimum and max HSV values to display
    lower = np.array([hMin, sMin, vMin])
    upper = np.array([hMax, sMax, vMax])

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    s = hsv[:, :, 1]
    mask = cv2.inRange(hsv, lower, upper)
    sobel_mask = abs_sobel_thresh(s, 'x', 3, (sobelMin, sobelMax)) * 255
    direction_mask = dir_threshold(s, 3, ((np.pi / 2) * directionMin / 255, (np.pi / 2) * directionMax / 255)) * 255
    magnitude_mask = mag_thresh(s, 3, (magnitudeMin, magnitudeMax)) * 255

    mask = cv2.bitwise_and(mask, sobel_mask)
    mask = cv2.bitwise_and(mask, direction_mask)
    mask = cv2.bitwise_and(mask, magnitude_mask)

    output = cv2.bitwise_and(image, image, mask=mask)

    # Display output image
    cv2.imshow('image', output)

    # Wait longer to prevent freeze for videos.
    if cv2.waitKey(wait_time) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
