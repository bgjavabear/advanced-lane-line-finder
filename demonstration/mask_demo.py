import cv2
from image_processing.mask import get_vertices
import matplotlib.pyplot as plt

img = cv2.imread('../data/main/test_images/test6.jpg')
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
lines = get_vertices(img.shape, 1, 0.1, 0.4)
mask = cv2.polylines(img_rgb, [lines], True, (255, 0, 0), 5)
plt.imshow(mask)
plt.show()
