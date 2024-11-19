import cv2
import numpy as np
from PIL import Image
from panna.util import resize_image
cv2.namedWindow("original")
cv2.namedWindow("generated")
vc = cv2.VideoCapture(0)
rval, frame = vc.read()
image = np.array(resize_image(Image.fromarray(frame), width=256, height=256))
while rval:
    cv2.imshow("original", image)
    cv2.imshow("generated", image)
    rval, frame = vc.read()
    image = np.array(resize_image(Image.fromarray(frame), width=256, height=256))
    key = cv2.waitKey(20)
    if key == 27: # exit on ESC
        break

vc.release()
cv2.destroyWindow("original")
cv2.destroyWindow("generated")