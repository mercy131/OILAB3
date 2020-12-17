import time

import cv2
import numpy as np

from ImageProceccing.Lab3.task1.task1 import canny

if __name__ == "__main__":
    img = cv2.imread('../images/img.jpg', 1)
    start_time = time.time()
    cv_canny = cv2.Canny(img, 75, 255)
    end = time.time() - start_time
    start_my_time = time.time()
    my_canny = canny(img)
    end_my = time.time() - start_my_time
    print("My canny time = " , end_my)
    print("Opencv canny time = ", end)
    cv2.imshow('cv2 image', cv_canny)
    cv2.imshow('my image', np.uint8(my_canny))
    cv2.waitKey()
