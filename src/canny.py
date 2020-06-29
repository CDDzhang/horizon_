import cv2
import numpy as np
import src.blur as blur

def img_canny(img):
    img = img
    # img = img[100:1600,375:1500]
    img1 = cv2.resize(img, (640, 480))
    img = img1
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # filter:
    img = blur.blur(img, Blurnum=3)

    ret, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    sure_bg = cv2.dilate(opening, kernel, iterations=5)
    sure_bg = cv2.erode(sure_bg,kernel,iterations=3)

    img = sure_bg
    # canny_extra
    img = cv2.Canny(img, 50, 150)

    return img,img1


