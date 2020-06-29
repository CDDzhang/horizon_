import cv2
from enum import Enum

class Blur(Enum):
    BLUR = 1
    BOXFilter = 2
    Gaussian = 3
    Median = 4

def blur(img,Blurnum):
    if(Blurnum==Blur.BLUR):
        img = cv2.blur(img,(3,3))
    elif(Blurnum==Blur.BOXFilter):
        img = cv2.boxFilter(img,-1,(3,3),normalize=True)
    elif(Blurnum==Blur.Gaussian):
        img = cv2.GaussianBlur(img,(25,25),0)
    elif(Blurnum==Blur.Median):
        img = cv2.medianBlur(img,5)
    return img
