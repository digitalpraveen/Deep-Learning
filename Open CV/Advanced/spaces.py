import cv2 as cv
import matplotlib.pyplot as plt

img =cv.imread("/home/praveen/Desktop/Python/Deep Learning/Open CV/Resources/Photos/park.jpg")
cv.imshow('boston',img)

#bgr to grayscale
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
cv.imshow('grey',gray)

#bgr to HSV
hsv= cv.cvtColor(img,cv.COLOR_BGR2HSV)
cv.imshow("HSV",hsv)

#bgr to L*A*B
lab = cv.cvtColor(img,cv.COLOR_BGR2LAB)
cv.imshow('LAB',lab)

#BGR to RGB
rgb = cv.cvtColor(img,cv.COLOR_BGR2RGB)
cv.imshow("RGB",rgb)

#HSV to BGR
hsv_bgr = cv.cvtColor(hsv,cv.COLOR_HSV2BGR)
cv.imshow("bgr",hsv_bgr)

cv.waitKey(0)