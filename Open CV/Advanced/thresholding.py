import cv2 as cv

img = cv.imread("/home/praveen/Desktop/Python/Deep Learning/Open CV/Resources/Photos/cats.jpg")
cv.imshow('cats',img)
gray  = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
cv.imshow('grey',gray)

#simple thrersholding
thresholding,thresh = cv.threshold(gray,150,255,cv.THRESH_BINARY)
cv.imshow("threshold",thresh)

thresholding,thresh_inv = cv.threshold(gray,150,255,cv.THRESH_BINARY_INV)
cv.imshow("threshold",thresh_inv)

#Adaptive thresholding
adaptive_thresh = cv.adaptiveThreshold(gray,255,cv.ADAPTIVE_THRESH_MEAN_C,cv.THRESH_BINARY,11,3)
cv.imshow('Adaptivr thresholding',adaptive_thresh)

cv.waitKey(0)