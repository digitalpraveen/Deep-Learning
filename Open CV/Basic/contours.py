import cv2 as cv
import numpy as np

img = cv.imread('/home/praveen/Desktop/Python/Deep Learning/Open CV/Resources/Photos/cats.jpg')
cv.imshow('cats',img)

blank=np.zeros(img.shape,dtype='uint8')
cv.imshow('blank',blank)

gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
cv.imshow('gray',gray)

blur = cv.GaussianBlur(gray,(5,5),cv.BORDER_DEFAULT)
cv.imshow("blur",blur)

canny =cv.Canny(blur,125,175)
cv.imshow("canny edge",canny)

ret,thresh=cv.threshold(gray,125,255,cv.THRESH_BINARY)
cv.imshow("thresh",thresh)

contours,hierchiess=cv.findContours(canny,cv.RETR_LIST,cv.CHAIN_APPROX_SIMPLE) #canny or thresh
print(len(contours))

cv.drawContours(blank,contours,-1,(0,0,255),2)
cv.imshow("contours drawn",blank)


cv.waitKey(0)