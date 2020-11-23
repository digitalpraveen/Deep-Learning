import cv2 as cv
import numpy as np

img = cv.imread("/home/praveen/Desktop/Python/Deep Learning/Open CV/Resources/Photos/cats.jpg")
cv.imshow('Cats',img)

gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
cv.imshow("cats",gray)

#laplacian
lap = cv.Laplacian(gray,cv.CV_64F)
lap = np.uint8(np.absolute(lap))
cv.imshow('Laplacian',lap)

#Sobel
sobelx = cv.Sobel(gray,cv.CV_64F,1,0)
sobely = cv.Sobel(gray,cv.CV_64F,0,1)
combined = cv.bitwise_or(sobelx,sobely)

cv.imshow('sobelx',sobelx)
cv.imshow('sobely',sobely)
cv.imshow('combined sobel',combined)

canny = cv.Canny(gray,150,175)
cv.imshow('canny',canny)


cv.waitKey(0)