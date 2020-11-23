import cv2 as cv

img = cv.imread("/home/praveen/Desktop/Python/Deep Learning/Open CV/Resources/Photos/cats.jpg")
cv.imshow('cat',img)

#Averaging
average = cv.blur(img,(3,3))
cv.imshow('average',average)

#gaussian blur
gauss = cv.GaussianBlur(img,(3,3),sigmaX=0)
cv.imshow("Gaussian blur",gauss)

#median blur
median = cv.medianBlur(img,7)
cv.imshow('Median blur',median)

#bilateral
bilateral = cv.bilateralFilter(img,10,35,25)
cv.imshow("bilateral",bilateral)

cv.waitKey(0)