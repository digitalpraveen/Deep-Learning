import cv2 as cv

img = cv.imread('/home/praveen/Desktop/Python/Deep Learning/Open CV/Resources/Photos/park.jpg')

#converting to grayscale
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
cv.imshow("cat",img)
cv.imshow('Gray',gray)

#Blur
blur = cv.GaussianBlur(img,(7,7),cv.BORDER_DEFAULT)
cv.imshow("Blur",blur)

#Edge Cascade
canny = cv.Canny(blur,125,175)
cv.imshow('canny',canny)

# dilating the image

dilated = cv.dilate(canny,(7,7),iterations=3)
cv.imshow('dilated',dilated)

#eroding
eroded = cv.erode(dilated,(3,3),iterations=3)
cv.imshow('eroded',eroded)

# Resize
resized = cv.resize(img,(500,500),interpolation=cv.INTER_CUBIC)
cv.imshow('Resized',resized)

#cropping
crop = img[50:200,200:400]
cv.imshow('cropped',crop)


cv.waitKey(0)