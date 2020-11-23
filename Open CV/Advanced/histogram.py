import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

img = cv.imread("/home/praveen/Desktop/Python/Deep Learning/Open CV/Resources/Photos/cats.jpg")
cv.imshow("cats",img)



gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
cv.imshow('gray',gray)

greyhist = cv.calcHist([gray],[0],0,[256],[0,256])

plt.figure()
plt.title("Grayscale Histogram")
plt.plot(greyhist)
plt.xlim([0,256])
plt.show()

cv.waitKey(0)