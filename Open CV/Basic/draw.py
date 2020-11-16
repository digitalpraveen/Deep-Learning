import cv2 as cv
import numpy as np

blank=np.zeros((500,500,3),dtype='uint8')
cv.imshow('blank',blank)

#blank[:]=0,255,0 #color change in image
#cv.imshow('green',blank)

#cv.rectangle(blank,(0,0),(250,250),(0,255,0),thickness=-1)
#cv.imshow("rectangle",blank)

#draw rectangle
cv.rectangle(blank,(0,0),(blank.shape[1]//2,blank.shape[0]//2),(0,255,0),thickness=-1)
cv.imshow("rectangle",blank)

#draw circle
cv.circle(blank,(blank.shape[1]//2,blank.shape[0]//2),40,(0,0,255),thickness=-1)
cv.imshow('circle',blank)

#draw line
cv.line(blank,(100,250),(300,400),(255,255,255))
cv.imshow('line',blank)

#text on an image
cv.putText(blank,'Hello',(255,255),cv.FONT_HERSHEY_TRIPLEX,fontScale=1.0,color=(0,255,0),thickness=2)
cv.imshow('text',blank)

cv.waitKey(0)