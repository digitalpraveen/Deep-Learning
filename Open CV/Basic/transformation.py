import cv2 as cv
import numpy as np

img=cv.imread("/home/praveen/Desktop/Python/Deep Learning/Open CV/Resources/Photos/park.jpg")
cv.imshow("park",img)

def translate(img,x,y):
    tranmatt = np.float32([[1,0,x],[0,1,y]])
    dimensions=img.shape[1],img.shape[0]
    return cv.warpAffine(img,tranmatt,dimensions)

# -x --> left
# -y --> up
#  x --> right
#  y --> down

translated=translate(img,100,255)
cv.imshow("tranlate",translated)


translated=translate(img,-100,255)
cv.imshow("tranlate",translated)


#Rotation
def rotate(img,angle,rotPoint=None):
    (height,width)=img.shape[:2]

    if rotPoint is None:
        rotPoint = (width//2,height//2)
    rotMatt = cv.getRotationMatrix2D(rotPoint,angle,1.0)
    dimensions = (width,height)

    return cv.warpAffine(img,rotMatt,dimensions)

rotated = rotate(img,90)
cv.imshow('rotate',rotated)

rotated_rotate = rotate(rotated,-90)
cv.imshow("rotatedrotate",rotated_rotate)

# Resize image
resized =cv.resize(img,(500,500),interpolation=cv.INTER_CUBIC)
cv.imshow("resized",resized)

#flipping
flip = cv.flip(img,0)
cv.imshow("flip",flip)

#crop
cropped =img[200:400,400:500]
cv.imshow('cropped',cropped)


cv.waitKey(0)