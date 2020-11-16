import cv2 as cv
def rescaleFrame(frame,scale=0.75): 
    #images,vedio, livescale
    width = int(frame.shape[1]*scale)
    height = int(frame.shape[0]*scale)
    dimensions = (width,height)

    return cv.resize(frame,dimensions,interpolation=cv.INTER_AREA)

def changeRes(width,height):
    #live vedio
    capture.set(3,width)
    capture.set(4,height)


import cv2 as cv
capture = cv.VideoCapture("/home/praveen/Desktop/Python/Deep Learning/Open CV/Resources/Videos/dog.mp4")
while True:
    isTrue, frame =capture.read()  
    frame_resized = rescaleFrame(frame,scale=.20)

    cv.imshow('frame',frame)
    cv.imshow('frame_resized',frame_resized)


    if cv.waitKey(20) & 0xFF==ord('d'):
        break

capture.release()
cv.destroyAllWindows()

