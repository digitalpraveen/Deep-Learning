import cv2 as cv
capture = cv.VideoCapture("/home/praveen/Desktop/Python/Deep Learning/Open CV/Resources/Videos/dog.mp4")
while True:
    isTrue, frame =capture.read()
    cv.imshow('Video',frame)

    if cv.waitKey(20) & 0xFF==ord('d'):
        break

capture.release()
cv.destroyAllWindows()

