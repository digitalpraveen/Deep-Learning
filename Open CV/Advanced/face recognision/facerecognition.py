import cv2 as cv
import numpy as np

haar_cascade = cv.CascadeClassifier('/home/praveen/Desktop/Python/Deep Learning/Open CV/Advanced/data/haar_face.xml')

# features = np.load('/home/praveen/Desktop/Python/Deep Learning/Open CV/Advanced/face recognision/feautures.npy')
#labels = np.load('/home/praveen/Desktop/Python/Deep Learning/Open CV/Advanced/face recognision/feautures.npy')

people =['Ben Afflek','Elton John','Jerry Seinfield','Madonna','Mindy Kaling']

face_recogniser = cv.face.LBPHFaceRecognizer_create()
face_recogniser.read('/home/praveen/Desktop/Python/Deep Learning/Open CV/Advanced/face recognision/face_trained.yaml')

img = cv.imread('/home/praveen/Desktop/Python/Deep Learning/Open CV/Resources/Faces/train/Elton John/4.jpg')
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
cv.imshow('person',gray)

face_rect = haar_cascade.detectMultiScale(gray,1.1,4)

for (x,y,w,h) in face_rect:
    face_roi = gray[y:y+h,x:x+w]


    labels,confidence= face_recogniser.predict(face_roi)


    cv.putText(img,str(people[labels]),(20,20),cv.FONT_HERSHEY_COMPLEX,1.0,(0,255,0),thickness=2)

    cv.rectangle(img,(x,y),(x+w,y+h),(0,255,0),thickness=2)

cv.imshow('Detected image',img)

cv.waitKey(0)
