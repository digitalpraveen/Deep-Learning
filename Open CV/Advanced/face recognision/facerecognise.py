import os
import cv2 as cv
import numpy as np

people =['Ben Afflek','Elton John','Jerry Seinfield','Madonna','Mindy Kaling']

#p=[]
#for i in os.listdir("/home/praveen/Desktop/Python/Deep Learning/Open CV/Resources/Faces/train"):
#    p.append(i)
#print(p)

DIR = r'/home/praveen/Desktop/Python/Deep Learning/Open CV/Resources/Faces/train'



haar_cascade = cv.CascadeClassifier('/home/praveen/Desktop/Python/Deep Learning/Open CV/Advanced/data/haar_face.xml')

feautures=[]
labels=[]
def create_train():
    for person in people:
        path=os.path.join(DIR,person)
        label = people.index(person)

        for img in os.listdir(path):
            img_path = os.path.join(path,img)

        img_array= cv.imread(img_path)
        gray = cv.cvtColor(img_array,cv.COLOR_BGR2GRAY)

        face_rect = haar_cascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=4)
        for (x,y,w,h) in face_rect:
            face_roi=gray[y:y+h,x:x+w]
            feautures.append(face_roi)
            labels.append(label)

create_train()

print(len(feautures))
print(len(labels))

feautures = np.array(feautures,dtype='object')
labels = np.array(labels)

face_recogniser = cv.face.LBPHFaceRecognizer_create()

face_recogniser.train(feautures,labels)

face_recogniser.save('face_trained.yaml')
np.save('feautures.npy',feautures)
np.save('labels.npy',labels)


