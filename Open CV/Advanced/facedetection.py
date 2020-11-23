import cv2 as cv

img = cv.imread("/home/praveen/Desktop/Python/Deep Learning/Open CV/Resources/Photos/group 1.jpg")
cv.imshow('people',img)

gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
cv.imshow("gray",gray)

haar_cascade = cv.CascadeClassifier('/home/praveen/Desktop/Python/Deep Learning/Open CV/Advanced/data/haar_face.xml')

faces_rect = haar_cascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=1)

print(len(faces_rect))

for (x,y,w,h) in faces_rect:
    cv.rectangle(img,(x,y),(x+w,y+h),(0,255,0),thickness=2)

cv.imshow('Detected faces',img)

cv.waitKey(0)