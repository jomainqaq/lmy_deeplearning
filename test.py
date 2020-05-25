#!/usr/bin/python3
import cv2
print("aaa")
faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
cap = cv2.VideoCapture(0)
#ret, image = cap.read()
image_path='photo_org_export1555326159325.jpg'
image = cv2.imread(image_path)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
faces = faceCascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(30, 30),)
for (x, y, width, height) in faces:
    cv2.rectangle(image, (x, y), (x + width, y + height), (0, 255, 0), 2)
cv2.imshow("Face", image)

cv2.waitKey(0)