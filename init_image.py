#!/usr/bin/python3
# # _*_ coding:utf-8 _*_
import cv2
import os
import matplotlib.pyplot as plt

CASE_PATH = "haarcascade_frontalface_default.xml"
RAW_IMAGE_DIR = 'me/'
DATASET_DIR = 'jm/'

face_cascade = cv2.CascadeClassifier(CASE_PATH)

def save_feces(img, name,x, y, width, height):
    image = img[y:y+height, x:x+width]
    cv2.imwrite(name, image)

image_list = os.listdir(RAW_IMAGE_DIR) #列出文件夹下所有的目录与文件
#fuck = 'C:/Users/jomain/Desktop/deeplearning/lmy'
count = 166
for image_path in image_list:
    image = cv2.imread( RAW_IMAGE_DIR + image_path )
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  #  cvtColor(image, image, CV_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray,
                                         scaleFactor=1.2,
                                         minNeighbors=5,
                                         minSize=(5, 5), )

    for (x, y, width, height) in faces:
        save_feces(gray, '%ss%d.bmp' % (DATASET_DIR, count), x, y - 30, width, height+30)
        count += 1
