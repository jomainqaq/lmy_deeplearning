#!/usr/bin/python3
# # _*_ coding:utf-8 _*_
import cv2
import os
import matplotlib.pyplot as plt
from PIL import Image
CASE_PATH = "haarcascade_frontalface_default.xml"
RAW_IMAGE_DIR = 'new/'
DATASET_DIR = 'jm/'

face_cascade = cv2.CascadeClassifier(CASE_PATH)

def save_feces(img, name,x, y, width, height):
    image = img[y:y+height, x:x+width]
    cv2.imwrite(name, image)
def fuckface(image,name):
    
    cv2.imwrite(name,image)
image_list = os.listdir(RAW_IMAGE_DIR) #列出文件夹下所有的目录与文件
#fuck = 'C:/Users/jomain/Desktop/deeplearning/lmy'
count = 177
same = 1
for image_path in image_list:
    image = cv2.imread( RAW_IMAGE_DIR + image_path )
    image = cv2.resize(image,(100,100),interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  #  cvtColor(image, image, CV_BGR2GRAY)
    #image = gray[100, 100]
    name =str(DATASET_DIR)+ str(count)
    fuckface(gray, '%ss%d.bmp' % (DATASET_DIR, count))
    count += 1
    same +=1
    if(same==11):
        fuckface(gray, '%ss%d.bmp' % (DATASET_DIR, count))
        count +=1
        same = 1
