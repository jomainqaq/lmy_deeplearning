import cv2
import numpy as np
import  keras
import os
from keras.models import load_model
import tensorflow as tf
import numpy as np
import keras

from tkinter import *
from PIL import Image, ImageTk

config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
 
config.gpu_options.per_process_gpu_memory_fraction = 0.3
tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))


CASE_PATH = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(CASE_PATH)

face_recognition_model = keras.Sequential()
MODEL_PATH = 'face_model.h5'
face_recognition_model = load_model(MODEL_PATH)

cap = cv2.VideoCapture(0) 
ret, image = cap.read()
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 

faces = faceCascade.detectMultiScale(gray, scaleFactor=1.2,
                                minNeighbors=5, minSize=(30, 30),) 
def resize_without_deformation(image, size = (100, 100)):
    height, width, _ = image.shape
    longest_edge = max(height, width)
    top, bottom, left, right = 0, 0, 0, 0
    if height < longest_edge:
        height_diff = longest_edge - height
        top = int(height_diff / 2)
        bottom = height_diff - top
    elif width < longest_edge:
        width_diff = longest_edge - width
        left = int(width_diff / 2)
        right = width_diff - left

    image_with_border = cv2.copyMakeBorder(image, top , bottom, left, right, cv2.BORDER_CONSTANT, value = [0, 0, 0])

    resized_image = cv2.resize(image_with_border, size)

    return resized_image

for (x, y, width, height) in faces:
    img = image[y:y+height, x:x+width]
    img = resize_without_deformation(img)

    img = img.reshape((1, 100, 100, 3))
    img = np.asarray(img, dtype = np.float32)
    img /= 255.0

    result = face_recognition_model.predict_classes(img)

    cv2.rectangle(image, (x, y), (x + width, y + height), (0, 255, 0), 2)
    font = cv2.FONT_HERSHEY_SIMPLEX
    #if result[0] == 15:
    if True:
        cv2.putText(image, 'Liu', (x, y-2), font, 0.7, (0, 255, 0), 2)
    else:
        cv2.putText(image, 'No.%d' % result[0], (x, y-2), font, 0.7, (0, 255, 0), 2)

cov= cv2.cvtColor(image,cv2.COLOR_RGB2BGR) 


root = Tk()
root.title("OpenCV Win")

img=Image.fromarray(cov)
img=ImageTk.PhotoImage(img)

canvas=Canvas(root,width=800,height=600)
canvas.pack()
canvas.create_image(0,0,anchor=NW,image=img) 
root.mainloop()
#cv2.imshow('', image)
#cv2.waitKey(0)