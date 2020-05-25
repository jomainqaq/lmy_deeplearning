import cv2
import numpy as np
import  keras
import os
from keras.models import load_model
import tensorflow as tf
import numpy as np
import keras
config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
 
config.gpu_options.per_process_gpu_memory_fraction = 0.3
tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))


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


def read_image(size = None):
    data_x, data_y = [], []
    for i in range(1,173):
        try:
            im = cv2.imread('jm/s%s.bmp' % str(i))
            #im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            if size is None:
                 size = (100, 100)
            im = resize_without_deformation(im, size)
            data_x.append(np.asarray(im, dtype = np.int8))
            data_y.append(str(int((i-1)/11.0)))
        except IOError as e:
            print(e)
        except:
            print('Unknown Error!')

    return data_x, data_y
    
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import  Dense, Dropout, Flatten
from keras.optimizers import SGD
from keras.models import Sequential
from keras.layers import Activation, Dense
from keras.models import Sequential 


IMAGE_SIZE = 100
raw_images, raw_labels = read_image(size=(IMAGE_SIZE, IMAGE_SIZE))
raw_images, raw_labels = np.asarray(raw_images, dtype = np.float32), np.asarray(raw_labels, dtype = np.int32) #把图像转换为float类型，方便归一化

from keras.utils import np_utils
ont_hot_labels = np_utils.to_categorical(raw_labels)

from sklearn.model_selection import  train_test_split
train_input, valid_input, train_output, valid_output =train_test_split(raw_images, 
                  ont_hot_labels,
                  test_size = 0.3)

train_input /= 255.0
valid_input /= 255.0


face_recognition_model = Sequential()
face_recognition_model.add(Conv2D(32, (3, 3), padding = 'same', input_shape = (IMAGE_SIZE, IMAGE_SIZE, 3))) # 当使用该层作为模型第一层时，需要提供 input_shape 参数 （整数元组，不包含batch_size）
face_recognition_model.add(Activation('relu'))
face_recognition_model.add(Conv2D(32, (3, 3)))
face_recognition_model.add(Activation('relu'))
face_recognition_model.add(MaxPooling2D(pool_size = (2,2))) # strides默认等于pool_size
face_recognition_model.add(Conv2D(64, (3, 3), padding = 'same'))
face_recognition_model.add(Activation('relu'))
face_recognition_model.add(Conv2D(64, (3, 3)))
face_recognition_model.add(Activation('relu'))
face_recognition_model.add(MaxPooling2D(pool_size = (2,2)))
face_recognition_model.add(Dropout(0.25))
face_recognition_model.add(Flatten())
face_recognition_model.add(Dense(512))
face_recognition_model.add(Activation('relu'))
face_recognition_model.add(Dropout(0.25))
face_recognition_model.add(Dense(2))
face_recognition_model.add(Activation('softmax'))




face_recognition_model.summary()

learning_rate = 0.01
decay = 1e-6
momentum = 0.9
nesterov = True
sgd_optimizer = SGD(lr = learning_rate, decay = decay,
                    momentum = momentum, nesterov = nesterov)

face_recognition_model.compile(loss = 'categorical_crossentropy',
                               optimizer = sgd_optimizer,
                               metrics = ['accuracy'])

batch_size = 20 #每批训练数据量的大小
epochs = 30
face_recognition_model.fit(train_input, train_output,
                           epochs = epochs,
                           batch_size = batch_size, 
                           shuffle = True,
                           validation_data = (valid_input, valid_output))
print(face_recognition_model.evaluate(train_input, train_output, verbose=0))
print(face_recognition_model.evaluate(valid_input, valid_output, verbose=0))
MODEL_PATH = 'face_model.h5'
face_recognition_model.save(MODEL_PATH) 
