# -*- coding: utf-8 -*-
"""
Created on Sun May 17 21:02:46 2020

@author: andre
"""

# Based on randerson112358 tutorial https://youtu.be/iGWbqhdjf2s

# Description: This program classifies images

import os
import tensorflow
import numpy as np
from skimage.io import imread
from skimage.transform import resize
import cv2
from sklearn import metrics


#Load the data

IMG_WIDTH = 64
IMG_HEIGHT = 64
IMG_CHANNELS = 3

TRAIN_COVID_PATH = '/Users/andre/Desktop/CT_COVID/train/'
TRAIN_NonCOVID_PATH = '/Users/andre/Desktop/CT_NonCOVID/train/'

TEST_COVID_PATH = '/Users/andre/Desktop/CT_COVID/test/'
TEST_NonCOVID_PATH = '/Users/andre/Desktop/CT_NonCOVID/test/'

train_COVID_ids = [f for f in os.listdir(TRAIN_COVID_PATH) if os.path.isfile(os.path.join(TRAIN_COVID_PATH, f))]
for i in range(0,len(train_COVID_ids)):
    train_COVID_ids[i] = TRAIN_COVID_PATH + train_COVID_ids[i]

train_NonCOVID_ids = [f for f in os.listdir(TRAIN_NonCOVID_PATH) if os.path.isfile(os.path.join(TRAIN_NonCOVID_PATH, f))]
for i in range(0,len(train_NonCOVID_ids)):
    train_NonCOVID_ids[i] = TRAIN_NonCOVID_PATH + train_NonCOVID_ids[i]

train_ids = train_COVID_ids + train_NonCOVID_ids

x_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype = np.uint8)

print('Resizing training images and masks')
for n, path in enumerate(train_ids):
    img = imread(path)#[:,:,:IMG_CHANNELS]
    img  = resize(img, (IMG_HEIGHT, IMG_WIDTH, 1), mode = 'constant', preserve_range = True)
    x_train[n] = img # Fill empty X_train with values from img

cont = 0
cont2 = 0
i = i + 1
y_train = []
while cont < len(train_COVID_ids):
    y_train.append([1])
    cont = cont + 1

while cont2 < len(train_NonCOVID_ids):
    y_train.append([0])
    cont2 = cont2 + 1
    

test_COVID_ids = [f for f in os.listdir(TEST_COVID_PATH) if os.path.isfile(os.path.join(TEST_COVID_PATH, f))]
for i in range(0,len(test_COVID_ids)):
    test_COVID_ids[i] = TEST_COVID_PATH + test_COVID_ids[i]

test_NonCOVID_ids = [f for f in os.listdir(TEST_NonCOVID_PATH) if os.path.isfile(os.path.join(TEST_NonCOVID_PATH, f))]
for i in range(0,len(test_NonCOVID_ids)):
    test_NonCOVID_ids[i] = TEST_NonCOVID_PATH + test_NonCOVID_ids[i]

test_ids = test_COVID_ids + test_NonCOVID_ids

x_test = np.zeros((len(test_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype = np.uint8)

print('Resizing testing images')
for n, path in enumerate(test_ids):
    img = imread(path)#[:,:,:IMG_CHANNELS]
    img  = resize(img, (IMG_HEIGHT, IMG_WIDTH, 1), mode = 'constant', preserve_range = True)
    x_test[n] = img # Fill empty X_train with values from img

cont = 0
cont2 = 0
i = i + 1
y_test = []
while cont < len(test_COVID_ids):
    y_test.append([1])
    cont = cont + 1

while cont2 < len(test_NonCOVID_ids):
    y_test.append([0])
    cont2 = cont2 + 1


classification = ['NonCOVID', 'COVID']

y_train_one_hot = tensorflow.keras.utils.to_categorical(y_train)
y_test_one_hot = tensorflow.keras.utils.to_categorical(y_test)

x_train = x_train / 255
x_test = x_test / 255

# Create the architecture
model = tensorflow.keras.models.Sequential()

# First layer: convlutional layer (extract features from the input image)
model.add(tensorflow.keras.layers.Conv2D(64, (5, 5), activation='relu', input_shape=(IMG_WIDTH,IMG_HEIGHT,IMG_CHANNELS)))

# Second layer: pooling layer with a 2x2 pixel filter
model.add(tensorflow.keras.layers.MaxPooling2D(pool_size=(2, 2)))

# Third layer: convolution layer and pooling layer
model.add(tensorflow.keras.layers.Conv2D(128, (5, 5), activation='relu'))
model.add(tensorflow.keras.layers.MaxPooling2D(pool_size=(2, 2)))

# Forth layer: flattening layer (reduce image to linear array)
model.add(tensorflow.keras.layers.Flatten())

# Neural network: first layer has 1000 neurons and the activation function ReLu
model.add(tensorflow.keras.layers.Dense(1000, activation='relu'))

# Fifth layer: drop out of 50%
model.add(tensorflow.keras.layers.Dropout(0.5))

model.add(tensorflow.keras.layers.Dense(500, activation='relu'))

model.add(tensorflow.keras.layers.Dropout(0.5))

# Neural network: first layer has 250 neurons and the activation function ReLu
model.add(tensorflow.keras.layers.Dense(250, activation='relu'))

# Last layer of this neural network with 10 neurons (one for each label) using the softmax function
model.add(tensorflow.keras.layers.Dense(2, activation='softmax'))

# COMPILE THE MODEL
# categorical_crossentropy: for classes greater thatn 2?

model.compile(loss = 'categorical_crossentropy', optimizer = 'adam')

callbacks = [tensorflow.keras.callbacks.EarlyStopping(patience = 10, monitor = 'val_loss'), tensorflow.keras.callbacks.TensorBoard(log_dir = 'logs')]

hist = model.fit(x_train, y_train_one_hot, validation_data = (x_test, y_test_one_hot), batch_size=32, epochs=30, callbacks = callbacks)


#MEASURE ACCURACY
pred = model.predict(x_test)
pred = np.argmax(pred,axis = 1)
y_compare = np.argmax(y_test_one_hot,axis = 1)
score = metrics.accuracy_score(y_compare, pred)
AUC = metrics.roc_auc_score(y_compare, pred)
DS = metrics.f1_score(y_compare, pred)
print("New Final accuracy: {}".format(score))
print("New Final AUC: {}".format(AUC))
print("New Final Dice Score: {}".format(DS))


model.save('CNN_originla_bo.h5')
model.save_weights("CNN_original_weights_bo.h5")
print("Saved model to disk")

#Try the model

new_image = cv2.imread('/Users/andre/Documents/covID/extraCOVID.png')
resized_image = resize(new_image, (IMG_WIDTH,IMG_HEIGHT,IMG_CHANNELS))
predictions = model.predict(np.array( [resized_image] ))

list_index = [0,1]
x = predictions
for i in range(2):
    for j in range(2):
        if x[0][list_index[i]] > x[0][list_index[j]]:
            temp = list_index[i]
            list_index[i] = list_index[j]
            list_index[j] = temp

i=0
for i in range(2):
    print(classification[list_index[i]], ':', round(predictions[0][list_index[i]] * 100, 2), '%')
