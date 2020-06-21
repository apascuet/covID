# -*- coding: utf-8 -*-
"""
Created on Sun May 31 11:45:09 2020

@author: andre
"""

# =============================================================================
# COVID CT-Images classification with a Residual Neural Network
# =============================================================================

# Code based on: https://pylessons.com/Keras-ResNet-tutorial/

# %% 0. Import libraries

import os
import cv2
import numpy as np
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D
from keras.models import Model
from keras.initializers import glorot_uniform
import keras.backend as K
import tensorflow
from sklearn import metrics

#%% 1. Load the data

ROWS = 64
COLS = 64
CHANNELS = 3
CLASSES = 2

def read_image(file_path):
    img = cv2.imread(file_path, cv2.IMREAD_COLOR)
    return cv2.resize(img, (ROWS, COLS), interpolation = cv2.INTER_CUBIC)

def prepare_data(imagesCOVID, imagesNonCOVID):
    images = imagesCOVID + imagesNonCOVID
    m = len(images)
    X = np.zeros((m, ROWS, COLS, CHANNELS), dtype = np.uint8)
    y = np.zeros((1, m), dtype = np.uint8)
    for i, image_file in enumerate(images):
        X[i,:] = read_image(image_file)
        if image_file in imagesCOVID:
            y[0, i] = 1
        elif image_file in imagesNonCOVID:
            y[0, i] = 0
    return X, y

def convert_to_one_hot(Y, C):
    Y = np.eye(C)[Y.reshape(-1)].T
    return Y

# %% 2. Normalize the dataset and learn about its shapes
TRAIN_COVID_DIR = '/Users/andre/Desktop/CT_COVID/train/'
TRAIN_NonCOVID_DIR = '/Users/andre/Desktop/CT_NonCOVID/train/'

TEST_COVID_DIR = '/Users/andre/Desktop/CT_COVID/test/'
TEST_NonCOVID_DIR = '/Users/andre/Desktop/CT_NonCOVID/test/'

train_COVID_images = [TRAIN_COVID_DIR+i for i in os.listdir(TRAIN_COVID_DIR)]
test_COVID_images = [TEST_COVID_DIR+i for i in os.listdir(TEST_COVID_DIR)]
train_NonCOVID_images = [TRAIN_NonCOVID_DIR+i for i in os.listdir(TRAIN_NonCOVID_DIR)]
test_NonCOVID_images = [TEST_NonCOVID_DIR+i for i in os.listdir(TEST_NonCOVID_DIR)]

train_set_x, train_set_y = prepare_data(train_COVID_images, train_NonCOVID_images)
test_set_x, test_set_y = prepare_data(test_COVID_images, test_NonCOVID_images)

X_train = train_set_x/255
X_test = test_set_x/255

Y_train = convert_to_one_hot(train_set_y, CLASSES).T
Y_test = convert_to_one_hot(test_set_y, CLASSES).T

print("Number of training examples = ", X_train.shape[1])
print("Number of test examples = ", X_test.shape[1])
print("X_train shape: ", X_train.shape)
print("Y_train shape: ", Y_train.shape)
print("X_test shape: ", X_test.shape)
print("Y_test shape: ", Y_test.shape)

# %% 3. Building a Residual Network in Keras

# =============================================================================
# IDENTITY BLOCK
# =============================================================================

def identity_block(X, f, filters, stage, block):
    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    # Retrieve Filters
    F1, F2, F3 = filters

    # Save the input value. We'll need this later to add back to the main path. 
    X_shortcut = X

    # First component of main path
    X = Conv2D(filters = F1, kernel_size = (1, 1), strides = (1,1), padding = 'valid', name = conv_name_base + '2a', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2a')(X)
    X = Activation('relu')(X)

    # Second component of main path
    X = Conv2D(filters = F2, kernel_size = (f, f), strides = (1,1), padding = 'same', name = conv_name_base + '2b', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    # Third component of main path
    X = Conv2D(filters = F3, kernel_size = (1, 1), strides = (1,1), padding = 'valid', name = conv_name_base + '2c', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2c')(X)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)

    return X


tensorflow.compat.v1.reset_default_graph()
with tensorflow.compat.v1.Session() as test:
    A_prev = tensorflow.compat.v1.placeholder("float", [3, 4, 4, 6])
    X = np.random.randn(3, 4, 4, 6)
    A = identity_block(A_prev, f = 2, filters = [2, 4, 6], stage = 1, block = 'a')
    test.run(tensorflow.compat.v1.global_variables_initializer())
    out = test.run([A], feed_dict={A_prev: X, K.learning_phase(): 0})
    print("out = ", out[0][1][1][0])

# =============================================================================
# CONVOLUTIONAL BLOCK
# =============================================================================

def convolutional_block(X, f, filters, stage, block, s = 2):
    # Defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    
    # Retrieve Filters
    F1, F2, F3 = filters
    
    # Save the input value
    X_shortcut = X
    
    # == MAIN PATH == #
    # First component of main path
    X = Conv2D(F1, (1, 1), strides = (s, s), name = conv_name_base + '2a', kernel_initializer = glorot_uniform(seed = 0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2a')(X)
    X = Activation('relu')(X)
    
    # Second component of main path
    X = Conv2D(filters = F2, kernel_size = (f, f), strides = (1, 1), padding = 'same', name = conv_name_base + '2b', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    # Third component of main path
    X = Conv2D(filters = F3, kernel_size = (1, 1), strides = (1, 1), padding = 'valid', name = conv_name_base + '2c', kernel_initializer = glorot_uniform(seed = 0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2c')(X)
    
    # == SHORTCUT PATH == #
    X_shortcut = Conv2D(F3, (1, 1), strides = (s, s), name = conv_name_base + '1', kernel_initializer = glorot_uniform(seed = 0))(X_shortcut)
    X_shortcut = BatchNormalization(axis = 3, name = bn_name_base + '1')(X_shortcut)
    
    # Final step: Add shortcut value to main path, and pass it through a RELU activation
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)
    
    return X

tensorflow.compat.v1.reset_default_graph()
with tensorflow.compat.v1.Session() as test:
    A_prev = tensorflow.compat.v1.placeholder("float", [3, 4, 4, 6])
    X = np.random.randn(3, 4, 4, 6)
    A = convolutional_block(A_prev, f = 2, filters = [2, 4, 6], stage = 1, block = 'a')
    test.run(tensorflow.compat.v1.global_variables_initializer())
    out = test.run([A], feed_dict = {A_prev: X, K.learning_phase(): 0})
    print("out = ", out[0][1][1][0])

# %% 4. ResNet model (50 layers)
def ResNet50(input_shape = (64, 64, 3), classes = 2):
    # Define the input as a tensor with shape input_shape
    X_input = Input(input_shape)
    
    # Zero-Padding
    X = ZeroPadding2D((3, 3))(X_input)
    
    # Stage 1
    X = Conv2D(128, (7, 7), strides = (2, 2), name = 'conv1', kernel_initializer = glorot_uniform(seed = 0))(X)
    X = BatchNormalization(axis = 3, name = 'bn_conv1')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((3, 3), strides = (2, 2))(X)
    
    # Stage 2
    X = convolutional_block(X, f = 3, filters = [128, 128, 512], stage = 2, block = 'a', s = 1)
    X = identity_block(X, 3, [128, 128, 512], stage = 2, block = 'b')
    X = identity_block(X, 3, [128, 128, 512], stage = 2, block = 'c')
    
    # Stage 3
    X = convolutional_block(X, f = 3, filters = [128, 128, 512], stage = 3, block = 'a', s = 2)
    X = identity_block(X, 3, [128, 128, 512], stage = 3, block = 'b')
    X = identity_block(X, 3, [128, 128, 512], stage = 3, block = 'c')
    X = identity_block(X, 3, [128, 128, 512], stage = 3, block = 'd')
    
    # Stage 4
    X = convolutional_block(X, f = 3, filters = [512, 512, 2048], stage = 4, block = 'a', s = 2)
    X = identity_block(X, 3, [512, 512, 2048], stage = 4, block = 'b')
    X = identity_block(X, 3, [512, 512, 2048], stage = 4, block = 'c')
    X = identity_block(X, 3, [512, 512, 2048], stage = 4, block = 'd')
    X = identity_block(X, 3, [512, 512, 2048], stage = 4, block = 'e')
    X = identity_block(X, 3, [512, 512, 2048], stage = 4, block = 'f')
    
    # Stage 5
    X = convolutional_block(X, f = 3, filters = [1024, 1024, 4096], stage = 5, block = 'a', s = 2)
    X = identity_block(X, 3, [1024, 1024, 4096], stage = 5, block = 'b')
    X = identity_block(X, 3, [1024, 1024, 4096], stage = 5, block = 'c')
    
    # Average Pooling
    X = AveragePooling2D((2,2), name = 'avg_pool')(X)
    
    # output layer
    X = Flatten()(X)
    X = Dense(classes, activation = 'softmax', name = 'fc' + str(classes), kernel_initializer = glorot_uniform(seed = 0))(X)
    
    # Create model
    model = Model(inputs = X_input, outputs = X, name = 'ResNet50')
    
    return model

# Build the model's graph:
model = ResNet50(input_shape = (ROWS, COLS, CHANNELS), classes = CLASSES)

# Configure the learning process by compiling the model:
model.compile(optimizer = 'adam', loss = 'categorical_crossentropy')

# Train the model until stabilized with a batch_size of 64:
callbacks = [tensorflow.keras.callbacks.EarlyStopping(patience = 10, monitor = 'val_loss'),
              tensorflow.keras.callbacks.TensorBoard(log_dir = 'logs')]
model.fit(X_train, Y_train, validation_data = (X_test, Y_test), batch_size = 32, epochs = 30, callbacks = callbacks)

# Measure metrics
pred = model.predict(X_test)
pred = np.argmax(pred,axis = 1)
y_compare = np.argmax(Y_test,axis = 1)
score = metrics.accuracy_score(y_compare, pred)
AUC = metrics.roc_auc_score(y_compare, pred)
DS = metrics.f1_score(y_compare, pred)
print("New Final accuracy: {}".format(score))
print("New Final AUC: {}".format(AUC))
print("New Final Dice Score: {}".format(DS))

model.save('ResNet50-V1.h5')
model.save_weights("ResNet50-V1_weights.h5")
print("Model saved to the disk")