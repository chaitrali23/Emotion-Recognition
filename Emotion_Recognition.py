#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 09:23:10 2019

@author: chaitralikshirsagar
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
import h5py

from numpy import array
from numpy import argmax
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

import csv
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,Flatten,Dense,Dropout,Activation
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras.optimizers import Adam

from keras import backend as K
from keras.models import Model
from keras.models import load_model
from keras.utils import plot_model
from sklearn.model_selection import train_test_split

#Load Images
filename1 = './Images_data.hdf5'
f1=h5py.File(filename1,'r')

a = list(f1.keys())[0]
x_data=list(f1[a])
x_data =np.asarray(x_data)

#Load Emotion data
filename2 = './Labels_data.hdf5'
f2=h5py.File(filename2,'r')

b = list(f2.keys())[0]
y_data=list(f2[b])
y_data =np.asarray(y_data)

# Split data into train and validation sets
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.1, random_state=42)

#set certain hyperparameters
batch_size = 32
epochs = 100
dropout_conv=0.25
activation = 'relu'
kernel_regularizer_mlp = l2(7e-4)

#Define model and architecture
model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same',
                 input_shape=x_train.shape[1:]))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(8))
model.add(Activation('softmax'))

# Train and test
optimizer = Adam(decay=1e-5)
loss = 'categorical_crossentropy'
metrics = ['accuracy']
model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

#Data augmentation
datagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True, validation_split = 0.1)

datagen.fit(x_train)

plot_model(model, to_file='Emotion_model.png', show_shapes=True, show_layer_names=False)

model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
                    steps_per_epoch=len(x_train) / 32, epochs=epochs)

model.save('./emotionrec_model_trained50ep.h5')
print("Model is saved!")