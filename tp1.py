#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Aprendizagem Profunda, TP1
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import SGD
from tp1_utils import load_data
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization, Conv2D, MaxPooling2D
from tensorflow.keras.layers import Activation, Flatten, Dropout, Dense

INIT_LR = 0.001
NUM_EPOCHS = 250
BATCH_SIZE = 32

all_data = load_data()

train_X = all_data['train_X']
test_X = all_data['test_X']
train_masks = all_data['train_masks']
test_masks = all_data['test_masks']
train_classes = all_data['train_classes']
train_labels = all_data['train_labels']
test_classes = all_data['test_classes']
test_labels = all_data['test_labels']

print(train_X.shape)
print(train_classes.shape)
print(test_X.shape)
print(test_classes.shape)

#data is already normalized


def create_model():
    m = Sequential()
    
    m.add(Conv2D(32, (3, 3), padding="same", input_shape=(64, 64, 3)))
    m.add(Activation('relu'))
    m.add(BatchNormalization())

    m.add(MaxPooling2D(pool_size=(2, 2)))

    m.add(Conv2D(32, (3, 3), padding='same'))
    m.add(Activation('relu'))
    m.add(BatchNormalization())
    
    m.add(MaxPooling2D(pool_size=(2, 2)))
    
    m.add(Conv2D(64, (3, 3), padding="same", input_shape=(64, 64, 3)))
    m.add(Activation('relu'))
    m.add(BatchNormalization())

    m.add(MaxPooling2D(pool_size=(2, 2)))

    m.add(Conv2D(64, (3, 3), padding='same'))
    m.add(Activation('relu'))
    m.add(BatchNormalization())

    m.add(MaxPooling2D(pool_size=(2, 2)))

    m.add(Flatten())
    m.add(Dense(256))
    m.add(Activation('relu'))

    m.add(BatchNormalization())

    m.add(Dense(128))
    m.add(Activation('relu'))
    m.add(BatchNormalization())

    m.add(Dropout(0.5))
    m.add(Dense(10))
    m.add(Activation("softmax"))
    return m


opt = tf.keras.optimizers.Adam(lr=INIT_LR)
model = create_model()
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

history = model.fit(train_X, train_classes, validation_data=(test_X, test_classes), batch_size=BATCH_SIZE,
                    epochs=NUM_EPOCHS)

model.summary()
