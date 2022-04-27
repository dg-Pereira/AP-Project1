#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Aprendizagem Profunda, TP1
"""

from tensorflow import keras
from tensorflow.keras.optimizers import SGD
from tp1_utils import load_data
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization, Conv2D, MaxPooling2D
from tensorflow.keras.layers import Activation, Flatten, Dropout, Dense

INIT_LR=0.001
NUM_EPOCHS=25
BATCH_SIZE = 64

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

def create_model():
    model = Sequential()
    model.add(Conv2D(32, (3,3), padding="same", input_shape=(64, 64, 3)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Conv2D(64, (3,3), padding='same'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    
    model.add(MaxPooling2D(pool_size=(2,2)))
    
    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    
    model.add(Dropout(0.5))
    model.add(Dense(10))
    model.add(Activation("softmax"))
    return model
    
opt = SGD(lr=INIT_LR, momentum=0.9, decay=INIT_LR/NUM_EPOCHS)
model = create_model()
model.compile(loss="categorical_crossentropy", optimizer = opt, metrics = ["accuracy"])

history = model.fit(train_X, train_classes, validation_data=(test_X, test_classes), batch_size=BATCH_SIZE, epochs=NUM_EPOCHS)

model.summary()