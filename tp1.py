#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Aprendizagem Profunda, TP1
"""

import tensorflow as tf
import numpy as np

from tp1_utils import load_data

# this is needed because of pycharm acting up
BatchNormalization = tf.keras.layers.BatchNormalization
Conv2D = tf.keras.layers.Conv2D
MaxPooling2D = tf.keras.layers.MaxPooling2D
Activation = tf.keras.layers.Activation
Flatten = tf.keras.layers.Flatten
Dropout = tf.keras.layers.Dropout
Dense = tf.keras.layers.Dense
Sequential = tf.keras.models.Sequential

INIT_LR = 0.001
NUM_EPOCHS = 250
BATCH_SIZE = 16

all_data = load_data()

train_X = all_data['train_X']
train_X, valid_X = np.split(train_X, [len(train_X)-500])
test_X = all_data['test_X']

train_masks = all_data['train_masks']
train_masks, valid_masks = np.split(train_masks, [len(train_masks)-500])
test_masks = all_data['test_masks']

train_classes = all_data['train_classes']
train_classes, valid_classes = np.split(train_classes, [len(train_classes)-500])
test_classes = all_data['test_classes']

train_labels = all_data['train_labels']
train_labels, valid_labels = np.split(train_labels, [len(train_labels)-500])
test_labels = all_data['test_labels']


# data is already normalized

# at the start was overfitting, added more convolution layers and it helped
# the validation error still jumps around a bit, but it's much better than it used to 

def create_model_1():
    m = Sequential()

    m.add(Conv2D(16, (3, 3), padding='same', input_shape=(64, 64, 3)))
    m.add(Activation('relu'))
    m.add(BatchNormalization())
    m.add(MaxPooling2D(pool_size=(2, 2)))

    m.add(Conv2D(16, (3, 3), padding='same'))
    m.add(Activation('relu'))
    m.add(BatchNormalization())
    m.add(MaxPooling2D(pool_size=(2, 2)))

    m.add(Conv2D(32, (3, 3), padding='same'))
    m.add(Activation('relu'))
    m.add(BatchNormalization())
    m.add(MaxPooling2D(pool_size=(2, 2)))

    m.add(Conv2D(32, (3, 3), padding='same'))
    m.add(Activation('relu'))
    m.add(BatchNormalization())
    m.add(MaxPooling2D(pool_size=(2, 2)))

    m.add(Conv2D(64, (3, 3), padding='same'))
    m.add(Activation('relu'))
    m.add(BatchNormalization())
    m.add(MaxPooling2D(pool_size=(2, 2)))

    m.add(Conv2D(96, (3, 3), padding='same'))
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

    m.add(Dense(10))
    m.add(Activation("softmax"))
    return m


def do_part_1():
    opt = tf.keras.optimizers.Adam(lr=INIT_LR, decay=INIT_LR/NUM_EPOCHS)
    model = create_model_1()
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["categorical_accuracy"])

    history = model.fit(train_X, train_classes, validation_data=(valid_X, valid_classes), batch_size=BATCH_SIZE,
                        epochs=NUM_EPOCHS)
    model.summary()
    

def create_model_2():
    m = Sequential()

    m.add(Conv2D(16, (3, 3), padding='same', input_shape=(64, 64, 3)))
    m.add(Activation('relu'))
    m.add(BatchNormalization())
    m.add(MaxPooling2D(pool_size=(2, 2)))

    m.add(Conv2D(16, (3, 3), padding='same'))
    m.add(Activation('relu'))
    m.add(BatchNormalization())
    m.add(MaxPooling2D(pool_size=(2, 2)))

    m.add(Conv2D(32, (3, 3), padding='same'))
    m.add(Activation('relu'))
    m.add(BatchNormalization())
    m.add(MaxPooling2D(pool_size=(2, 2)))

    m.add(Conv2D(32, (3, 3), padding='same'))
    m.add(Activation('relu'))
    m.add(BatchNormalization())
    m.add(MaxPooling2D(pool_size=(2, 2)))

    m.add(Conv2D(64, (3, 3), padding='same'))
    m.add(Activation('relu'))
    m.add(BatchNormalization())
    m.add(MaxPooling2D(pool_size=(2, 2)))

    m.add(Conv2D(96, (3, 3), padding='same'))
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

    m.add(Dense(10))
    m.add(Activation("sigmoid"))
    return m

    
def do_part_2():
    opt = tf.keras.optimizers.Adam(lr=INIT_LR, decay=INIT_LR/NUM_EPOCHS)
    model = create_model_1()
    model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["binary_accuracy"])

    history = model.fit(train_X, train_labels, validation_data=(valid_X, valid_labels), batch_size=BATCH_SIZE,
                        epochs=NUM_EPOCHS)
    model.summary()
    
do_part_1()
