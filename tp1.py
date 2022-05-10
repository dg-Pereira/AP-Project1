#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Aprendizagem Profunda, TP1
"""

import tensorflow as tf
import numpy as np
from keras import Input

from tp1_utils import load_data, overlay_masks

from datetime import datetime

# we have to do "imports" this way because of pycharm acting up
BatchNormalization = tf.keras.layers.BatchNormalization
Conv2D = tf.keras.layers.Conv2D
MaxPooling2D = tf.keras.layers.MaxPooling2D
Activation = tf.keras.layers.Activation
Flatten: object = tf.keras.layers.Flatten
Dropout = tf.keras.layers.Dropout
Dense = tf.keras.layers.Dense
Sequential = tf.keras.models.Sequential
InverseTimeDecay = tf.keras.optimizers.schedules.InverseTimeDecay
LeakyReLU = tf.keras.layers.LeakyReLU
UpSampling2D = tf.keras.layers.UpSampling2D
Reshape = tf.keras.layers.Reshape
Model = tf.keras.models.Model

np.set_printoptions(threshold=np.inf)

INIT_LR = 0.001
NUM_EPOCHS_MULTICLASS = 100
NUM_EPOCHS_MULTILABEL = 100
NUM_EPOCHS_AUTOENCODER = 20
BATCH_SIZE = 12

all_data = load_data()

train_X = all_data['train_X']
train_X, valid_X = np.split(train_X, [len(train_X) - 500])
test_X = all_data['test_X']

train_classes = all_data['train_classes']
train_classes, valid_classes = np.split(train_classes, [len(train_classes) - 500])
test_classes = all_data['test_classes']

train_labels = all_data['train_labels']
train_labels, valid_labels = np.split(train_labels, [len(train_labels) - 500])
test_labels = all_data['test_labels']

train_masks = all_data['train_masks'][:, :, :, 0].reshape((4000, 64 * 64))
train_masks, valid_masks = np.split(train_masks, [len(train_masks) - 500])
test_masks = all_data['test_masks'][:, :, :, 0].reshape((500, 64 * 64))


# data is already normalized

# at the start was overfitting, added more convolution layers and it helped
# the validation error still jumps around a bit, but it's much better than it used to


def create_model_1():
    m = Sequential()

    m.add(Conv2D(32, (3, 3), padding='same', input_shape=(64, 64, 3)))
    m.add(Activation('relu'))
    m.add(MaxPooling2D(pool_size=(2, 2)))
    m.add(BatchNormalization())

    m.add(Conv2D(64, (3, 3), padding='same', input_shape=(64, 64, 3)))
    m.add(Activation('relu'))
    m.add(MaxPooling2D(pool_size=(2, 2)))
    m.add(BatchNormalization())

    m.add(Conv2D(64, (3, 3), padding='same'))
    m.add(Activation('relu'))
    m.add(MaxPooling2D(pool_size=(2, 2)))
    m.add(BatchNormalization())

    m.add(Conv2D(96, (3, 3), padding='same'))
    m.add(Activation('relu'))
    m.add(BatchNormalization())

    m.add(Conv2D(96, (3, 3), padding='same'))
    m.add(Activation('relu'))
    m.add(BatchNormalization())

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


def do_part_1():
    learning_rate_fn = InverseTimeDecay(INIT_LR, 1.0, INIT_LR / NUM_EPOCHS_MULTICLASS)
    opt = tf.keras.optimizers.Adam(learning_rate=learning_rate_fn)
    
    model = create_model_1()
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["categorical_accuracy"])

    log_dir = "logs/fit/" + datetime.utcnow().strftime("%Y%m%d%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    history = model.fit(train_X, train_classes, validation_data=(valid_X, valid_classes), batch_size=BATCH_SIZE,
                        epochs=NUM_EPOCHS_MULTICLASS, callbacks=[tensorboard_callback])
    model.summary()


def create_model_2():
    m = Sequential()

    m.add(Conv2D(32, (3, 3), padding='same'))
    m.add(Activation('relu'))
    m.add(MaxPooling2D(pool_size=(2, 2)))
    m.add(BatchNormalization())

    m.add(Conv2D(32, (3, 3), padding='same'))
    m.add(Activation('relu'))
    m.add(MaxPooling2D(pool_size=(2, 2)))
    m.add(BatchNormalization())

    m.add(Conv2D(64, (3, 3), padding='same'))
    m.add(Activation('relu'))
    m.add(BatchNormalization())

    m.add(Conv2D(96, (3, 3), padding='same'))
    m.add(Activation('relu'))
    m.add(BatchNormalization())

    m.add(Conv2D(96, (3, 3), padding='same'))
    m.add(Activation('relu'))
    m.add(BatchNormalization())

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
    learning_rate_fn = InverseTimeDecay(INIT_LR, 1.0, INIT_LR / NUM_EPOCHS_MULTILABEL)
    opt = tf.keras.optimizers.Adam(learning_rate=learning_rate_fn)
    model = create_model_2()
    model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["binary_accuracy"])

    log_dir = "logs/fit/" + datetime.utcnow().strftime("%Y%m%d%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)


    history = model.fit(train_X, train_labels, validation_data=(valid_X, valid_labels), batch_size=BATCH_SIZE,
                        epochs=NUM_EPOCHS_MULTILABEL, callbacks = [tensorboard_callback])
    model.summary()

'''
def create_model_3_v1():
    m = Sequential()

    m.add(Conv2D(16, (3, 3), padding='same', input_shape=(64, 64, 3)))
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

    m.add(Flatten())

    m.add(Dense(64 * 64))
    m.add(Activation("sigmoid"))
    return m


def do_part_3_v1():
    learning_rate_fn = InverseTimeDecay(INIT_LR, 1.0, INIT_LR / NUM_EPOCHS_AUTOENCODER)
    opt = tf.keras.optimizers.Adam(learning_rate=learning_rate_fn)
    model = create_model_3_v1()
    model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["binary_accuracy"])

    history = model.fit(train_X, train_masks, validation_data=(valid_X, valid_masks), batch_size=BATCH_SIZE,
                        epochs=NUM_EPOCHS_AUTOENCODER)

    model.summary()

    predicts = model.predict(test_X).reshape(500, 64, 64, 1)
    overlay_masks('test_overlay.png', test_X, predicts)
    
'''

'''
def create_model_3_v2():
    inputs = Input(shape=(64, 64, 3), name='inputs')

    layer = Conv2D(32, (3, 3), padding="same", input_shape=(64, 64, 3))(inputs)
    layer = Activation("relu")(layer)
    layer = BatchNormalization(axis=-1)(layer)
    layer = MaxPooling2D(pool_size=(2, 2))(layer)

    layer = Conv2D(8, (3, 3), padding="same")(layer)
    layer = Activation("relu")(layer)
    layer = BatchNormalization(axis=-1)(layer)

    layer = Conv2D(8, (3, 3), padding="same")(layer)
    layer = Activation("relu")(layer)
    layer = BatchNormalization(axis=-1)(layer)

    layer = Conv2D(16, (3, 3), padding="same")(layer)
    layer = Activation("relu")(layer)
    layer = BatchNormalization(axis=-1)(layer)

    layer = UpSampling2D(size=(2, 2))(layer)

    layer = Conv2D(32, (3, 3), padding="same")(layer)
    layer = Activation("relu")(layer)
    layer = BatchNormalization(axis=-1)(layer)

    layer = Conv2D(1, (3, 3), padding="same")(layer)
    layer = Activation("sigmoid")(layer)

    autoencoder = Model(inputs=inputs, outputs=layer)
    return autoencoder


def do_part_3_v2():
    ae = create_model_3_v2()
    ae.compile(loss="binary_crossentropy", optimizer="adam", metrics=["binary_accuracy"])
    ae.summary()
    ae.fit(train_X, train_masks.reshape((3500, 64, 64, 1)),
           validation_data=(valid_X, valid_masks.reshape((500, 64, 64, 1))), batch_size=BATCH_SIZE,
           epochs=NUM_EPOCHS_AUTOENCODER)

    predicts = ae.predict(test_X).reshape(500, 64, 64, 1)
    overlay_masks('test_v2_overlay.png', test_X, predicts)
'''

def create_model_3_v3():
    inputs = Input(shape=(64, 64, 3), name='inputs')
    layer = Conv2D(4, (3, 3), padding="same", input_shape=(64, 64, 3))(inputs)
    layer = Activation("relu")(layer)
    layer = BatchNormalization(axis=-1)(layer)

    layer = Conv2D(8, (3, 3), padding="same")(layer)
    layer = Activation("relu")(layer)
    layer = BatchNormalization(axis=-1)(layer)

    layer = Conv2D(8, (3, 3), padding="same")(layer)
    layer = Activation("relu")(layer)
    layer = BatchNormalization(axis=-1)(layer)

    layer = Conv2D(16, (3, 3), padding="same")(layer)
    layer = Activation("relu")(layer)
    layer = BatchNormalization(axis=-1)(layer)

    layer = Conv2D(32, (3, 3), padding="same")(layer)
    layer = Activation("relu")(layer)
    layer = BatchNormalization(axis=-1)(layer)

    layer = Conv2D(64, (3, 3), padding="same")(layer)
    layer = Activation("relu")(layer)
    layer = BatchNormalization(axis=-1)(layer)

    layer = Conv2D(1, (3, 3), padding="same")(layer)
    layer = Activation("sigmoid")(layer)

    autoencoder = Model(inputs=inputs, outputs=layer)

    return autoencoder


def do_part_3_v3():
    ae = create_model_3_v3()
    ae.compile(loss="binary_crossentropy", optimizer="adam", metrics=["binary_accuracy"])
    ae.summary()

    log_dir = "logs/fit/" + datetime.utcnow().strftime("%Y%m%d%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    ae.fit(train_X, train_masks.reshape((3500, 64, 64, 1)),
           validation_data=(valid_X, valid_masks.reshape((500, 64, 64, 1))), batch_size=BATCH_SIZE,
           epochs=NUM_EPOCHS_AUTOENCODER, callbacks = [tensorboard_callback])

    predicts = ae.predict(test_X).reshape(500, 64, 64, 1)
    overlay_masks('test_v2_overlay.png', test_X, predicts)


do_part_3_v3()
