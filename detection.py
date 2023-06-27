from keras.models import Sequential
from keras.layers import (
    Dense, Activation, Dropout, Flatten,
    Conv2D, MaxPooling2D, AveragePooling2D,
    BatchNormalization, DropoutWrapper)

from keras.optimizers import SGD, Adam
from keras.utils import np_utils


"""
Initial variables

"""
input_shape = (128, 128, 3)
activation = 'sigmoid'
padding = 'same'
kernel_initializer = 'he_uniform'


"""
Detection
"""


def detection_model():

    model = Sequential()

    model.add(Conv2D(
        32, (3, 3),
        activation=activation,
        padding=padding,
        input_shape=input_shape))

    model.add(BatchNormalization())

    model.add(Conv2D(
        32, (3, 3),
        activation=activation,
        padding=padding,
        kernel_initializer=kernel_initializer))

    model.add(BatchNormalization())
    model.add(MaxPooling2D())

    model.add(Conv2D(
        64, (3, 3),
        activation=activation,
        padding=padding,
        kernel_initializer=kernel_initializer))

    model.add(BatchNormalization())

    model.add(Conv2D(
        64, 3, activation=activation,
        padding=padding,
        kernel_initializer=kernel_initializer))

    model.add(BatchNormalization())
    model.add(MaxPooling2D())

    model.add(Flatten())

    return model
