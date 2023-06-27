""""
The models defined in here are the feature
    extraction layers for:
             - LeNet-5,
             - AlexNet,
             - ZFNet,
             - VGGNet
"""
from keras.models import Sequential
from keras.layers import (
    Dense, Activation,
    Dropout, Flatten,
    Conv2D, MaxPooling2D,
    AveragePooling2D,
    BatchNormalization,
    DropoutWrapper)
from keras.optimizers import SGD, Adam
from keras.utils import np_utils


"""
Initial variables

"""
input_shape = (128, 128, 3)
activation = ['tanh', 'relu']
padding = ['same', 'valid']


"""
1./ LeNet-5
"""


def le_net_5_model():

    model = Sequential()

    """
    First, a convolutional layer
    """
    model.add(Conv2D(
        6, (5, 5), strides=(1, 1),
        padding=padding[0],
        activation=activation[0],
        input_shape=input_shape))

    """
    Second, a pooling layer
    """
    model.add(AveragePooling2D(
        pool_size=(2, 2),
        strides=(1, 1)
    ))

    """
    Third, a convolutional layer
    """
    model.add(Conv2D(
        16, (5, 5), strides=(1, 1),
        activation=activation[0],
        padding=padding[0]
    ))

    """
    Fourth, a pooling layer
    """
    model.add(AveragePooling2D(
        pool_size=(2, 2),
        strides=(2, 2),
        padding=padding[0]
    ))

    """
    Fifth, a fully connected convolutional layer
    """
    model.add(Conv2D(
        120, (5, 5), strides=(1, 1),
        activation=activation[0],
        padding=padding[0]
    ))

    """
    Flatten
    """
    model.add(Flatten())

    return model


# model = le_net_5_model()


"""
2./ AlexNet
"""


def alex_net_model():

    model = Sequential()

    """
    1./ First convolutional layer
    """
    model.add(
        Conv2D(
            96, (11, 11),
            strides=(4, 4),
            input_shape=input_shape,
            padding=padding[0],
            activation=activation[1]
        ))

    """
    Normalize and pool
    """
    model.add(BatchNormalization())
    model.add(MaxPooling2D(
        pool_size=(3, 3), strides=(2, 2)))

    """
    2./ Second convolutional layer
    """
    model.add(
        Conv2D(
            256, (5, 5),
            strides=(1, 1),
            padding=padding[0],
            activation=activation[1]
        ))

    """
    Normalize and pool
    """
    model.add(BatchNormalization())
    model.add(MaxPooling2D(
        pool_size=(3, 3), strides=(2, 2)))

    """
    3./ Third convolutional layer
    """
    model.add(
        Conv2D(
            384, (3, 3),
            strides=(1, 1),
            padding=padding[0],
            activation=activation[1]
        ))

    """
    4./ Fourth convolutional layer
    """
    model.add(
        Conv2D(
            384, (3, 3),
            strides=(1, 1),
            padding=padding[0],
            activation=activation[1]
        ))

    """
    5./ Fifth convolutional layer
    """
    model.add(
        Conv2D(
            256, (3, 3),
            strides=(1, 1),
            padding=padding[0],
            activation=activation[1]
        ))

    """
    Normalize and pool
    """
    model.add(BatchNormalization())
    model.add(MaxPooling2D(
        pool_size=(3, 3), strides=(2, 2)))

    """
    Flatten
    """
    model.add(Flatten())

    return model


# model = alex_net_model()


"""
3./ ZFNet
"""


def zf_net_model():

    model = Sequential()

    """
    1./ First convolutional layer
    """
    model.add(
        Conv2D(
            96, (7, 7),
            strides=(2, 2),
            input_shape=input_shape,
            padding=padding[0],
            activation=activation[1]
        ))

    """
    Normalize and pool
    """
    model.add(MaxPooling2D(
        pool_size=(3, 3), strides=(2, 2)))
    model.add(BatchNormalization(axis=1))

    """
    2./ Second convolutional layer
    """
    model.add(
        Conv2D(
            256, (5, 5),
            strides=(2, 2),
            padding=padding[0],
            activation=activation[1]
        ))

    """
    Normalize and pool
    """
    model.add(MaxPooling2D(
        pool_size=(3, 3), strides=(2, 2)))
    model.add(BatchNormalization(axis=1))

    """
    3./ Third convolutional layer and
        batch normalization.
    """
    model.add(
        Conv2D(
            384, (3, 3),
            strides=(1, 1),
            padding=padding[0],
            activation=activation[1]
        ))

    model.add(BatchNormalization(axis=1))

    """
    4./ Fourth convolutional layer and
        batch normalization.
    """
    model.add(
        Conv2D(
            384, (3, 3),
            strides=(1, 1),
            padding=padding[0],
            activation=activation[1]
        ))

    model.add(BatchNormalization(axis=1))

    """
    5./ Fifth convolutional layer 
    """
    model.add(
        Conv2D(
            256, (3, 3),
            strides=(1, 1),
            padding=padding[0],
            activation=activation[1]
        ))

    """
    Normalize and pool
    """
    model.add(MaxPooling2D(
        pool_size=(3, 3), strides=(2, 2)))
    model.add(BatchNormalization(axis=1))

    """
    Flatten
    """
    model.add(Flatten())

    return model


# model = zf_net_model()


"""
4./ VGGNet
"""


def vgg_net_model():

    model = Sequential([

        ######

        Conv2D(
            64, (3, 3),
            input_shape=input_shape,
            padding=padding[0],
            activation=activation[1]
        ),

        ######

        Conv2D(
            64, (3, 3),
            padding=padding[0],
            activation=activation[1]
        ),

        ######

        MaxPooling2D(
            pool_size=(3, 3), strides=(2, 2)),

        ######

        Conv2D(
            128, (3, 3),
            input_shape=input_shape,
            padding=padding[0],
            activation=activation[1]
        ),

        ######

        Conv2D(
            128, (3, 3),
            padding=padding[0],
            activation=activation[1]
        ),

        ######

        MaxPooling2D(
            pool_size=(3, 3), strides=(2, 2)),

        ######

        Conv2D(
            256, (3, 3),
            input_shape=input_shape,
            padding=padding[0],
            activation=activation[1]
        ),

        ######

        Conv2D(
            256, (3, 3),
            padding=padding[0],
            activation=activation[1]
        ),

        ######

        Conv2D(
            256, (3, 3),
            padding=padding[0],
            activation=activation[1]
        ),

        ######

        MaxPooling2D(
            pool_size=(3, 3), strides=(2, 2)),

        ######

        Conv2D(
            512, (3, 3),
            input_shape=input_shape,
            padding=padding[0],
            activation=activation[1]
        ),

        ######

        Conv2D(
            512, (3, 3),
            padding=padding[0],
            activation=activation[1]
        ),

        ######

        Conv2D(
            512, (3, 3),
            padding=padding[0],
            activation=activation[1]
        ),

        ######

        MaxPooling2D(
            pool_size=(3, 3), strides=(2, 2)),

        ######

        Conv2D(
            512, (3, 3),
            input_shape=input_shape,
            padding=padding[0],
            activation=activation[1]
        ),

        ######

        Conv2D(
            512, (3, 3),
            padding=padding[0],
            activation=activation[1]
        ),

        ######

        Conv2D(
            512, (3, 3),
            padding=padding[0],
            activation=activation[1]
        ),

        ######

        MaxPooling2D(
            pool_size=(3, 3), strides=(2, 2)),

        ######

        Flatten()

    ])

    return model


# model = vgg_net_model()


"""
Save extraction layers
"""


from EnowNet.detection import detection_model


leNet_5 = le_net_5_model()
alex_net = alex_net_model()
zf_net = zf_net_model()
vgg_net = vgg_net_model()
detection = detection_model()


def save_models():
    leNet_5.save("more/leNet_5.h5")
    alex_net.save("more/alex_net.h5")
    zf_net.save("more/zf_net.h5")
    vgg_net.save("more/vgg_net.h5")
    detection.save("more/detection.h5")


# save_models()
