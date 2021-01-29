from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.layers.convolutional import Convolution2D, Conv2D
from keras.layers.pooling import MaxPooling2D

def get_basic_model(normalize=False):
    """ Basic model """
    model = Sequential()
    if normalize:
        model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))
        model.add(Flatten())
    else:
        model.add(Flatten(input_shape=(160,320,3)))
    model.add(Dense(1))
    return model

def get_lenet_model(crop=True):
    """ LeNet model """
    model = Sequential()
    model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))
    if crop:
        model.add(Cropping2D(cropping=((70,25),(0,0))))
    model.add(Convolution2D(6,5,5,activation="relu"))
    model.add(MaxPooling2D())
    model.add(Convolution2D(6,5,5,activation="relu"))
    model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dense(120))
    model.add(Dense(84))
    model.add(Dense(1))
    return model

def get_nvidia_model(dropout=0.5):
    model = Sequential()
    model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))
    model.add(Cropping2D(cropping=((70,25),(0,0))))
    model.add(Conv2D(24, (5, 5), activation="relu", strides=(2, 2)))
    model.add(Dropout(dropout))
    model.add(Conv2D(36, (5, 5), activation="relu", strides=(2, 2)))
    model.add(Dropout(dropout))
    model.add(Conv2D(48, (5, 5), activation="relu", strides=(2, 2)))
    model.add(Dropout(dropout))
    model.add(Conv2D(64, (3, 3), activation="relu"))
    model.add(Dropout(dropout))
    model.add(Conv2D(64, (3, 3), activation="relu"))
    model.add(Dropout(dropout))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))
    return model
