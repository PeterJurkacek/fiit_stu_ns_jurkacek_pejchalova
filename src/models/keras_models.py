import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.applications import ResNet50

def get_cnn(input_shape, classes, activation):
    return Sequential([
        Conv2D(16, 3, padding='same', activation='relu', input_shape=input_shape),
        MaxPooling2D(),
        Conv2D(32, 3, padding='same', activation='relu'),
        MaxPooling2D(),
        Conv2D(64, 3, padding='same', activation='relu'),
        MaxPooling2D(),
        Flatten(),
        Dense(512, activation='relu'),
        #Dropout(0.25),
        Dense(len(classes), activation=activation)
    ])


def get_resnet50(input_shape, classes, activation):
    resnet50 = ResNet50(input_shape=input_shape, include_top=True, weights='imagenet')
    resnet50.summary()

    return Sequential([
        resnet50,
        Flatten(),
        Dense(512, activation='relu'),
        Dense(len(classes), activation=activation)
    ])