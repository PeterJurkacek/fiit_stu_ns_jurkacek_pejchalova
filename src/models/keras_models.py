from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from src import config
from tensorflow.keras.applications import ResNet50

def get_cnn(input_shape, classes):
    return Sequential([
        Conv2D(config.filters1, config.kernel_size1, padding=config.padding_same, activation=config.activation_relu, input_shape=input_shape),
        MaxPooling2D(),
        Conv2D(config.filters2, config.kernel_size2, padding=config.padding_same, activation=config.activation_relu),
        MaxPooling2D(),
        Conv2D(config.filters3, config.kernel_size3, padding=config.padding_same, activation=config.activation_relu),
        MaxPooling2D(),
        Flatten(),
        Dense(config.units, activation=config.activation_relu),
        #Dropout(0.25),
        Dense(len(classes), activation=config.activation_softmax)
    ])

def get_resnet50(input_shape, classes):

    resnet50 = ResNet50(input_shape=input_shape, include_top=config.resnet_include_top,weights=config.resnet_weights)
    resnet50.summary()

    return Sequential([
        resnet50,
        Flatten(),
        Dense(config.units, activation=config.activation_relu),
        Dense(len(classes), activation=config.activation_softmax)
    ])