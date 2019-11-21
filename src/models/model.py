import sys

sys.path.append('/labs')
print(sys.path)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from src import config

def get_model(input_shape):
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
        Dense(1, activation=config.activation_sigmoid)
    ])
