import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.python.keras.models import Sequential
from src import config

def get_model():
    resnet50 = ResNet50(input_shape=config.resnet_input_shape, include_top=config.resnet_include_top, weights=config.resnet_weights)
    resnet50.summary()

    return Sequential([
        resnet50,
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation=config.activation_relu),
        tf.keras.layers.Dense(2, activation=config.activation_softmax)
    ])
