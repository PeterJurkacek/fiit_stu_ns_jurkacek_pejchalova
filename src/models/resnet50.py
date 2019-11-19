import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.python.keras.models import Sequential


def get_model():
    resnet50 = ResNet50(input_shape=(224, 224, 3), include_top=True, weights='imagenet')
    resnet50.summary()

    return Sequential([
        resnet50,
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(2, activation='softmax')
    ])
