import logging
import random
import tensorflow as tf
from six.moves import xrange
from tensorflow.keras.applications.resnet import ResNet50
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import MaxPool2D, Input, Flatten, Dropout, Dense, Conv2D, Reshape

from src import config
from src.config import Config, Hyperparams

logging.info(tf.__version__)


def get_cnn_with(hparams, seed,
                 hidden_activation,
                 output_activation,
                 padding,
                 input_shape, number_of_classes, hyperparams: Hyperparams):
    """Create a Keras model with the given hyperparameters.

    Args:
      hparams: A dict mapping hyperparameters in `HPARAMS` to values.
      seed: A hashable object to be used as a random seed (e.g., to
        construct dropout layers in the model).

    Returns:
      A compiled Keras model.
    """
    rng = random.Random(seed)

    model = tf.keras.models.Sequential()
    # Add convolutional layers.
    conv_filters = 8
    for index, _ in enumerate(xrange(hparams[hyperparams.HP_CONV_LAYERS])):
        logging.info(f"index: {index}")
        if index == 0:
            model.add(tf.keras.layers.Conv2D(
                filters=conv_filters,
                kernel_size=hparams[hyperparams.HP_CONV_KERNEL_SIZE],
                padding=padding,
                activation=hidden_activation,
                input_shape=input_shape
            ))
        else:
            model.add(tf.keras.layers.Conv2D(
                filters=conv_filters,
                kernel_size=hparams[hyperparams.HP_CONV_KERNEL_SIZE],
                padding=padding,
                activation=hidden_activation,
            ))
        model.add(tf.keras.layers.MaxPool2D(pool_size=2, padding=padding))
        conv_filters *= 2

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dropout(hparams[hyperparams.HP_DROPOUT], seed=rng.random()))

    # Add fully connected layers.
    dense_neurons = 32
    for _ in xrange(hparams[hyperparams.HP_DENSE_LAYERS]):
        model.add(tf.keras.layers.Dense(dense_neurons, activation=hidden_activation))
        dense_neurons *= 2

    # Add the final output layer.
    model.add(tf.keras.layers.Dense(number_of_classes, activation=output_activation))
    return model


def get_cnn(hidden_activation, output_activation, input_shape, number_of_classes):
    return Sequential([
        Conv2D(16, 3, padding='same', activation=hidden_activation, input_shape=input_shape),
        MaxPool2D(pool_size=3),
        Conv2D(32, 3, padding='same', activation=hidden_activation),
        MaxPool2D(pool_size=3),
        Conv2D(64, 3, padding='same', activation=hidden_activation),
        MaxPool2D(pool_size=3),
        Flatten(),
        Dense(512, activation=hidden_activation),
        Dense(number_of_classes, activation=output_activation)
    ])


def get_resnet50(hidden_activation, output_activation, input_shape, number_of_classes):
    resnet_include_top = True,
    resnet_weights = 'imagenet'
    resnet50 = ResNet50(input_shape=input_shape, include_top=resnet_include_top, weights=resnet_weights)
    resnet50.summary()

    return Sequential([
        resnet50,
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation=hidden_activation),
        tf.keras.layers.Dense(number_of_classes, activation=output_activation)
    ])
