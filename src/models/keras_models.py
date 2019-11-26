import random

from tensorboard.plugins.hparams import api as hp
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, Input
from tensorflow_core.python.keras.engine.input_layer import Input

from six.moves import xrange

from src import config
from tensorflow.keras.applications import ResNet50


def get_cnn_with(hparams, seed, input_shape, classes):
    """Create a Keras model with the given hyperparameters.

    Args:
      hparams: A dict mapping hyperparameters in `HPARAMS` to values.
      seed: A hashable object to be used as a random seed (e.g., to
        construct dropout layers in the model).

    Returns:
      A compiled Keras model.
    """
    rng = random.Random(seed)

    model = Sequential()
    # Add convolutional layers.
    conv_filters = 8
    for _ in xrange(hparams[config.HP_CONV_LAYERS]):
        model.add(Conv2D(
            filters=conv_filters,
            kernel_size=hparams[config.HP_CONV_KERNEL_SIZE],
            padding="same",
            activation="relu",
        ))
        model.add(MaxPooling2D(pool_size=config.pool_size2, padding=config.padding_same))
        conv_filters *= 2

    model.add(Flatten())
    model.add(Dropout(hparams[config.HP_DROPOUT], seed=rng.random()))

    # Add fully connected layers.
    dense_neurons = 32
    for _ in xrange(hparams[config.HP_DENSE_LAYERS]):
        model.add(Dense(dense_neurons, activation=config.activation_relu))
        dense_neurons *= 2

    # Add the final output layer.
    model.add(Dense(len(classes), activation=config.activation_softmax))

    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=hparams[config.HP_OPTIMIZER],
        metrics=["accuracy"],
    )
    return model


def get_cnn(input_shape, classes, hparams):
    return Sequential([
        Conv2D(config.filters1, config.kernel_size1, padding=config.padding_same, activation=config.activation_relu,
               input_shape=input_shape),
        MaxPooling2D(),
        Conv2D(config.filters2, config.kernel_size2, padding=config.padding_same, activation=config.activation_relu),
        MaxPooling2D(),
        Conv2D(config.filters3, config.kernel_size3, padding=config.padding_same, activation=config.activation_relu),
        MaxPooling2D(),
        Flatten(),
        Dense(hparams[config.HP_NUM_UNITS], activation=config.activation_relu),
        Dropout(hparams[config.HP_DROPOUT]),
        Dense(len(classes), activation=config.activation_softmax)
    ])


def get_resnet50(input_shape, classes):
    resnet50 = ResNet50(input_shape=input_shape, include_top=config.resnet_include_top, weights=config.resnet_weights)
    resnet50.summary()

    return Sequential([
        resnet50,
        Flatten(),
        Dense(config.units, activation=config.activation_relu),
        Dense(len(classes), activation=config.activation_softmax)
    ])
