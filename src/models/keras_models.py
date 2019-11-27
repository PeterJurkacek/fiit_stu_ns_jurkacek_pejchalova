import logging
import random

from six.moves import xrange
from tensorflow.keras.applications.resnet import ResNet50
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import MaxPooling2D, Input, Flatten, Dropout, Dense, Conv2D

from src import config


def get_cnn_with(hparams, seed, classes):
    """Create a Keras model with the given hyperparameters.

    Args:
      hparams: A dict mapping hyperparameters in `HPARAMS` to values.
      seed: A hashable object to be used as a random seed (e.g., to
        construct dropout layers in the model).

    Returns:
      A compiled Keras model.
    """
    rng = random.Random(seed)

    logging.info(f"config.image_shape: {config.image_shape}")
    model = Sequential()
    # Add convolutional layers.
    conv_filters = 8
    for index, _ in enumerate(xrange(hparams[config.HP_CONV_LAYERS])):
        logging.info(f"index: {index}")
        if index == 0:
            model.add(Conv2D(
                filters=conv_filters,
                kernel_size=hparams[config.HP_CONV_KERNEL_SIZE],
                padding=config.padding,
                activation=config.hidden_activation,
                input_shape=config.input_shape
            ))
        else:
            model.add(Conv2D(
                filters=conv_filters,
                kernel_size=hparams[config.HP_CONV_KERNEL_SIZE],
                padding=config.padding,
                activation=config.hidden_activation,
            ))
        model.add(MaxPooling2D())
        conv_filters *= 2

    model.add(Flatten())
    model.add(Dropout(hparams[config.HP_DROPOUT], seed=rng.random()))

    # Add fully connected layers.
    dense_neurons = 32
    for _ in xrange(hparams[config.HP_DENSE_LAYERS]):
        model.add(Dense(dense_neurons, activation=config.hidden_activation))
        dense_neurons *= 2

    # Add the final output layer.
    model.add(Dense(len(classes), activation=config.output_activation))

    model.compile(
        loss=config.loss,
        optimizer=hparams[config.HP_OPTIMIZER],
        metrics=config.metrics,
    )
    return model


def get_cnn(input_shape, classes):
    return Sequential([
        Conv2D(config.filters1, config.kernel_size1, padding=config.padding, activation=config.hidden_activation,
               input_shape=input_shape),
        MaxPooling2D(),
        Conv2D(config.filters2, config.kernel_size2, padding=config.padding, activation=config.hidden_activation),
        MaxPooling2D(),
        Conv2D(config.filters3, config.kernel_size3, padding=config.padding, activation=config.hidden_activation),
        MaxPooling2D(),
        Flatten(),
        Dense(config.num_units, activation=config.hidden_activation),
        Dense(len(classes), activation=config.output_activation)
    ])


def get_resnet50(input_shape, classes):
    resnet50 = ResNet50(input_shape=input_shape, include_top=config.resnet_include_top, weights=config.resnet_weights)
    resnet50.summary()

    return Sequential([
        resnet50,
        Flatten(),
        Dense(config.units, activation=config.hidden_activation),
        Dense(len(classes), activation=config.output_activation)
    ])
