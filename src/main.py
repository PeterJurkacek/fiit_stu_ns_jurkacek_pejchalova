#!/usr/bin/python3
import logging

import tensorflow as tf

from src import config
import src.data.load_dataset as loader
import src.logger as logger
import src.models.simple_trainer as trainer
from src.models.keras_models import get_cnn, get_resnet50
from src.utils import timestamp
import src.tunning as tunning


def main():
    logger.start()
    input_shape = loader.get_input_shape()
    classes = loader.get_unique_classes()
    #trainer.start(model=get_cnn(input_shape=input_shape, classes=classes))
    tunning.start(verbose=True)
    logger.end()


if __name__ == '__main__':
    main()
