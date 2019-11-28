from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging

from src import tunning
from src.config import get_experiment_0_config, get_experiment_1_2_config, get_experiment_4_config, \
    get_experiment_3_config
from src.data.load_dataset import ImageDataLoader
from src.logger import Logger
from src.models.keras_models import get_cnn, get_cnn_with, get_resnet50
from src.models.simple_trainer import Trainer
from src.tunning import HyperTrainer


def experiment_0():
    config = get_experiment_0_config()
    logger = Logger(config)
    logger.start()
    loader = ImageDataLoader(config)
    input_shape = config.input_shape
    logging.info(f"input_shape:{input_shape}")
    number_of_classes = len(loader.labbel_mapper.labels)
    logging.info(f"number_of_classes:{number_of_classes}")
    trainer = Trainer(config=config, loader=loader, logger=logger)
    trainer.start(model=get_cnn(hidden_activation=config.hidden_activation,
                                output_activation=config.output_activation,
                                input_shape=input_shape, number_of_classes=number_of_classes),
                  run_id="cnn_model")

    trainer.start(model=get_resnet50(hidden_activation=config.hidden_activation,
                                     output_activation=config.output_activation,
                                     input_shape=input_shape, number_of_classes=number_of_classes),
                  run_id="resnet50_model")

    trainer = HyperTrainer(config=config, loader=loader, logger=logger)
    trainer.start_tunning()
    logger.end()


def experiment_1():
    config = get_experiment_1_2_config()
    logger = Logger(config)
    logger.start()
    loader = ImageDataLoader(config)
    input_shape = config.input_shape
    logging.info(f"input_shape:{input_shape}")
    number_of_classes = len(loader.labbel_mapper.labels)
    logging.info(f"number_of_classes:{number_of_classes}")
    trainer = Trainer(config=config, loader=loader, logger=logger)
    trainer.start(model=get_resnet50(hidden_activation=config.hidden_activation,
                                     output_activation=config.output_activation,
                                     input_shape=input_shape, number_of_classes=number_of_classes),
                  run_id="resnet50_model")
    logger.end()


def experiment_2():
    config = get_experiment_1_2_config()
    logger = Logger(config)
    logger.start()
    loader = ImageDataLoader(config)
    input_shape = config.input_shape
    logging.info(f"input_shape:{input_shape}")
    number_of_classes = len(loader.labbel_mapper.labels)
    logging.info(f"number_of_classes:{number_of_classes}")
    trainer = Trainer(config=config, loader=loader, logger=logger)
    trainer.start(model=get_cnn(hidden_activation=config.hidden_activation,
                                output_activation=config.output_activation,
                                input_shape=input_shape, number_of_classes=number_of_classes), run_id="cnn_model")
    logger.end()


def experiment_3():
    config = get_experiment_3_config()
    logger = Logger(config)
    logger.start()
    loader = ImageDataLoader(config)
    input_shape = config.input_shape
    logging.info(f"input_shape:{input_shape}")
    number_of_classes = len(loader.labbel_mapper.labels)
    logging.info(f"number_of_classes:{number_of_classes}")
    trainer = Trainer(config=config, loader=loader, logger=logger)
    trainer.start(model=get_cnn(hidden_activation=config.hidden_activation,
                                output_activation=config.output_activation,
                                input_shape=input_shape, number_of_classes=number_of_classes), run_id="cnn_model")
    logger.end()


def experiment_4():
    config = get_experiment_4_config()
    logger = Logger(config)
    logger.start()
    loader = ImageDataLoader(config)
    trainer = HyperTrainer(config=config, loader=loader, logger=logger)
    trainer.start_tunning()
    logger.end()


if __name__ == '__main__':
    experiment_0()
