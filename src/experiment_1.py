from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging

from src.config import get_experiment_1_config
from src.data.load_dataset import ImageDataLoader
from src.logger import Logger
from src.models.keras_models import get_cnn
from src.models.simple_trainer import Trainer


def main():
    config = get_experiment_1_config()
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
                                input_shape=input_shape, number_of_classes=number_of_classes))
    logger.end()


if __name__ == '__main__':
    main()
