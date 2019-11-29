from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging

from absl import app

from src.config import get_experiment_0_config
from src.data.load_dataset import ImageDataLoader
from src.logger import Logger
from src.models.keras_models import get_cnn, get_resnet50, get_mobilenet
from src.models.simple_trainer import Trainer
from src.tunning import HyperTrainer


def experiment_0():
    config = get_experiment_0_config()
    config.logger.start()
    loader = ImageDataLoader(config)
    input_shape = config.input_shape
    logging.info(f"input_shape:{input_shape}")
    number_of_classes = len(loader.labbel_mapper.classes_name)
    logging.info(f"number_of_classes:{number_of_classes}")
    trainer = Trainer(config=config, loader=loader)

    trainer.start(model=get_mobilenet(hidden_activation=config.hidden_activation,
                                     output_activation=config.output_activation,
                                     input_shape=input_shape, number_of_classes=number_of_classes),
                  run_id="mobilenet_model")

    trainer.start(model=get_cnn(hidden_activation=config.hidden_activation,
                                output_activation=config.output_activation,
                                input_shape=input_shape, number_of_classes=number_of_classes),
                  run_id="cnn_model")

    trainer = HyperTrainer(config=config, loader=loader)
    trainer.start_tunning()
    config.logger.end()


if __name__ == '__main__':
    app.run(main)
