from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from absl import app, logging

tf.executing_eagerly()

from src.config import get_experiment_1_config, get_experiment_3_config, \
    get_experiment_4_config, get_experiment_5_config
from src.data.load_dataset import ImageDataLoader
from src.logger import Logger
from src.models.keras_models import get_cnn, get_resnet50, get_mobilenet, get_vgg16, get_densenet121
from src.models.simple_trainer import Trainer
from src.tunning import HyperTrainer


def experiment_1(trainable: bool):
    if trainable:
        config = get_experiment_1_config(experiment_name='experiment_1_trainable_true')
    else:
        config = get_experiment_1_config(experiment_name='experiment_1_trainable_false')
    config.logger.start()
    loader = ImageDataLoader(config)
    input_shape = config.input_shape
    logging.info(f"input_shape:{input_shape}")
    number_of_classes = len(loader.labbel_mapper.classes_name)
    logging.info(f"number_of_classes:{number_of_classes}")
    trainer = Trainer(config=config, loader=loader)
    trainer.start(model=get_mobilenet(hidden_activation=config.hidden_activation,
                                      output_activation=config.output_activation,
                                      input_shape=input_shape,
                                      number_of_classes=number_of_classes,
                                      trainable=trainable),
                  run_id="mobilenet_model")

    trainer.start(model=get_vgg16(hidden_activation=config.hidden_activation,
                                  output_activation=config.output_activation,
                                  input_shape=input_shape,
                                  number_of_classes=number_of_classes,
                                  trainable=trainable),
                  run_id="vgg16_model")

    trainer.start(model=get_densenet121(hidden_activation=config.hidden_activation,
                                        output_activation=config.output_activation,
                                        input_shape=input_shape,
                                        number_of_classes=number_of_classes,
                                        trainable=trainable),
                  run_id="densenet121_model")

    trainer.start(model=get_resnet50(hidden_activation=config.hidden_activation,
                                     output_activation=config.output_activation,
                                     input_shape=input_shape,
                                     number_of_classes=number_of_classes,
                                     trainable=trainable),
                  run_id="resnet50_model")

    trainer.start(model=get_cnn(hidden_activation=config.hidden_activation,
                                output_activation=config.output_activation,
                                input_shape=input_shape, number_of_classes=number_of_classes),
                  run_id="cnn_model")

    config.logger.end()


def experiment_3():
    config = get_experiment_3_config()
    config.logger.start()
    loader = ImageDataLoader(config)
    input_shape = config.input_shape
    logging.info(f"input_shape:{input_shape}")
    number_of_classes = len(loader.labbel_mapper.classes_name)
    logging.info(f"number_of_classes:{number_of_classes}")
    trainer = Trainer(config=config, loader=loader)
    trainer.start(model=get_cnn(hidden_activation=config.hidden_activation,
                                output_activation=config.output_activation,
                                input_shape=input_shape, number_of_classes=number_of_classes),
                  run_id="cnn_model")
    config.logger.end()


def experiment_4():
    config = get_experiment_4_config()
    config.logger.start()
    loader = ImageDataLoader(config)
    trainer = HyperTrainer(config=config, loader=loader)
    trainer.start_tunning()
    config.logger.end()


def experiment_5():
    config = get_experiment_5_config()
    logger = Logger(config)
    logger.start()
    loader = ImageDataLoader(config)
    input_shape = config.input_shape
    logging.info(f"input_shape:{input_shape}")
    number_of_classes = len(loader.labbel_mapper.classes_name)
    logging.info(f"number_of_classes:{number_of_classes}")
    trainer = Trainer(config=config, loader=loader)
    trainer.start(model=get_cnn(hidden_activation=config.hidden_activation,
                                output_activation=config.output_activation,
                                input_shape=input_shape, number_of_classes=number_of_classes),
                  run_id="cnn_model_greyscale")
    logger.end()


def main(argv):
    #experiment_1(trainable=False)
    #experiment_3()
    experiment_4()
    #experiment_5()
    # experiment_6()
    #experiment_1(trainable=True)


if __name__ == '__main__':
    app.run(main)
