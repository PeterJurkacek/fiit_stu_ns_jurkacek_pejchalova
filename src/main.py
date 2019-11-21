#!/usr/bin/python3
import logging

import tensorflow as tf
from src.data.load_dataset import ImageDataLoader
from src.logger import Logger
from src.models.trainer import Trainer
from src.models.keras_models import get_cnn, get_resnet50
from src.utils import timestamp


def main():
    log_name = "experiment_cnn"
    logger = Logger(log_name=log_name)
    logger.start()
    loader = ImageDataLoader(batch_size=10, greyscale=False, dataset_name="DATASET")
    trainer = Trainer(loader=loader, logger=logger,
                      model=get_cnn(loader.get_input_shape(), loader.get_unique_classes()), epochs=2)

    run_id = f"model_{timestamp()}"
    trainer.train(run_id)
    trainer.evaluate(run_id)
    logger.end()


if __name__ == '__main__':
    main()
