#!/usr/bin/python3
import tensorflow as tf
from src.data.load_dataset import ImageDataLoader
from src.logger import Logger
from src.models.trainer import Trainer
from src.models.keras_models import get_cnn, get_resnet50
from src.utils import timestamp

def main():
    log_id = "log_id"
    logger = Logger(log_id=log_id)
    logger.start()
    loader = ImageDataLoader(batch_size=10, greyscale=False, dataset_name="test_dataset")
    trainer = Trainer(loader=loader, logger=logger, model=get_cnn(loader.get_input_shape(), loader.get_unique_classes(), 'softmax'), epochs=2)

    run_id = f"model_{timestamp()}"
    trainer.train(run_id)
    trainer.evaluate(run_id)
    logger.end()

if __name__ == '__main__':
    main()

