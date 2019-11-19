#!/usr/bin/python3
import tensorflow as tf
import sys
if not '/labs' in sys.path:
    sys.path.append('/labs')
import getopt

from src.data.load_dataset import ImageDataLoader
from src.logger import Logger
from src.models.trainer import Trainer
from src.models.model import get_model
import timeit

if __name__ == "__main__":
    loader = ImageDataLoader(batch_size=64, greyscale=False, dataset_name='DATASET')
    trainer = Trainer(loader, model=get_model(loader.get_input_shape()), epochs=7)
    trainer.train()
    trainer.evaluate()
