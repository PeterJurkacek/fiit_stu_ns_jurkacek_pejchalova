#!/usr/bin/python3
import tensorflow as tf
import sys

sys.path.append('/labs')
import getopt

from src.data.load_data import DataLoader
from src.logger import Logger
from src.models.train import Trainer

if __name__ == "__main__":
    trainer = Trainer(logger=Logger(), loader=DataLoader(batch_size=64))
    trainer.train()
