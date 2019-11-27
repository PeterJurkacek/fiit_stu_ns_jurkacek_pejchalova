from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import logging

from src.models.custom_model import ConvolutionalNeuralNetwork
from src.logger import Logger
from src.data.load_dataset import ImageDataLoader
from src.utils import calculate_steps_per_epoch
from src import config


class Trainer:
    def __init__(self, hparams,
                 loader: ImageDataLoader,
                 logger: Logger,
                 model,
                 session_id,
                 epochs):
        self.logger = logger
        self.epochs = epochs
        self.loader = loader
        self.hparams = hparams
        self.model = model
        self.train(session_id)
        # accuracy = self.evaluate(session_id)

    def train(self, run_id):
        logging.info(f"run_id: {run_id}")
        logging.debug("model.fit")
        history = self.model.fit(self.loader.train_ds,
                                 steps_per_epoch=self.steps_per_epoch_train(),
                                 epochs=self.epochs,
                                 validation_data=self.loader.test_ds,
                                 validation_steps=self.steps_per_epoch_validate(),
                                 callbacks=self.logger.get_callbacks_with_hparams(train_or_test='train', run_id=run_id,
                                                                                  hparams=self.hparams))
        # Save the model
        model_path = self.logger.get_model_path(run_id)
        self.model.save(model_path)
        logging.info(f"model_path: {model_path}")

    def evaluate(self, run_id):
        # Recreate the exact same model, including its weights and the optimizer
        model_path = self.logger.get_model_path(run_id)
        model = tf.keras.models.load_model(model_path)
        logging.info(f"model was loaded from{model_path}")

        # Show the model architecture
        model.summary()
        _, accuracy = model.evaluate(self.loader.load_test_dataset(),
                                     steps=self.loader.test_data_count,
                                     callbacks=self.logger.get_callbacks_with_hparams('evaluate', run_id, self.hparams))
        self.model = model
        return accuracy

    def steps_per_epoch_train(self):
        return calculate_steps_per_epoch(self.loader.train_data_count, self.loader.batch_size)

    def steps_per_epoch_validate(self):
        return calculate_steps_per_epoch(self.loader.test_data_count, self.loader.batch_size)
