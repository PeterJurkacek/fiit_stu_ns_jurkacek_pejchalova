from __future__ import absolute_import, division, print_function, unicode_literals

from pathlib import Path

import tensorflow as tf
# Umozni pracovat s vlastnymi modulmi
import logging

from src.models.custom_model import ConvolutionalNeuralNetwork
from src.logger import Logger
from src.data.load_dataset import ImageDataLoader
from src.utils import calculate_steps_per_epoch


class Trainer:
    def __init__(self,
                 loader: ImageDataLoader,
                 logger: Logger,
                 model,
                 learning_rate=0.001,
                 loss='sparse_categorical_crossentropy',
                 epochs=7,
                 metrics=['accuracy']):
        self.logger = logger
        self.epochs = epochs
        logging.info(f"epochs:{self.epochs}")
        self.loader = loader
        self.model = self.compile(model, learning_rate, loss, metrics)

    def compile(self, model, learning_rate, loss, metrics):
        logging.debug(f"model.compile(model)")
        logging.info(f"learning_rate:{learning_rate}")
        logging.info(f"loss:{loss}")
        logging.info(f"metrics:{metrics}")
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                      loss=loss,
                      metrics=metrics)
        # tf.keras.metrics.Precision(),
        # tf.keras.metrics.Recall()])
        return model

    def train(self, run_id):
        logging.info(f"run_id: {run_id}")
        logging.debug("model.fit")
        history = self.model.fit(self.loader.load_train_dataset(),
                                 steps_per_epoch=self.steps_per_epoch_train(),
                                 epochs=self.epochs,
                                 validation_data=self.loader.load_test_dataset(),
                                 validation_steps=self.steps_per_epoch_validate(),
                                 callbacks=self.logger.get_callbacks('train', run_id))
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
        logging.debug("model.evaluate")
        model.evaluate(self.loader.load_test_dataset(),
                       steps=self.loader.test_data_count,
                       callbacks=self.logger.get_callbacks('evaluate', run_id))
        self.model = model

    def steps_per_epoch_train(self):
        return calculate_steps_per_epoch(self.loader.train_data_count, self.loader.batch_size)

    def steps_per_epoch_validate(self):
        return calculate_steps_per_epoch(self.loader.test_data_count, self.loader.batch_size)
