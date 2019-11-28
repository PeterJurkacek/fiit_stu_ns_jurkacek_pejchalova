from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import logging
from src.logger import Logger
from src.data.load_dataset import ImageDataLoader
from src.utils import calculate_steps_per_epoch, timestamp
from src.config import Config


class Trainer:
    def __init__(self, config: Config, loader: ImageDataLoader, logger: Logger):
        self.config = config
        self.loader = loader
        self.loss = config.loss
        self.epochs = config.epochs
        self.metrics = config.metrics
        self.optimizer = config.optimizer
        self.steps_per_epoch = calculate_steps_per_epoch(loader.train_data_info.count, loader.batch_size)
        self.validation_steps = calculate_steps_per_epoch(loader.test_data_info.count, loader.batch_size)
        self.logger = logger

    def start(self, model, run_id=timestamp()):
        self.compile(model)
        self.train(model=model, train_data=self.loader.train_data, validation_data=self.loader.test_data, run_id=run_id)
        #self.evaluate(run_id=run_id, test_data=self.loader.test_data, steps=self.loader.test_data_info.count)

    def compile(self, model):
        logging.info(f"model.compile()")
        logging.info(f"loss:{self.loss}")
        logging.info(f"metrics:{self.metrics}")
        logging.info(f"optimizer:{self.optimizer}")
        #logging.info(f"learning_rate:{self.learning_rate}")

        model.compile(
            loss=self.loss,
            optimizer = self.optimizer,
            metrics=self.metrics,
        )
        return model

    def train(self, model, train_data, validation_data, run_id):
        logging.info(f"model.fit()")
        logging.info(f"run_id: {run_id}")
        logging.info(f"epochs:{self.epochs}")
        logging.info(f"steps_per_epoch:{self.steps_per_epoch}")
        logging.info(f"validation_steps:{self.validation_steps}")

        log_callbacks = [self.logger.create_tensorboard_callback(run_id),self.logger.create_csv_logger_callback(run_id)]
        history = model.fit(train_data,
                            steps_per_epoch=self.steps_per_epoch,
                            epochs=self.epochs,
                            validation_data=validation_data,
                            validation_steps=self.validation_steps,
                            callbacks=log_callbacks)
        # Save the model
        model_path = self.logger.get_model_path(run_id)
        model.save(model_path)
        logging.info(f"Model {run_id} saved to model_path: {model_path}")

    def evaluate(self, run_id, test_data, steps):
        # Recreate the exact same model, including its weights and the optimizer
        logging.info(f"model.evaluate()")
        logging.info(f"run_id: {run_id}")
        logging.info(f"test_data: {test_data}")
        logging.info(f"steps: {steps}")
        model_path = self.logger.get_model_path(run_id)
        model = tf.keras.models.load_model(model_path)
        logging.info(f"model was loaded from{model_path}")

        # Show the model architecture
        model.summary(print_fn=logging.info)
        model.evaluate(test_data,
                       steps=steps,
                       callbacks=[self.logger.create_tensorboard_callback(run_id=run_id)])
