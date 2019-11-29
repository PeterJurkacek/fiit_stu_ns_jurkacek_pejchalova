# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Write sample summary data for the hparams plugin.

See also `hparams_minimal_demo.py` in this directory for a demo that
runs much faster, using synthetic data instead of actually training
MNIST models.
"""
import numpy as np
import tensorflow as tf
from absl import logging
import random

from absl import flags
from six.moves import xrange  # pylint: disable=redefined-builtin

from src.data.load_dataset import ImageDataLoader
from src.config import Config
from src.logger import Logger
from src.models.keras_models import get_cnn_with
from src.models.simple_trainer import Trainer


class HyperTrainer(Trainer):
    def __init__(self, config: Config, loader: ImageDataLoader):
        super().__init__(config=config, loader=loader)
        self.input_shape = config.input_shape
        logging.info(f"input_shape:{self.input_shape}")
        self.number_of_classes = len(self.loader.labbel_mapper.classes_name)
        logging.info(f"number_of_classes:{self.number_of_classes}")
        self.hyperparams = config.hyperparams
        self.hidden_activation = config.hidden_activation
        self.output_activation = config.output_activation
        self.padding = config.padding

    def start_tunning(self, verbose=True):
        rng = random.Random(0)
        sessions_per_group = 2
        num_sessions = self.hyperparams.num_session_groups * sessions_per_group
        session_index = 0  # across all session groups
        for group_index in xrange(self.hyperparams.num_session_groups):
            hparams = {h: h.domain.sample_uniform(rng) for h in self.hyperparams.HPARAMS}
            hparams_string = str(hparams)
            for repeat_index in xrange(sessions_per_group):
                session_id = str(session_index)
                session_index += 1
                if verbose:
                    logging.info(f"--- Running training session {session_index}, {num_sessions}")
                    logging.info(hparams_string)
                    logging.info("--- repeat #: %d" % (repeat_index + 1))
                self.run(model=get_cnn_with(hparams, session_id,
                                            hidden_activation=self.hidden_activation,
                                            output_activation=self.output_activation,
                                            input_shape=self.input_shape,
                                            padding=self.padding,
                                            number_of_classes=self.number_of_classes, hyperparams=self.hyperparams),
                         session_id=session_id, hparams=hparams)

    def run(self, model, session_id, hparams):
        self.build(model, hparams)
        self.train_one(model=model, run_id=session_id,
                       train_data=self.loader.train_data, validation_data=self.loader.test_data,
                       hparams=hparams)
        self.evaluate_one(run_id=session_id,
                          test_data=self.loader.test_data,
                          steps=self.loader.test_data_info.count,
                          hparams=hparams)

    def train_one(self, model, train_data, validation_data, run_id, hparams):
        logging.info(f"model.fit()")
        logging.info(f"run_id: {run_id}")
        logging.info(f"epochs:{self.epochs}")
        logging.info(f"steps_per_epoch:{self.steps_per_epoch}")
        logging.info(f"validation_steps:{self.validation_steps}")

        log_callbacks = [self.logger.create_tensorboard_callback(run_id),
                         self.logger.create_csv_logger_callback(run_id),
                         self.logger.create_hp_params_callback(run_id, hparams)]

        training_history = model.fit(train_data,
                                     steps_per_epoch=self.steps_per_epoch,
                                     epochs=self.epochs,
                                     validation_data=validation_data,
                                     validation_steps=self.validation_steps,
                                     callbacks=log_callbacks)

        tf.print("Average test loss: ", np.average(training_history.history['loss']),
                 output_stream=self.logger.log_file_path)
        tf.print("Average test accuracy: ", np.average(training_history.history['accuracy']),
                 output_stream=self.logger.log_file_path)
        # Save the model
        model_path = self.logger.get_model_path(run_id)
        model.save(model_path)
        logging.info(f"Model {run_id} saved to model_path: {model_path}")

    def build(self, model, hparams):
        logging.info(f"model.build()")
        # logging.info(f"learning_rate:{self.learning_rate}")

        model.compile(
            loss=self.loss,
            optimizer=hparams[self.hyperparams.HP_OPTIMIZER],
            metrics=['accuracy'],
        )
        return model

    def evaluate_one(self, run_id, test_data, steps, hparams):
        # Recreate the exact same model, including its weights and the optimizer
        logging.info(f"model.evaluate()")
        logging.info(f"run_id: {run_id}")
        logging.info(f"steps: {steps}")
        model_path = self.logger.get_model_path(run_id)
        model = tf.keras.models.load_model(model_path)
        logging.info(f"model was loaded from{model_path}")

        # Show the model architecture
        model.summary(print_fn=logging.info)
        _, accuracy = model.evaluate(test_data,
                                     steps=steps,
                                     callbacks=[self.logger.create_tensorboard_callback(run_id=run_id)])
        tf.print("accuracy:", accuracy, output_stream=self.logger.log_file_path)
        self.logger.create_scalar_summary(run_id=run_id, accuracy=accuracy, hparams=hparams)
