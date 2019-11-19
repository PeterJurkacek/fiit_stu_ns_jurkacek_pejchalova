from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow.keras.models import Sequential
import sys

print(sys.path)
sys.path.append('/labs')
# Umozni pracovat s vlastnymi modulmi
from src.models.cnn import ConvolutionalNeuralNetwork
from src.logger import Logger
from src.data.load_dataset import ImageDataLoader
from src.utils import calculate_steps_per_epoch


class Trainer:
    def __init__(self,
                 loader: ImageDataLoader,
                 model: Sequential,
                 optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                 loss='binary_crossentropy',
                 epochs=7):
        self.loss = loss
        self.epochs = epochs
        self.loader = loader
        self.logger = Logger()
        self.optimizer = optimizer
        # self.model = ConvolutionalNeuralNetwork(dim_output=len(self.loader.classes))
        self.model = self.compile(model)

    def compile(self, model):
        model.compile(optimizer=self.optimizer,
                      loss=self.loss,
                      metrics=['acc',
                               tf.keras.metrics.Precision(),
                               tf.keras.metrics.Recall()])
        return model

    def train(self):
        history = self.model.fit(self.loader.load_train_dataset(),
                                 steps_per_epoch=self.steps_per_epoch_train(),
                                 epochs=self.epochs,
                                 validation_data=self.loader.load_test_dataset(),
                                 validation_steps=self.steps_per_epoch_validate(),
                                 callbacks=self.logger.callbacks)
        # Save the model
        self.model.save(self.logger.model_path)

    def evaluate(self):
        # Recreate the exact same model, including its weights and the optimizer
        model = tf.keras.models.load_model(self.logger.get_model_path())

        # Show the model architecture
        model.summary()
        model.evaluate(self.loader.load_test_dataset(),
                       steps=self.loader.test_data_count,
                       callbacks=self.logger.callbacks)
        self.model = model

    def steps_per_epoch_train(self):
        return calculate_steps_per_epoch(self.loader.train_data_count, self.loader.batch_size)

    def steps_per_epoch_validate(self):
        return calculate_steps_per_epoch(self.loader.test_data_count, self.loader.batch_size)
