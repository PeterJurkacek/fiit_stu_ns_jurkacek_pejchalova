import tensorflow as tf

import sys

print(sys.path)
sys.path.append('/labs')
# Umozni pracovat s vlastnymi modulmi
from src.models.cnn import ConvolutionalNeuralNetwork
from src.logger import Logger
from src.data.load_data import DataLoader
from src.utils import calculate_steps_per_epoch
from src.models.model import get_model
from tensorflow.keras.metrics import binary_accuracy


class Trainer:
    def __init__(self, logger: Logger, loader: DataLoader,
                 learning_rate=0.001,
                 loss='binary_crossentropy',
                 epochs=10):
        self.epochs = epochs
        self.loader = loader
        self.logger = logger
        # self.model = ConvolutionalNeuralNetwork(dim_output=len(self.loader.classes))
        self.model = get_model(image_height=self.loader.IMG_HEIGHT, image_width=self.loader.IMG_WIDTH)
        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                           loss=loss,
                           metrics=['accuracy'])

    def train(self):
        history = self.model.fit_generator(
            self.loader.get_train_data_gen(),
            steps_per_epoch=self.steps_per_epoch_train(),
            epochs=self.epochs,
            validation_data=self.loader.get_test_data_gen(),
            validation_steps=self.steps_per_epoch_validate(),
            callbacks=self.logger.callbacks)

        # Save the model
        self.model.save(self.logger.logs_dir / 'model.h5')

    def steps_per_epoch_train(self):
        return calculate_steps_per_epoch(self.loader.test_data_count, self.loader.BATCH_SIZE)

    def steps_per_epoch_validate(self):
        return calculate_steps_per_epoch(self.loader.test_data_count, self.loader.BATCH_SIZE)
