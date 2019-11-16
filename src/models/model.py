import datetime
import os
import sys

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense


sys.path.append('/labs')
print(sys.path)
from src.data.load_data import DataLoader
from src.models.cnn import ConvolutionalNeuralNetwork
import src.utils as utils
import numpy as np
import tensorflow as tf
from tensorflow.keras import datasets, layers, models


datagen = DataLoader()
datagen.get_train_data_gen()

model = ConvolutionalNeuralNetwork(dim_output=2)

#
# model = Sequential([
#     Conv2D(16, 3, padding='same', activation='relu', input_shape=(datagen.IMG_HEIGHT, datagen.IMG_WIDTH ,3)),
#     MaxPooling2D(),
#     Conv2D(32, 3, padding='same', activation='relu'),
#     MaxPooling2D(),
#     Conv2D(64, 3, padding='same', activation='relu'),
#     MaxPooling2D(),
#     Flatten(),
#     Dense(512, activation='relu'),
#     Dense(1, activation='sigmoid')
# ])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

history = model.fit_generator(
    datagen.get_train_data_gen(),
    steps_per_epoch=datagen.steps_per_epoch,
    epochs=2,
    validation_data=datagen.get_validation_data_gen(),
    validation_steps=datagen.validation_steps
)