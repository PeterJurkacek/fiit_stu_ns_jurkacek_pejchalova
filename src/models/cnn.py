from __future__ import absolute_import, division, print_function
from tensorflow.python.ops import control_flow_util
control_flow_util.ENABLE_CONTROL_FLOW_V2 = True
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dense, MaxPooling2D, Flatten, Input


class ConvolutionalNeuralNetwork(tf.keras.models.Model):

    def __init__(self, input_shape=(224, 224, 3), output_shape=1):
        super(ConvolutionalNeuralNetwork, self).__init__(name='convolutional_neural_network')

        self.model_layers = [
            Input(shape=input_shape),
            Conv2D(
                filters=16,  # Number of neurons
                kernel_size=3,
                padding='same',  # 'same' for zero padding, 'valid' for no padding
                activation='relu'),
            MaxPooling2D(pool_size=(3, 3)),
            Conv2D(
                filters=32,
                kernel_size=3,
                padding='same',
                activation='relu'),
            MaxPooling2D(pool_size=(3, 3)),
            Conv2D(
                filters=64,
                kernel_size=3,
                padding='same',
                activation='relu'),
            MaxPooling2D(pool_size=(3, 3)),
            Flatten(),  # Flatten the sample from (width x height x channel) 3D matrix into a simple array.
            # We need to use it for the dense layer.
            Dense(
                units=512,
                activation='relu'),
            Dense(
                units=output_shape,
                activation='sigmoid')
        ]

    @tf.function
    def call(self, x):
        for layer in self.model_layers:
            x = layer(x)
        return x
