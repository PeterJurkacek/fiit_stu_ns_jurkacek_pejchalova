from __future__ import absolute_import, division, print_function
from tensorflow.python.ops import control_flow_util
control_flow_util.ENABLE_CONTROL_FLOW_V2 = True
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dense, MaxPooling2D, Flatten, Input
from src import config

class ConvolutionalNeuralNetwork(tf.keras.models.Model):

    def __init__(self, input_shape=config.input_shape, output_shape=config.output_shape):
        super(ConvolutionalNeuralNetwork, self).__init__(name='convolutional_neural_network')

        self.model_layers = [
            Input(shape=input_shape),
            Conv2D(
                filters=config.filters1,  # Number of neurons
                kernel_size=config.kernel_size1,
                padding=config.padding_same,  # 'same' for zero padding, 'valid' for no padding
                activation=config.activation_relu),
            MaxPooling2D(pool_size=config.pool_size1),
            Conv2D(
                filters=config.filters2,
                kernel_size=config.kernel_size2,
                padding=config.padding_same,
                activation=config.activation_relu),
            MaxPooling2D(pool_size=config.pool_size2),
            Conv2D(
                filters=config.filters3,
                kernel_size=config.kernel_size3,
                padding=config.padding_same,
                activation=config.activation_relu),
            MaxPooling2D(pool_size=config.pool_size3),
            Flatten(),  # Flatten the sample from (width x height x channel) 3D matrix into a simple array.
            # We need to use it for the dense layer.
            Dense(
                units=config.units,
                activation=config.activation_relu),
            Dense(
                units=output_shape,
                activation=config.activation_sigmoid)
        ]

    @tf.function
    def call(self, x):
        for layer in self.model_layers:
            x = layer(x)
        return x
