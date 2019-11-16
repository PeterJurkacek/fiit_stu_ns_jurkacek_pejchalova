from tensorflow import keras
from tensorflow.keras.layers import Conv2D, Dense, MaxPooling2D, Flatten

class ConvolutionalNeuralNetwork(keras.Model):

    def __init__(self, dim_output):
        super(ConvolutionalNeuralNetwork, self).__init__(name='convolutional_neural_network')

        self.model_layers = [
            Conv2D(
                filters=16,  # Number of neurons
                kernel_size=3,
                padding='same',  # 'same' for zero padding, 'valid' for no padding
                activation='relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(
                filters=32,
                kernel_size=3,
                padding='same',
                activation='relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(
                filters=64,
                kernel_size=3,
                padding='same',
                activation='relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Flatten(),  # Flatten the sample from (width x height x channel) 3D matrix into a simple array.
            # We need to use it for the dense layer.
            Dense(
                units=512,
                activation='relu'),
            Dense(
                units=dim_output,
                activation='softmax')
        ]

    def call(self, x):
        for layer in self.model_layers:
            x = layer(x)
        return x
