import os
from abc import ABC, abstractmethod

import numpy as np
import tensorflow as tf


class BaseRNN(ABC):
    def __init__(self, weights_path, num_layers=3, num_mixtures=20, num_cells=400, lr=0.0001, gradient_clip=100):
        self.input_size = 3
        self.num_layers = num_layers
        self.num_mixtures = num_mixtures
        self.num_cells = num_cells
        self.gradient_clip = gradient_clip
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        self.initalizer = tf.keras.initializers.TruncatedNormal(stddev=0.075)
        self.weights_path = weights_path

    @abstractmethod
    def build_model(self):
        pass

    @abstractmethod
    def train_step(self):
        pass

    @abstractmethod
    def generate(self):
        pass

    def lstm_layer(self, layer_inputs, input_shape):
        return tf.keras.layers.LSTM(self.num_cells, input_shape=input_shape, return_sequences=True,
                                    kernel_initializer=self.initalizer,
                                    recurrent_initializer=self.initalizer,
                                    use_bias=True, bias_initializer="zeros")(layer_inputs)

    # expands dims of coordinates for gaussian and loss calculations
    def expand_dims(self, inputs, axis, num_dims):
        return tf.concat([tf.expand_dims(inputs, axis) for _ in range(num_dims)], axis)

    def save(self):
        self.model.save_weights(self.weights_path, overwrite=True)
        print(f"Model weights saved to {self.weights_path}")

    def load(self):
        if os.path.exists(self.weights_path):
            self.model.load_weights(self.weights_path)
            print(f"Model weights loaded from {self.weights_path}")
        else:
            print("No model weights to load found")
