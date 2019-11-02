import os
from pathlib import Path

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from utils import plot_stroke
from models.base_rnn import BaseRNN


class ConditionalRNN(BaseRNN):
    def __init__(self, *args, **kwargs):
        super().__init__("models/weights/conditional.h5", *args, **kwargs)
        self.gaussian_functions = 10

    def build_model(self, batch_size):
        self.inputs = tf.keras.Input(shape=(None, self.input_size), batch_size=batch_size)
        self.inputs = tf.keras.Input(shape=(None, self.input_size), batch_size=batch_size)

        rnns = []
        lstm_states = []
        x = self.lstm_layer(self.inputs, (batch_size, None, self.input_size))
        window = tf.keras.layers.Dense(self.num_cells,
                                       3 * self.gaussian_functions)()

        rnns.append(x)
        for i in range(self.num_layers - 1):
            output_rnn = tf.keras.layers.concatenate([self.inputs, x])
            x = self.lstm_layer(output_rnn, (None, self.num_cells + self.input_size))
            rnns.append(x)

        # two-dimensional mean and standard deviation, scalar correlation, weights
        params_per_mixture = 6
        output_rnn = tf.keras.layers.concatenate([rnn for rnn in rnns]) # output skip connections
        self.outputs = tf.keras.layers.Dense(params_per_mixture * self.num_mixtures + 1,
                                             input_shape=(self.num_layers * self.num_cells,))(output_rnn)

        self.model = tf.keras.Model(inputs=self.inputs, outputs=self.outputs)
        self.load()

    def loss(self, x, y, input_end, mean1, mean2, stddev1, stddev2, correl, mixture_weight, end_stroke):
        pass

    @tf.function
    def train_step(self, inputs):
        pass

    def generate(self, max_timesteps=400, seed=None, filepath='samples/conditional/generated.jpeg'):
        pass