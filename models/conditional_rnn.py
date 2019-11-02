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
        inputs = tf.keras.Input(shape=(None, self.input_size), batch_size=batch_size)
        window_inputs = tf.keras.Input(shape=(None, self.input_size), batch_size=batch_size)

        lstm_1 = self.lstm_layer((batch_size, None, self.input_size))(inputs)
        skip = tf.keras.layers.concatenate([inputs, lstm_1])
        lstm_2 = self.lstm_layer((None, self.num_cells + self.input_size))(skip)
        skip = tf.keras.layers.concatenate([inputs, lstm_2])
        lstm_3 = self.lstm_layer((None, self.num_cells + self.input_size))(skip)

        skip = tf.keras.layers.concatenate([lstm_1, lstm_2, lstm_3])
        outputs = tf.keras.layers.Dense(self.params_per_mixture * self.num_mixtures + 1,
                                             input_shape=(self.num_layers * self.num_cells,))(skip)

        self.model = tf.keras.Model(inputs=[inputs, window_inputs], outputs=outputs)
        self.load()

    def loss(self, x, y, input_end, mean1, mean2, stddev1, stddev2, correl, mixture_weight, end_stroke):
        pass

    @tf.function
    def train_step(self, inputs):
        pass

    def generate(self, max_timesteps=400, seed=None, filepath='samples/conditional/generated.jpeg'):
        pass