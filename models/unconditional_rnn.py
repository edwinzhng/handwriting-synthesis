import os
from pathlib import Path

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from models.base_rnn import BaseRNN
from utils import plot_stroke


class UnconditionalRNN(BaseRNN):
    def __init__(self, *args, **kwargs):
        super().__init__('unconditional', *args, **kwargs)

    def build_model(self, seq_length, sentence_length=None, load_suffix='_best'):
        inputs = tf.keras.Input((seq_length, self.input_size))
        input_h_1 = tf.keras.Input(self.num_cells)
        input_c_1 = tf.keras.Input(self.num_cells)
        input_h_2 = tf.keras.Input(self.num_cells)
        input_c_2 = tf.keras.Input(self.num_cells)
        input_h_3 = tf.keras.Input(self.num_cells)
        input_c_3 = tf.keras.Input(self.num_cells)
        input_states = [input_h_1, input_c_1, input_h_2, input_c_2, input_h_3, input_c_3]

        lstm_1, h_1, c_1 = self.lstm_layer(name='lstm_1')(inputs, initial_state=[input_h_1, input_c_1])

        skip_1 = tf.keras.layers.concatenate([inputs, lstm_1], name='skip_1')
        lstm_2, h_2, c_2 = self.lstm_layer(name='lstm_2')(skip_1, initial_state=[input_h_2, input_c_2])

        skip_2 = tf.keras.layers.concatenate([inputs, lstm_2], name='skip_2')
        lstm_3, h_3, c_3 = self.lstm_layer(name='lstm_3')(skip_2, initial_state=[input_h_3, input_c_3])

        skip_3 = tf.keras.layers.concatenate([lstm_1, lstm_2, lstm_3], name='skip_3')
        outputs = tf.keras.layers.Dense(6 * self.num_mixtures + 1, name='mdn')(skip_3)

        output_states = [h_1, c_1, h_2, c_2, h_3, c_3]
        self.model = tf.keras.Model(inputs=[inputs, input_states], outputs=[outputs, output_states])
        self.load(load_suffix)

    def train_step(self, batch):
        inputs, next_inputs, lengths = batch
        input_states = [tf.zeros((tf.shape(inputs)[0], self.num_cells))] * 2 * self.num_layers

        with tf.GradientTape() as tape:
            tape.watch(inputs)
            tape.watch(next_inputs)

            # create sequence mask
            mask = tf.sequence_mask(lengths, tf.shape(inputs)[1])

            # calculate loss
            outputs, output_states = self.model([inputs, input_states])
            end_stroke, mixture_weight, mean1, mean2, stddev1, stddev2, correl = self.output_vector(outputs)
            input_end_stroke = next_inputs[:,:,0]
            x = next_inputs[:,:,1]
            y = next_inputs[:,:,2]
            loss = self.loss(x, y, input_end_stroke, mean1, mean2, stddev1, stddev2,
                             correl, mixture_weight, end_stroke, mask)

        return self.apply_gradients(loss, tape)

    def generate(self, timesteps=400, seed=None, filepath='samples/unconditional.jpeg'):
        self.build_model(seq_length=1)
        sample = np.zeros((1, timesteps + 1, 3), dtype='float32')
        input_states = [tf.zeros((1, self.num_cells))] * 2 * self.num_layers

        for i in range(timesteps):
            outputs, input_states = self.model([sample[:,i:i+1,:], input_states])
            sample[0,i+1] = self.sample(outputs, seed)
            inputs = outputs

        # remove first zeros
        sample = sample[0,1:]
        plot_stroke(sample, save_name=filepath)
        return sample
