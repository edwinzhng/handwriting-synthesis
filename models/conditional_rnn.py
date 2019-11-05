import os
from pathlib import Path

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from utils import plot_stroke
from models.base_rnn import BaseRNN


class LSTMAttentionCell(tf.keras.layers.Layer):
    def __init__(self, num_cells, num_characters, window_gaussians, *args, **kwargs):
        super(LSTMAttentionCell, self).__init__(*args, **kwargs)
        self.lstm = tf.keras.layers.LSTMCell(num_cells + num_characters)
        self.window = tf.keras.layers.Dense(3 * window_gaussians, input_shape=(num_cells,))
        self.num_characters = num_characters
        self.window_gaussians = window_gaussians

        # lstm cell state, lstm recurrent state, prev_kappa, prev_w
        self.state_size = (num_cells, num_cells, window_gaussians, num_characters)
        self.output_size = (num_cells, window_gaussians, num_characters)

    def call(self, inputs, states, constants):
        sentence_inputs, sentence_lengths = constants
        prev_h_lstm, prev_c_lstm, prev_kappa, prev_w = states

        lstm, lstm_states = self.lstm(tf.concat([inputs, prev_w], axis=-1), (prev_h_lstm, prev_c_lstm))
        h_lstm, c_lstm = lstm_states

        window_out = self.window(lstm)
        w, kappa = self.process_window(window_out, sentence_inputs, sentence_lengths, prev_kappa)
        return (lstm, kappa, w), (h_lstm, c_lstm, kappa, w)

    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        if inputs is not None:
            batch_size = tf.shape(inputs)[0]
        h_lstm, c_lstm = self.lstm.get_initial_state(inputs, batch_size, dtype)
        return (h_lstm, c_lstm, tf.zeros((batch_size, self.window_gaussians)),
                tf.zeros((batch_size, self.num_characters)))

    def process_window(self, window_outputs, sentence, sentence_length, prev_kappa):
        alpha_hat, beta_hat, kappa_hat = tf.split(window_outputs, 3, -1)
        mask = tf.sequence_mask(sentence_length)

        # Equations (49-51)
        alpha = tf.math.exp(alpha_hat)
        beta = tf.math.exp(beta_hat)
        kappa = prev_kappa + tf.math.exp(kappa_hat)

        # Equations (46-47)
        u = tf.range(1, tf.shape(sentence)[-2], delta=1.0, dtype=tf.float32)
        u = tf.expand_dims(u, axis=0)
        u = tf.expand_dims(u, axis=-1)
        phi = alpha * tf.math.exp(tf.math.negative(beta) * tf.math.square(kappa - u))
        phi = tf.reduce_sum(phi, axis=-1)
        phi = tf.where(mask, phi, tf.zeros_like(phi))
        w = tf.reduce_sum(phi * sentence, axis=1)
        return w, kappa

class ConditionalRNN(BaseRNN):
    def __init__(self, *args, **kwargs):
        super().__init__('conditional', *args, **kwargs)
        self.num_characters = 57
        self.window_gaussians = 10

    def build_model(self, train, load_suffix='_best'):
        batch_size = None if train else 1
        stateful = False if train else True

        inputs = tf.keras.Input(shape=(None, self.input_size), batch_size=batch_size)
        sentence_inputs = tf.keras.Input(shape=(None, self.num_characters), batch_size=batch_size)
        sentence_lengths = tf.keras.Input(shape=(1), batch_size=batch_size)
        prev_kappa = tf.keras.Input(shape=(None, 10), batch_size=batch_size)
        prev_w = tf.keras.Input(shape=(self.num_characters), batch_size=batch_size)

        attention_out = tf.keras.layers.RNN(
            LSTMAttentionCell(self.num_cells, self.num_characters, self.window_gaussians),
            return_sequences=True,
            return_state=True,
            stateful=stateful)(inputs, constants=[sentence_inputs, sentence_lengths])

        lstm_1, kappa, w = attention_out[0]

        skip = tf.keras.layers.concatenate([inputs, lstm_1, w])
        lstm_2 = self.lstm_layer((None, self.num_cells + self.input_size + self.num_characters), stateful=stateful)(skip)
        skip = tf.keras.layers.concatenate([inputs, lstm_2, w])
        lstm_3 = self.lstm_layer((None, self.num_cells + self.input_size + self.num_characters), stateful=stateful)(skip)
        skip = tf.keras.layers.concatenate([lstm_1, lstm_2, lstm_3])
        outputs = tf.keras.layers.Dense(6 * self.num_mixtures + 1, input_shape=(self.num_layers * self.num_cells,))(skip)

        self.model = tf.keras.Model(inputs=[inputs, sentence_inputs, sentence_lengths, prev_kappa, prev_w],
                                    outputs=[outputs, kappa, w])
        self.load(load_suffix)

    @tf.function
    def train_step(self, batch, update_gradients=True):
        stroke_inputs, stroke_lengths, sentence_inputs, sentence_lengths = batch
        stroke_lengths = tf.cast(stroke_lengths, dtype=tf.float32)
        sentence_lengths = tf.cast(sentence_lengths, dtype=tf.float32)

        with tf.GradientTape() as tape:
            prev_kappa = 0.0
            prev_w = tf.zeros((tf.shape(stroke_inputs)[0], self.num_characters))
            tape.watch(stroke_inputs)
            tape.watch(stroke_lengths)
            tape.watch(sentence_inputs)
            tape.watch(sentence_lengths)

            stroke_mask = tf.expand_dims(tf.sequence_mask(stroke_lengths, tf.shape(stroke_inputs)[1]), -1)

            outputs, kappa, w = self.model((stroke_inputs, sentence_inputs,
                                                sentence_lengths, prev_kappa, prev_w))
            end_stroke, mixture_weight, mean1, mean2, stddev1, stddev2, correl = self.output_vector(outputs)
            input_end_stroke = tf.expand_dims(tf.gather(stroke_inputs, 0, axis=-1), axis=-1)
            x = tf.expand_dims(tf.gather(stroke_inputs, 1, axis=-1), axis=-1)
            y = tf.expand_dims(tf.gather(stroke_inputs, 2, axis=-1), axis=-1)
            loss = self.loss(x, y, input_end_stroke, mean1, mean2, stddev1, stddev2,
                             correl, mixture_weight, end_stroke, stroke_mask)

        trainable_vars = self.model.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        gradients = tf.clip_by_value(gradients, -self.gradient_clip, self.gradient_clip)

        if update_gradients:
            self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        return loss, gradients, w

    def generate(self, text, seed=None, filepath='samples/conditional/generated.jpeg'):
        self.build_model(True)
        sample = np.zeros((1, 800 + 1, 3), dtype='float32')
