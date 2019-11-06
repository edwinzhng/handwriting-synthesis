import os
from pathlib import Path

import numpy as np
import tensorflow as tf

from models.base_rnn import BaseRNN
from utils import plot_stroke
from utils.data import char_to_index, one_hot_encode


class LSTMAttentionCell(tf.keras.layers.Layer):
    def __init__(self, num_cells, num_characters, window_gaussians, *args, **kwargs):
        super(LSTMAttentionCell, self).__init__(*args, **kwargs)
        self.lstm = tf.keras.layers.LSTMCell(num_cells + num_characters)
        self.window = tf.keras.layers.Dense(3 * window_gaussians, input_shape=(num_cells,))
        self.num_characters = num_characters
        self.window_gaussians = window_gaussians

        # lstm cell state, lstm recurrent state, prev_kappa, prev_w
        self.state_size = (num_cells, num_cells, window_gaussians, num_characters)
        self.output_size = (num_cells, num_characters, num_characters)

    def call(self, inputs, states, constants, training):
        sentence_inputs, sentence_lengths = constants
        prev_h_lstm, prev_c_lstm, prev_kappa, prev_w = states

        lstm, lstm_states = self.lstm(tf.concat([inputs, prev_w], axis=-1), (prev_h_lstm, prev_c_lstm))
        h_lstm, c_lstm = lstm_states

        window_out = self.window(lstm)
        alpha_hat, beta_hat, kappa_hat = tf.split(window_out, 3, -1)
        mask = tf.cast(tf.sequence_mask(sentence_lengths,
                       tf.shape(sentence_inputs)[1]), dtype=tf.float32)

        # Equations (49-51)
        alpha = tf.expand_dims(tf.math.exp(alpha_hat), -1)
        beta = tf.expand_dims(tf.math.exp(beta_hat), -1)
        kappa = tf.expand_dims(prev_kappa + tf.math.exp(kappa_hat), -1)

        u = tf.range(1, tf.shape(sentence_inputs)[-2] + 1, delta=1.0, dtype=tf.float32)
        u = tf.expand_dims(u, 0)
        u = tf.expand_dims(u, 0)
        u = tf.tile(u, (tf.shape(sentence_inputs)[0], self.window_gaussians, 1))

        # Equations (46-47)
        phi = alpha * tf.math.exp(tf.math.negative(beta) * tf.math.square(kappa - u))
        phi = tf.reduce_sum(phi, axis=1)
        w = tf.reduce_sum(tf.expand_dims(phi, -1) * sentence_inputs, axis=1)
        kappa = tf.squeeze(kappa, -1)
        return (lstm, phi, w), (h_lstm, c_lstm, kappa, w)

    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        if inputs is not None:
            batch_size = tf.shape(inputs)[0]
        h_lstm, c_lstm = self.lstm.get_initial_state(inputs, batch_size, dtype)
        return (h_lstm, c_lstm, tf.zeros((batch_size, self.window_gaussians)),
                tf.zeros((batch_size, self.num_characters)))

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
        sentence_lengths = tf.keras.Input(shape=(1, ), batch_size=batch_size)

        attention_out = tf.keras.layers.RNN(
            LSTMAttentionCell(self.num_cells, self.num_characters, self.window_gaussians),
            return_sequences=True,
            return_state=True,
            stateful=stateful)(inputs, constants=[sentence_inputs, sentence_lengths], training=train)

        lstm_1, phi, w = attention_out[0]

        skip = tf.keras.layers.concatenate([inputs, lstm_1, w])
        lstm_2 = self.lstm_layer((None, self.num_cells + self.input_size + self.num_characters), stateful=stateful)(skip)
        skip = tf.keras.layers.concatenate([inputs, lstm_2, w])
        lstm_3 = self.lstm_layer((None, self.num_cells + self.input_size + self.num_characters), stateful=stateful)(skip)
        skip = tf.keras.layers.concatenate([lstm_1, lstm_2, lstm_3])
        outputs = tf.keras.layers.Dense(6 * self.num_mixtures + 1, input_shape=(self.num_layers * self.num_cells,))(skip)

        self.model = tf.keras.Model(inputs=[inputs, sentence_inputs, sentence_lengths],
                                    outputs=[outputs, phi, w])
        self.load(load_suffix)

    @tf.function
    def train_step(self, batch, update_gradients=True):
        stroke_inputs, stroke_lengths, sentence_inputs, sentence_lengths = batch
        stroke_lengths = tf.cast(stroke_lengths, dtype=tf.float32)
        sentence_lengths = tf.cast(sentence_lengths, dtype=tf.float32)

        with tf.GradientTape() as tape:
            tape.watch(stroke_inputs)
            tape.watch(stroke_lengths)
            tape.watch(sentence_inputs)
            tape.watch(sentence_lengths)

            stroke_mask = tf.expand_dims(tf.sequence_mask(stroke_lengths, tf.shape(stroke_inputs)[1]), -1)

            outputs, phi, w = self.model((stroke_inputs, sentence_inputs, sentence_lengths))
            end_stroke, mixture_weight, mean1, mean2, stddev1, stddev2, correl = self.output_vector(outputs)
            input_end_stroke = tf.expand_dims(tf.gather(stroke_inputs, 0, axis=-1), axis=-1)
            x = tf.expand_dims(tf.gather(stroke_inputs, 1, axis=-1), axis=-1)
            y = tf.expand_dims(tf.gather(stroke_inputs, 2, axis=-1), axis=-1)
            loss = self.loss(x, y, input_end_stroke, mean1, mean2, stddev1, stddev2,
                             correl, mixture_weight, end_stroke, stroke_mask)

        trainable_vars = self.model.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        for i, grad in enumerate(gradients):
            gradients[i] = tf.clip_by_value(gradients[i], -self.gradient_clip, self.gradient_clip)

        if update_gradients:
            self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        return loss, gradients

    def generate(self, text, timesteps=800, seed=None, filepath='samples/conditional/generated.png'):
        self.build_model(True)
        sample = np.zeros((1, timesteps + 1, 3), dtype='float32')
        char_index, _ = char_to_index()

        one_hot_text = tf.expand_dims(one_hot_encode(text, self.num_characters, char_index), 0)
        text_length = tf.expand_dims(len(text), 0)

        for i in range(timesteps):
            outputs, phi, w = self.model((sample[:,i:i+1,:], one_hot_text, text_length))
            sample[0,i+1] = self.sample(outputs, seed)
            inputs = outputs

            # stopping heuristic
            finished = True
            phi_last = phi[0,0,len(text)-1]
            for phi_u in phi[0,0,:len(text)-1]:
                if phi_u.numpy() >= phi_last.numpy():
                    finished = False
                    break

            if finished:
                break

        # remove first zeros and discard unused timesteps
        sample = sample[0,1:i]
        plot_stroke(sample, save_name=filepath)
        return sample
