import os
from pathlib import Path

import numpy as np
import tensorflow as tf

from models.base_rnn import BaseRNN
from utils import plot_stroke
from utils.data import char_to_index, one_hot_encode


class LSTMAttentionCell(tf.keras.layers.Layer):
    def __init__(self, num_cells, num_characters, window_gaussians, regularizer, *args, **kwargs):
        super(LSTMAttentionCell, self).__init__(*args, **kwargs)
        self.lstm = tf.keras.layers.LSTMCell(num_cells,
                                             kernel_regularizer=regularizer,
                                             recurrent_regularizer=regularizer,
                                             name='lstm_1')
        self.window = tf.keras.layers.Dense(3 * window_gaussians, name='window')
        self.num_characters = num_characters
        self.window_gaussians = window_gaussians

        # h_1, c_1, prev_kappa, prev_w
        self.state_size = (num_cells, num_cells, window_gaussians, num_characters)
        # lstm, phi
        self.output_size = (num_cells, num_characters, num_characters)

    def call(self, inputs, states, constants):
        sentence_inputs, sentence_lengths = constants
        prev_h_lstm, prev_c_lstm, prev_kappa, prev_w = states

        lstm, lstm_states = self.lstm(tf.concat([inputs, prev_w], axis=-1), (prev_h_lstm, prev_c_lstm))
        h_lstm, c_lstm = lstm_states

        window_out = self.window(lstm)
        alpha_hat, beta_hat, kappa_hat = tf.split(window_out, 3, -1)

        mask = tf.sequence_mask(sentence_lengths, tf.shape(sentence_inputs)[-2])
        mask = tf.cast(mask, dtype=tf.float32)
        mask = tf.reshape(mask, (tf.shape(inputs)[0], 1, tf.shape(sentence_inputs)[-2]))

        # Equations (49-51)
        alpha = tf.expand_dims(tf.math.exp(alpha_hat), -1)
        beta = tf.expand_dims(tf.math.exp(beta_hat), -1)
        kappa = tf.expand_dims(prev_kappa + tf.math.exp(kappa_hat), -1)

        u = tf.range(1, tf.shape(sentence_inputs)[-2] + 1, delta=1.0, dtype=tf.float32)
        u = tf.expand_dims(u, 0)
        u = tf.expand_dims(u, 0)

        # shape (batch_size, num_gaussians, max_sentence_length)
        u = tf.tile(u, (tf.shape(sentence_inputs)[0], self.window_gaussians, 1))

        # Equations (46-47)
        phi = alpha * tf.math.exp(tf.math.negative(beta) * tf.math.square(kappa - u))
        phi = phi * mask
        phi = tf.reduce_sum(phi, axis=1) # shape (batch_size, max_sentence_length)
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
        self.initial_states(1)

    def build_model(self, seq_length, max_sentence_length, load_suffix='_best'):
        inputs = tf.keras.Input((seq_length, self.input_size))
        sentence_inputs = tf.keras.Input((max_sentence_length, self.num_characters))
        sentence_lengths = tf.keras.Input(1)
        input_h_1 = tf.keras.Input(self.num_cells)
        input_c_1 = tf.keras.Input(self.num_cells)
        input_h_2 = tf.keras.Input(self.num_cells)
        input_c_2 = tf.keras.Input(self.num_cells)
        input_h_3 = tf.keras.Input(self.num_cells)
        input_c_3 = tf.keras.Input(self.num_cells)
        input_kappa = tf.keras.Input(self.window_gaussians)
        input_w = tf.keras.Input(self.num_characters)
        input_states = [input_h_1, input_c_1, input_h_2, input_c_2,
                        input_h_3, input_c_3, input_kappa, input_w]

        outputs, h_1, c_1, kappa, _ = tf.keras.layers.RNN(
                                            LSTMAttentionCell(
                                                self.num_cells,
                                                self.num_characters,
                                                self.window_gaussians,
                                                self.regularizer
                                            ),
                                            return_sequences=True,
                                            return_state=True,
                                            name='lstm_attention'
                                        )(inputs,
                                          constants=[sentence_inputs, sentence_lengths],
                                          initial_state=[input_h_1, input_h_2, input_kappa, input_w])
        lstm_1, phi, w = outputs
        skip_1 = tf.keras.layers.concatenate([inputs, lstm_1, w], name='skip_1')
        lstm_2, h_2, c_2 = self.lstm_layer(name='lstm_2')(skip_1, initial_state=[input_h_2, input_c_2])

        skip_2 = tf.keras.layers.concatenate([inputs, lstm_2, w], name='skip_2')
        lstm_3, h_3, c_3 = self.lstm_layer(name='lstm_3')(skip_2, initial_state=[input_h_3, input_c_3])

        skip_3 = tf.keras.layers.concatenate([lstm_1, lstm_2, lstm_3], name='skip_3')
        outputs = tf.keras.layers.Dense(6 * self.num_mixtures + 1, name='mdn')(skip_3)

        output_states = [h_1, c_1, h_2, c_2, h_3, c_3, kappa, w]
        self.model = tf.keras.Model(inputs=[inputs, input_states, sentence_inputs, sentence_lengths],
                                    outputs=[outputs, output_states, phi])
        self.load(load_suffix)

    def train_step(self, batch):
        stroke_inputs, stroke_next_inputs, stroke_lengths, sentence_inputs, sentence_lengths = batch
        input_states = self.initial_states(tf.shape(stroke_inputs)[0])
        with tf.GradientTape() as tape:
            tape.watch(stroke_inputs)
            tape.watch(sentence_inputs)

            # create sequence mask
            stroke_mask = tf.sequence_mask(stroke_lengths, tf.shape(stroke_inputs)[1])

            # calculate loss
            outputs, output_states, phi = self.model([stroke_inputs, input_states, sentence_inputs, sentence_lengths])
            end_stroke, mixture_weight, mean1, mean2, stddev1, stddev2, correl = self.output_vector(outputs)
            input_end_stroke = stroke_next_inputs[:,:,0]
            x = stroke_next_inputs[:,:,1]
            y = stroke_next_inputs[:,:,2]
            loss = self.loss(x, y, input_end_stroke, mean1, mean2, stddev1, stddev2,
                             correl, mixture_weight, end_stroke, stroke_mask)

        return self.apply_gradients(loss, tape)

    def generate(self, text, timesteps=1000, seed=None, filepath='samples/conditional.png'):
        self.build_model(seq_length=1, max_sentence_length=len(text))
        sample = np.zeros((1, timesteps + 1, 3), dtype='float32')
        char_index, _ = char_to_index()

        one_hot_text = tf.expand_dims(one_hot_encode(text, self.num_characters, char_index), 0)
        text_length = tf.expand_dims(tf.constant(len(text)), 0)

        input_states = self.initial_states(1)
        for i in range(timesteps):
            outputs, input_states, phi = self.model([sample[:,i:i+1,:], input_states, one_hot_text, text_length])
            input_states[-1] = tf.reshape(input_states[-1], (1, self.num_characters))
            sample[0,i+1] = self.sample(outputs, seed)

            # stopping heuristic
            finished = True
            phi_last = phi[0,0,-1]
            for phi_u in phi[0,0,:-1]:
                if phi_u.numpy() > phi_last.numpy():
                    finished = False
                    break

            # prevent early stopping
            if i < 100:
                finished = False

            if finished:
                break

        # remove first zeros and discard unused timesteps
        sample = sample[0,1:i]
        plot_stroke(sample, save_name=filepath)
        return sample

    def initial_states(self, batch_size):
        input_kappa = tf.zeros((batch_size, self.window_gaussians))
        input_w = tf.zeros((batch_size, self.num_characters))
        input_states = [tf.zeros((batch_size, self.num_cells))] * 2 * self.num_layers
        input_states.extend([input_kappa, input_w])
        return input_states
