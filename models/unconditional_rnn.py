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

    def build_model(self, train, load_suffix='_best'):
        batch_size = None if train else 1
        stateful = False if train else True

        inputs = tf.keras.Input(shape=(None, self.input_size), batch_size=batch_size)
        lstm_1 = self.lstm_layer((batch_size, None, self.input_size), stateful=stateful)(inputs)
        skip = tf.keras.layers.concatenate([inputs, lstm_1])
        lstm_2 = self.lstm_layer((None, self.num_cells + self.input_size), stateful=stateful)(skip)
        skip = tf.keras.layers.concatenate([inputs, lstm_2])
        lstm_3 = self.lstm_layer((None, self.num_cells + self.input_size), stateful=stateful)(skip)
        skip = tf.keras.layers.concatenate([lstm_1, lstm_2, lstm_3])
        outputs = tf.keras.layers.Dense(6 * self.num_mixtures + 1, input_shape=(self.num_layers * self.num_cells,))(skip)
        self.model = tf.keras.Model(inputs=inputs, outputs=outputs)
        self.load(load_suffix)

    @tf.function
    def train_step(self, batch, update_gradients=True):
        inputs, lengths = batch
        lengths = tf.cast(lengths, dtype=tf.float32)
 
        with tf.GradientTape() as tape:
            tape.watch(inputs)
            tape.watch(lengths)

            mask = tf.expand_dims(tf.sequence_mask(lengths, tf.shape(inputs)[1]), -1)

            outputs = self.model(inputs)
            end_stroke, mixture_weight, mean1, mean2, stddev1, stddev2, correl = self.output_vector(outputs)
            input_end_stroke = tf.expand_dims(tf.gather(inputs, 0, axis=-1), axis=-1)
            x = tf.expand_dims(tf.gather(inputs, 1, axis=-1), axis=-1)
            y = tf.expand_dims(tf.gather(inputs, 2, axis=-1), axis=-1)
            loss = self.loss(x, y, input_end_stroke, mean1, mean2, stddev1, stddev2,
                             correl, mixture_weight, end_stroke, mask)

        trainable_vars = self.model.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        for i, grad in enumerate(gradients):
            gradients[i] = tf.clip_by_value(gradients[i], -self.gradient_clip, self.gradient_clip)

        if update_gradients:
            self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        return loss, gradients

    def generate(self, timesteps=400, seed=None, filepath='samples/unconditional/generated.jpeg'):
        self.build_model(False)
        sample = np.zeros((1, timesteps + 1, 3), dtype='float32')
        for i in range(timesteps):
            outputs = self.model(sample[:,i:i+1,:])
            end_stroke, mixture_weight, mean1, mean2, stddev1, stddev2, correl = self.output_vector(outputs)

            # sample for MDN index from mixture weights
            mixture_dist = tfp.distributions.Categorical(probs=mixture_weight[0,0])
            mixture_idx = mixture_dist.sample(seed=seed)

            # retrieve correct distribution values from mixture
            mean1 = tf.gather(mean1, mixture_idx, axis=-1)
            mean2 = tf.gather(mean2, mixture_idx, axis=-1)
            stddev1 = tf.gather(stddev1, mixture_idx, axis=-1)
            stddev2 = tf.gather(stddev2, mixture_idx, axis=-1)
            correl = tf.gather(correl, mixture_idx, axis=-1)

            # sample for x, y offsets
            cov_matrix = [[stddev1 * stddev1, correl * stddev1 * stddev2],
                          [correl * stddev1 * stddev2, stddev2 * stddev2]]
            bivariate_gaussian_dist = tfp.distributions.MultivariateNormalDiag(loc=[mean1, mean2], scale_diag=cov_matrix)
            bivariate_sample = bivariate_gaussian_dist.sample(seed=seed)
            x, y = bivariate_sample[0,0], bivariate_sample[1,1]

            # sample for end of stroke
            bernoulli = tfp.distributions.Bernoulli(probs=end_stroke)
            end_cur_stroke = bernoulli.sample(seed=seed)

            sample[0,i+1] = [end_cur_stroke, x, y]
            inputs = outputs

        # remove first zeros
        sample = sample[0,1:]
        plot_stroke(sample, save_name=filepath)
        return sample
