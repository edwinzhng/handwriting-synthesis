import os
from pathlib import Path

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from utils import plot_stroke
from models.base_rnn import BaseRNN

class UnconditionalRNN(BaseRNN):
    def __init__(self, *args, **kwargs):
        super().__init__("models/weights/unconditional.h5", *args, **kwargs)
        self.build_model()

    def build_model(self):
        # input layer
        self.inputs = tf.keras.Input(shape=(None, self.input_size))

        # build RNN layers with skip connections to input
        rnns = []
        lstm_states = []
        x = self.lstm_layer(self.inputs, (None, self.input_size))
        rnns.append(x)
        for i in range(self.num_layers - 1):
            output_rnn = tf.keras.layers.concatenate([self.inputs, x])
            x = self.lstm_layer(output_rnn, (None, self.num_cells + self.input_size))
            rnns.append(x)

        # two-dimensional mean and standard deviation, scalar correlation, weights
        params_per_mixture = 6
        output_rnn = tf.keras.layers.concatenate([rnn for rnn in rnns]) # output skip connections
        self.outputs = tf.keras.layers.Dense(params_per_mixture * self.num_mixtures + 1,
                                        input_shape=(self.num_layers * self.num_cells, ))(output_rnn)

        self.model = tf.keras.Model(inputs=self.inputs, outputs=self.outputs)
        self.model.summary()

    # Equations (18 - 23)
    def output_vector(self, outputs):
        e_hat = outputs[:, :, 0]
        pi_hat, mu_hat1, mu_hat2, sigma_hat1, sigma_hat2, rho_hat = tf.split(outputs[:, :, 1:], 6, 2)

        # calculate actual values
        end_stroke = tf.math.sigmoid(e_hat)
        mixture_weight = tf.math.softmax(pi_hat, axis=-1)
        mean1 = mu_hat1
        mean2 = mu_hat2
        stddev1 = tf.math.exp(sigma_hat1)
        stddev2 = tf.math.exp(sigma_hat2)
        correl = tf.math.tanh(rho_hat)

        return [end_stroke, mixture_weight, mean1, mean2, stddev1, stddev2, correl]

    # Equation (24, 25)
    def bivariate_gaussian(self, x, y, stddev1, stddev2, mean1, mean2, correl):
        Z = tf.math.square((x - mean1) / stddev1) + tf.math.square((y - mean2) / stddev2) \
            - (2 * correl * (x - mean1) * (y - mean2) / (stddev1 * stddev2))
        return tf.math.exp(-Z / (2 * (1 - tf.math.square(correl)))) \
            / (2 * np.pi * stddev1 * stddev2 * tf.math.sqrt(1 - tf.math.square(correl)))

    # Equation (26)
    def loss(self, x, y, input_end, mean1, mean2, stddev1, stddev2, correl, mixture_weight, end_stroke):
        min_value = 1e-18 # required for logs to not be NaN when value is zero
        gaussian = self.bivariate_gaussian(self.expand_dims(x, -1, self.num_mixtures),
                                           self.expand_dims(y, -1, self.num_mixtures),
                                           stddev1, stddev2, mean1, mean2, correl)
        gaussian_loss = tf.reduce_sum(tf.math.multiply(mixture_weight, gaussian), axis=-1, keepdims=True)
        gaussian_loss = tf.math.log(tf.maximum(gaussian_loss, min_value))

        bernoulli_loss = (end_stroke * input_end) + ((1 - end_stroke) * (1 - input_end))
        bernoulli_loss = tf.math.log(tf.maximum(bernoulli_loss, min_value))
        bernoulli_loss = self.expand_dims(bernoulli_loss, -1, 1)

        return tf.reduce_sum(tf.math.negative(gaussian_loss + bernoulli_loss), axis=-1)

    @tf.function
    def train_step(self, inputs):
        with tf.GradientTape() as tape:
            outputs = self.model(inputs)
            end_stroke, mixture_weight, mean1, mean2, stddev1, stddev2, correl = self.output_vector(outputs)
            loss_value = self.loss(inputs[:,:,1], inputs[:,:,2], inputs[:,:,0], mean1, mean2,
                                   stddev1, stddev2, correl, mixture_weight, end_stroke)
            # divide by batch size
            loss_value /= inputs.shape[0]

        trainable_vars = self.model.trainable_variables
        gradients = tape.gradient(loss_value, trainable_vars)
        gradients, _ = tf.clip_by_global_norm(gradients, self.gradient_clip)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        return loss_value

    def generate(self, max_timesteps=400, seed=None):
        sample = np.zeros((1, max_timesteps + 1, 3), dtype='float32')
        for i in range(max_timesteps):
            outputs = self.model(sample[:,i:i+1,:])
            end_stroke, mixture_weight, mean1, mean2, stddev1, stddev2, correl = self.output_vector(outputs)

            # sample for MDN index from mixture weights
            mixture_dist = tfp.distributions.Categorical(probs=mixture_weight[0,0])
            mixture_idx = mixture_dist.sample()

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
            x, y = bivariate_gaussian_dist.sample(seed=seed)[0]

            # sample for end of stroke
            bernoulli = tfp.distributions.Bernoulli(probs=end_stroke)
            end_cur_stroke = bernoulli.sample(1, seed=seed)

            sample[0,i+1] = [end_cur_stroke, x, y]
            inputs = outputs

        # remove first zeros
        sample = sample[:,1:]
        plot_stroke(sample, save_name='samples/unconditional/generated.jpeg')
        return sample
