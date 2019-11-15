import datetime
import os
from pathlib import Path

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tqdm import tqdm

from utils.data import Dataloader


class BaseRNN():
    def __init__(self,
                 name,
                 num_mixtures=20,
                 num_cells=400,
                 gradient_clip=10):
        self.input_size = 3
        self.num_layers = 3
        self.num_cells = 400
        self.epsilon = 1e-7 # numerical stability
        self.num_mixtures = num_mixtures
        self.num_cells = num_cells
        self.gradient_clip = gradient_clip
        self.name = name
        self.regularizer = tf.keras.regularizers.l2()

        self.train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
        self.gradient_norm = tf.keras.metrics.Mean('gradient_norm', dtype=tf.float32)

        base_path = Path(__file__).parent
        self.weights_path = str((base_path / "../models/weights").resolve())

    def lstm_layer(self, name=None):
        return tf.keras.layers.LSTM(self.num_cells,
                                    return_sequences=True,
                                    return_state=True,
                                    kernel_regularizer=self.regularizer,
                                    recurrent_regularizer=self.regularizer,
                                    name=name)

    # expands dims of input coordinates for gaussian and loss calculations
    def expand_input_dims(self, inputs):
        return tf.stack([inputs] * self.num_mixtures, -1)

    # Equations (18 - 23)
    def output_vector(self, outputs):
        e_hat = outputs[:, :, 0]
        pi_hat, mu_hat1, mu_hat2, sigma_hat1, sigma_hat2, rho_hat = tf.split(outputs[:, :, 1:], 6, -1)

        # calculate actual values
        end_stroke = tf.math.sigmoid(-e_hat)
        mixture_weight = tf.nn.softmax(pi_hat)
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
    def loss(self, x, y, input_end, mean1, mean2, stddev1, stddev2, correl, mixture_weight, end_stroke, mask):
        gaussian = mixture_weight * self.bivariate_gaussian(self.expand_input_dims(x),
                                                            self.expand_input_dims(y),
                                                            stddev1, stddev2, mean1, mean2, correl)
        gaussian_loss = tf.math.log(tf.reduce_sum(gaussian, axis=-1) + self.epsilon)
        bernoulli_loss = tf.where(tf.math.equal(tf.ones_like(input_end), input_end), end_stroke, 1 - end_stroke)
        bernoulli_loss = tf.math.log(bernoulli_loss + self.epsilon)

        # apply mask to timesteps to avoid gradient flow for padded input
        negative_log_loss = -gaussian_loss - bernoulli_loss
        negative_log_loss = tf.where(mask, negative_log_loss, tf.zeros_like(negative_log_loss))
        return tf.reduce_mean(tf.math.reduce_sum(negative_log_loss, axis=1))

    def save(self, suffix=''):
        filepath = '{}/{}{}.h5'.format(self.weights_path, self.name, suffix)
        self.model.save_weights(filepath, overwrite=True)
        print('Model weights saved to {}'.format(filepath))

    def load(self, suffix=''):
        filepath = '{}/{}{}.h5'.format(self.weights_path, self.name, suffix)
        try:
            self.model.load_weights(filepath)
        except:
            print('Could not load weights from {}'.format(filepath))

    def train(self, epochs=50, batch_size=64, learning_rate=0.0001, epochs_per_save=10):
        dataloader = Dataloader(batch_size=batch_size)
        self.build_model(dataloader.max_sequence_length, dataloader.max_sentence_length)
        self.model.summary()
        self.optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate, rho=0.95,
                                                     momentum=0.9, epsilon=0.0001)

        current_time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
        train_log_dir = 'logs/{}_{}/train'.format(self.name, current_time)
        train_summary_writer = tf.summary.create_file_writer(train_log_dir)

        best_loss = float('inf')

        print('Training model...')
        for epoch in range(epochs):
            dataloader.load_datasets(self.name == 'conditional')
            batches = tqdm(dataloader.train_dataset, total=dataloader.num_train_batches,
                            leave=True, desc='Epoch: {}/{}'.format(epoch + 1, epochs))
            for batch in batches:
                loss, gradients = self.train_step(batch)
                self.train_loss(loss)
                self.gradient_norm(tf.linalg.global_norm(gradients))
                batches.set_description('Epoch: {}/{} Loss: {:.6f}'.format(epoch + 1, epochs,
                                                                           self.train_loss.result()))

            print('Epoch {}: training loss {:.6f}'.format(epoch + 1, self.train_loss.result()))

            if self.train_loss.result() < best_loss:
                self.save('_best')
                best_loss = self.train_loss.result()
                # self.generate(filepath='samples/{}.png'.format(self.name))
                # self.build_model(dataloader.max_sequence_length, '_best')

            if epoch > 0 and (epoch + 1) % epochs_per_save == 0:
                self.save('_{}'.format(epoch))

            # log metrics
            with train_summary_writer.as_default():
                tf.summary.scalar('train_loss', self.train_loss.result(), step=epoch)
                tf.summary.scalar('gradient', self.gradient_norm.result(), step=epoch)

            self.train_loss.reset_states()
            self.gradient_norm.reset_states()
            tf.keras.backend.clear_session()

    def apply_gradients(self, loss, tape):
        trainable_vars = self.model.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        for i, grad in enumerate(gradients):
            if trainable_vars[i].name.startswith('lstm'):
                gradients[i] = tf.clip_by_value(grad, -10.0, 10.0)
            elif trainable_vars[i].name.startswith('mdn'):
                gradients[i] = tf.clip_by_value(grad, -100.0, 100.0)

        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        return loss, gradients

    def sample(self, outputs, seed):
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
        return [end_cur_stroke, x, y]
