import datetime
import os

import numpy as np
import tensorflow as tf
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
        self.num_mixtures = num_mixtures
        self.num_cells = num_cells
        self.gradient_clip = gradient_clip
        self.name = name
        self.weights_path = 'models/weights'

        self.train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
        self.validation_loss = tf.keras.metrics.Mean('validation_loss', dtype=tf.float32)
        self.gradient_norm = tf.keras.metrics.Mean('gradient_norm', dtype=tf.float32)

    def lstm_layer(self, input_shape, stateful=False):
        return tf.keras.layers.LSTM(self.num_cells,
                                    input_shape=input_shape,
                                    stateful=stateful,
                                    return_sequences=True,
                                    use_bias=True,
                                    bias_initializer='zeros')

    # expands dims of input coordinates for gaussian and loss calculations
    def expand_input_dims(self, inputs):
        return tf.concat([inputs for _ in range(self.num_mixtures)], -1)

    # Equations (18 - 23)
    def output_vector(self, outputs):
        e_hat = outputs[:, :, 0:1]
        pi_hat, mu_hat1, mu_hat2, sigma_hat1, sigma_hat2, rho_hat = tf.split(outputs[:, :, 1:], 6, -1)

        # calculate actual values
        end_stroke = tf.math.sigmoid(-e_hat)
        mixture_weight = tf.nn.softmax(pi_hat, axis=-1)
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
        epsilon = 1e-8 # required for logs to not be NaN when value is zero
        gaussian = mixture_weight * self.bivariate_gaussian(self.expand_input_dims(x),
                                                            self.expand_input_dims(y),
                                                            stddev1, stddev2, mean1, mean2, correl)
        gaussian_loss = tf.expand_dims(tf.reduce_sum(gaussian, axis=-1), axis=-1)
        gaussian_loss = tf.math.log(tf.maximum(gaussian_loss, epsilon))
        bernoulli_loss = tf.where(tf.math.equal(tf.ones_like(input_end), input_end), end_stroke, 1 - end_stroke)
        bernoulli_loss = tf.math.log(tf.maximum(bernoulli_loss, epsilon))

        # apply mask to timesteps to avoid gradient flow for padded input
        negative_log_loss = tf.math.negative(gaussian_loss + bernoulli_loss)
        negative_log_loss = tf.where(mask, negative_log_loss, tf.zeros_like(negative_log_loss))
        return tf.reduce_mean(tf.reduce_sum(negative_log_loss, axis=1))

    def save(self, suffix=''):
        filepath = '{}/{}{}.h5'.format(self.weights_path, self.name, suffix)
        self.model.save_weights(filepath, overwrite=True)
        print('Model weights saved to {}'.format(filepath))

    def load(self, suffix=''):
        filepath = '{}/{}{}.h5'.format(self.weights_path, self.name, suffix)
        try:
            self.model.load_weights(filepath)
            print('Model weights loaded from {}'.format(filepath))
        except:
            print('Could not load weights from {}'.format(filepath))

    def train(self, epochs=50, batch_size=64, learning_rate=0.0001, epochs_per_save=10):
        self.build_model(True)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        #self.optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate, rho=0.95,
        #                                             momentum=0.9, epsilon=0.0001)
        dataloader = Dataloader(batch_size=batch_size)

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
                batches.set_description('Epoch: {}/{}, Loss: {:.6f}'.format(epoch + 1, epochs,
                                                                            self.train_loss.result()))

            self.validation(dataloader, batch_size)

            print('Finished Epoch {} with training loss {:.6f} and validation loss {:.6f}'.format(
                  epoch + 1, self.train_loss.result(), self.validation_loss.result()))

            if self.train_loss.result() < best_loss:
                self.save('_best')
                best_loss = self.train_loss.result()
                self.generate(filepath='samples/{}/generated_best.jpeg'.format(self.name))
                self.build_model(True, '_best')

            if epoch > 0 and (epoch + 1) % epochs_per_save == 0:
                self.save('_{}'.format(epoch))

            # log metrics
            with train_summary_writer.as_default():
                tf.summary.scalar('train_loss', self.train_loss.result(), step=epoch)
                tf.summary.scalar('validation_loss', self.validation_loss.result(), step=epoch)
                tf.summary.scalar('gradient', self.gradient_norm.result(), step=epoch)

            self.train_loss.reset_states()
            self.validation_loss.reset_states()
            self.gradient_norm.reset_states()

    def validation(self, dataloader, batch_size=32):
        batches = tqdm(dataloader.valid_dataset, total=dataloader.num_valid_batches,
                       leave=True, desc='Validation')
        for batch in batches:
            loss, _ = self.train_step(batch, update_gradients=False)
            self.validation_loss(loss)
            batches.set_description('Validation Loss: {:.6f}'.format(self.validation_loss.result()))
