import os
from abc import ABC, abstractmethod

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from utils.data import Dataloader


class BaseRNN(ABC):
    def __init__(self,
                 weights_path,
                 num_layers=3,
                 num_mixtures=20,
                 num_cells=400,
                 lr=0.0001,
                 gradient_clip=100):
        self.input_size = 3
        self.num_layers = num_layers
        self.num_mixtures = num_mixtures
        self.num_cells = num_cells
        self.gradient_clip = gradient_clip
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        self.initalizer = tf.keras.initializers.TruncatedNormal(stddev=0.075)
        self.weights_path = weights_path

    @abstractmethod
    def build_model(self):
        pass

    @abstractmethod
    def loss(self):
        pass

    @abstractmethod
    def train_step(self):
        pass

    @abstractmethod
    def generate(self):
        pass

    def lstm_layer(self, layer_inputs, input_shape, stateful=True):
        return tf.keras.layers.LSTM(self.num_cells,
                                    input_shape=input_shape,
                                    stateful=stateful,
                                    return_sequences=True,
                                    kernel_initializer=self.initalizer,
                                    recurrent_initializer=self.initalizer,
                                    use_bias=True,
                                    bias_initializer="zeros")(layer_inputs)

    # expands dims of coordinates for gaussian and loss calculations
    def expand_dims(self, inputs, axis, num_dims):
        return tf.concat([tf.expand_dims(inputs, axis) for _ in range(num_dims)], axis)

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

    def save(self):
        self.model.save_weights(self.weights_path, overwrite=True)
        print("Model weights saved to {}".format(self.weights_path))

    def load(self):
        if os.path.exists(self.weights_path):
            self.model.load_weights(self.weights_path)
            print("Model weights loaded from {}".format(self.weights_path))
        else:
            print("No model weights to load found")

    def train(self, epochs=10, batch_size=32):
        self.build_model(batch_size=batch_size)
        self.model.summary()
        dataloader = Dataloader(batch_size=batch_size)
        train_loss = float('inf')
        prev_loss = float('inf')

        print("Training model...")
        for epoch in range(epochs):
            dataloader.load_datasets()
            train_losses = []
            batches = tqdm(dataloader.train_dataset, total=dataloader.num_train_batches,
                            leave=True, desc="Epoch: {}/{}".format(epoch + 1, epochs))
            for batch in batches:
                loss = self.train_step(batch)
                train_losses.append(loss)
                train_loss = np.mean(train_losses)
                batches.set_description("Epoch: {}/{}, Loss: {:.6f}".format(epoch + 1, epochs, train_loss))

            validation_loss = self.validation(dataloader, batch_size)
            print("Finished Epoch {} with training loss {:.6f} and validation loss {:.6f}".format(
                  epoch + 1, train_loss, validation_loss))

            if train_loss < prev_loss:
                self.save()
                self.generate()
                self.build_model(batch_size=batch_size)
                prev_loss = train_loss
            else:
                print("Skipping model save, loss increased from {} to {}".format(prev_loss, train_loss))

    def validation(self, dataloader, batch_size=32):
        valid_losses = []
        batches = tqdm(dataloader.valid_dataset, total=dataloader.num_valid_batches,
                       leave=True, desc="Validation")
        for batch in batches:
            loss = self.train_step(batch, update_gradients=False)
            valid_losses.append(loss)
            batches.set_description("Validation Loss: {:.6f}".format(np.mean(valid_losses)))

        return np.mean(valid_losses)
