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

    def build_model(self, batch_size):
        pass

    def loss(self, x, y, input_end, mean1, mean2, stddev1, stddev2, correl, mixture_weight, end_stroke):
        pass

    @tf.function
    def train_step(self, inputs):
        pass

    def generate(self, max_timesteps=400, seed=None, filepath='samples/conditional/generated.jpeg'):
        pass