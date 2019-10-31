import os

import numpy as np
import tensorflow as tf

from models.unconditional_rnn import UnconditionalRNN
from utils.data import Dataloader

def train_unconditional(epochs=10):
    dataloader = Dataloader()
    model = UnconditionalRNN()
    for epoch in range(epochs):
        train_losses = []
        for batch in dataloader.train_dataset:
            loss = model.train_step(batch)
            train_losses.append(loss)
        print(f"Finished Epoch {epoch}")
        print(f"    Loss: {np.mean(train_losses)}")
        model.save()

train_unconditional()
