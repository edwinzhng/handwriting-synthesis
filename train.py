import os

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from models.unconditional_rnn import UnconditionalRNN
from utils.data import Dataloader


def train_unconditional(epochs=10):
    dataloader = Dataloader()
    model = UnconditionalRNN()
    model.load()
    prev_loss = float('inf')

    print(f"Starting training...")
    for epoch in range(epochs):
        train_losses = []
        epoch_loss = 0.0
        batches = tqdm(dataloader.train_dataset, total=dataloader.num_train_batches,
                        leave=False, desc=f"Epoch: {epoch + 1}/{epochs}")
        for batch in batches:
            loss = model.train_step(batch)
            train_losses.append(loss)
            epoch_loss = np.mean(train_losses)
            batches.set_description(f"Epoch: {epoch + 1}/{epochs}, Loss: {epoch_loss} ")
        print(f"Completed Epoch {epoch + 1}")
        print(f"    Loss: {epoch_loss}")

        if epoch_loss < prev_loss:
            model.save()
            prev_loss = epoch_loss
        else:
            print(f"Skipping model save, loss increased from {prev_loss} to {epoch_loss}")

train_unconditional()
