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

    print("Starting training...")
    for epoch in range(epochs):
        train_losses = []
        epoch_loss = 0.0
        batches = tqdm(dataloader.train_dataset, total=dataloader.num_train_batches,
                        leave=False, desc="Epoch: {}/{}".format(epoch + 1, epochs))
        for batch in batches:
            loss = model.train_step(batch)
            train_losses.append(loss)
            epoch_loss = np.mean(train_losses)
            batches.set_description("Epoch: {}/{}, Loss: {:.6f} ".format(epoch + 1, epochs, epoch_loss))
        print("Finished Epoch {} with average loss of {:.6f}".format(epoch + 1, epoch_loss))

        if epoch_loss < prev_loss:
            model.save()
            model.generate()
            prev_loss = epoch_loss
        else:
            print("Skipping model save, loss increased from {} to {}".format(prev_loss, epoch_loss))

train_unconditional()
