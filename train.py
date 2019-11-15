import argparse

import numpy as np

from models.conditional_rnn import ConditionalRNN
from models.unconditional_rnn import UnconditionalRNN


def train_unconditional(epochs, batch_size, learning_rate):
    model = UnconditionalRNN()
    model.train(epochs, batch_size, learning_rate)

def train_conditional(epochs, batch_size, learning_rate):
    model = ConditionalRNN()
    model.train(epochs, batch_size, learning_rate)


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, choices=['conditional', 'unconditional'], help='model type')
    parser.add_argument('--batch_size', '-b', type=int, default=32, help='batch size')
    parser.add_argument('--epochs', '-e', type=int, default=50, help='number of epochs')
    parser.add_argument('--learning-rate', '-lr', type=float, default=0.0001, help='learning rate')
    args = parser.parse_args()

    if args.model == 'unconditional':
        train_unconditional(args.epochs, args.batch_size, args.learning_rate)
    else:
        train_conditional(args.epochs, args.batch_size, args.learning_rate)