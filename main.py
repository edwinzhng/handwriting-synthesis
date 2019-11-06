import argparse

import numpy as np

from models.conditional_rnn import ConditionalRNN
from models.unconditional_rnn import UnconditionalRNN


def generate_unconditionally(random_seed=None, filepath=None):
    model = UnconditionalRNN()
    return model.generate(seed=random_seed, filepath=filepath)

def generate_conditionally(text='welcome to lyrebird', random_seed=None, filepath=None):
    model = ConditionalRNN()
    return model.generate(text, seed=random_seed, filepath=filepath)

def recognize_stroke(stroke):
    return 'welcome to lyrebird'


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, choices=['conditional', 'unconditional'], help='model type')
    parser.add_argument('--seed', '-s', type=int, default=None, help='random seed')
    parser.add_argument('--filepath', '-f', type=str, default=None, help='file path to save generated image')
    parser.add_argument('--text', '-t', type=str, default='welcome to lyrebird', help='text for conditional generation')
    args = parser.parse_args()

    if args.model == 'unconditional':
        generate_unconditionally(args.seed, args.filepath)
    else:
        generate_conditionally(args.text, args.seed, args.filepath)
