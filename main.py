import numpy as np

from models.conditional_rnn import ConditionalRNN
from models.unconditional_rnn import UnconditionalRNN


def generate_unconditionally(random_seed=1):
    model = UnconditionalRNN()
    return model.generate()

def generate_conditionally(text='welcome to lyrebird', random_seed=1):
    model = ConditionalRNN()
    return model.generate()

def recognize_stroke(stroke):
    return 'welcome to lyrebird'

def train_unconditional(epochs=10, batch_size=32):
    model = UnconditionalRNN()
    model.train(epochs, batch_size)

def train_conditional(epochs=10, batch_size=32):
    model = ConditionalRNN(batch_size)
    model.train(epochs, batch_size)


if __name__=="__main__":
    train_unconditional(epochs=80, batch_size=24)