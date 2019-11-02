from pathlib import Path

import numpy as np
import tensorflow as tf


class Dataloader:
    def __init__(self,
                 batch_size=32,
                 train_split=0.95,
                 buffer_size=100,
                 drop_remainder=True,
                 max_sequence_length=800):
        self.batch_size = batch_size
        self.train_split = train_split
        self.buffer_size = buffer_size
        self.drop_remainder = drop_remainder
        self.max_sequence_length = max_sequence_length

        base_path = Path(__file__).parent
        strokes_path = (base_path / "../data/data.npy").resolve()
        sentences_path = (base_path / "../data/sentences.txt").resolve()

        strokes = np.load(strokes_path, allow_pickle=True)
        with open(sentences_path) as f:
            sentences = f.readlines()

        self.strokes = []
        self.sentences = []

        # filter out long sequences
        for stroke, sentence in zip(strokes, sentences):
            if len(stroke) > self.max_sequence_length:
                continue
            self.strokes.append(stroke)
            self.sentences.append(sentence)

        self.strokes = tf.keras.preprocessing.sequence.pad_sequences(self.strokes,
                                                                     dtype='float32',
                                                                     padding='post',
                                                                     value=0.0)

    def load_datasets(self):
        train_strokes = self.strokes[:int(len(self.strokes) * self.train_split)]
        valid_strokes = self.strokes[int(len(self.strokes) * self.train_split):]

        train_dataset = tf.data.Dataset.from_tensor_slices(train_strokes)
        train_dataset = train_dataset.batch(self.batch_size, drop_remainder=self.drop_remainder)
        self.train_dataset = train_dataset.shuffle(buffer_size=self.buffer_size)
        self.num_train_batches = len(train_strokes) // self.batch_size

        valid_dataset = tf.data.Dataset.from_tensor_slices(valid_strokes)
        valid_dataset = valid_dataset.batch(self.batch_size, drop_remainder=self.drop_remainder)
        self.valid_dataset = valid_dataset.shuffle(buffer_size=self.buffer_size)
        self.num_valid_batches = len(valid_strokes) // self.batch_size
