from pathlib import Path

import numpy as np
import tensorflow as tf


class Dataloader:
    def __init__(self,
                 batch_size=32,
                 train_split=0.95,
                 buffer_size=100,
                 drop_remainder=False,
                 max_sequence_length=800):
        self.batch_size = batch_size
        self.train_split = train_split
        self.buffer_size = buffer_size
        self.drop_remainder = drop_remainder
        self.max_sequence_length = max_sequence_length

        base_path = Path(__file__).parent
        strokes_path = str((base_path / "../data/data.npy").resolve())
        sentences_path = str((base_path / "../data/sentences.txt").resolve())

        strokes = np.load(strokes_path, allow_pickle=True)
        with open(sentences_path) as f:
            sentences = f.readlines()

        self.train_strokes = []
        self.train_sentences = []
        self.valid_strokes = []
        self.valid_sentences = []

        # filter out long sequences and use them as validation set
        for stroke, sentence in zip(strokes, sentences):
            if len(stroke) > self.max_sequence_length:
                self.valid_strokes.append(stroke)
                self.valid_sentences.append(sentence)
            else:
                self.train_strokes.append(stroke)
                self.train_sentences.append(sentence)

        self.train_strokes = tf.keras.preprocessing.sequence.pad_sequences(self.train_strokes,
                                                                           dtype='float32',
                                                                           padding='post',
                                                                           value=0.0)
        self.valid_strokes = tf.keras.preprocessing.sequence.pad_sequences(self.valid_strokes,
                                                                           dtype='float32',
                                                                           padding='post',
                                                                           value=0.0)

    def load_datasets(self):
        train_dataset = tf.data.Dataset.from_tensor_slices(self.train_strokes)
        train_dataset = train_dataset.batch(self.batch_size, drop_remainder=self.drop_remainder)
        self.train_dataset = train_dataset.shuffle(buffer_size=self.buffer_size)
        self.num_train_batches = len(self.train_strokes) // self.batch_size

        valid_dataset = tf.data.Dataset.from_tensor_slices(self.valid_strokes)
        valid_dataset = valid_dataset.batch(self.batch_size, drop_remainder=self.drop_remainder)
        self.valid_dataset = valid_dataset.shuffle(buffer_size=self.buffer_size)
        self.num_valid_batches = len(self.valid_strokes) // self.batch_size
