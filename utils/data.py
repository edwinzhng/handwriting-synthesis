from pathlib import Path

import numpy as np
import tensorflow as tf


class Dataloader:
    def __init__(self, batch_size=24, train_split=0.95, buffer_size=100, drop_remainder=True):
        self.batch_size = batch_size
        self.train_split = train_split
        self.buffer_size = buffer_size
        self.drop_remainder = drop_remainder

    def load_datasets(self):
        base_path = Path(__file__).parent
        path = (base_path / "../data/data.npy").resolve()
        strokes = np.load(path, allow_pickle=True)
        strokes = tf.keras.preprocessing.sequence.pad_sequences(strokes,
                                                                dtype='float32',
                                                                padding='post',
                                                                value=0.0)

        train_strokes = strokes[:int(len(strokes) * self.train_split)]
        valid_strokes = strokes[int(len(strokes) * self.train_split):]

        train_dataset = tf.data.Dataset.from_tensor_slices(train_strokes)
        train_dataset = train_dataset.batch(self.batch_size, drop_remainder=self.drop_remainder)
        self.train_dataset = train_dataset.shuffle(buffer_size=self.buffer_size)
        self.num_train_batches = np.ceil(len(train_strokes) / self.batch_size)

        valid_dataset = tf.data.Dataset.from_tensor_slices(valid_strokes)
        valid_dataset = valid_dataset.batch(self.batch_size, drop_remainder=self.drop_remainder)
        self.valid_dataset = valid_dataset.shuffle(buffer_size=self.buffer_size)
        self.num_valid_batches = np.ceil(len(valid_strokes) / self.batch_size)
