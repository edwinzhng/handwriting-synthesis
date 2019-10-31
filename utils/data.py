from pathlib import Path

import numpy as np
import tensorflow as tf


class Dataloader:
    def __init__(self, batch_size=32, train_split=0.95, buffer_size=100):
        self.batch_size = batch_size
        self.train_split = train_split
        self.buffer_size = buffer_size
        self.load_datasets()

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
        train_dataset = train_dataset.batch(self.batch_size)
        self.train_dataset = train_dataset.shuffle(buffer_size=self.buffer_size)

        valid_dataset = tf.data.Dataset.from_tensor_slices(valid_strokes)
        valid_dataset = valid_dataset.batch(self.batch_size)
        self.valid_dataset = valid_dataset.shuffle(buffer_size=self.buffer_size)
