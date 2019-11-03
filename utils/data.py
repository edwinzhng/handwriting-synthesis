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
        self.init_datasets()

    def init_datasets(self):
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

        # removed any characters belonging in '0123456789\n+/#():;'
        characters = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ .,"\'!?'
        self.char_to_index = {}
        self.num_characters = len(characters) + 1

        # first index reserved for unknown
        for idx, char in enumerate(characters):
            self.char_to_index[char] = idx + 1

        # filter out long sequences and use them as validation set
        for stroke, sentence in zip(strokes, sentences):
            if len(stroke) > self.max_sequence_length:
                self.valid_strokes.append(stroke)
                self.valid_sentences.append(sentence)
            else:
                self.train_strokes.append(stroke)
                self.train_sentences.append(sentence)

        self.max_train_sentence_length = max([len(sentence) for sentence in self.train_sentences])
        self.max_valid_sentence_length = max([len(sentence) for sentence in self.valid_sentences])
        self.num_train_batches = np.math.ceil(len(self.train_strokes) // self.batch_size)
        self.num_valid_batches = np.math.ceil(len(self.valid_strokes) // self.batch_size)

        # build strokes datasets
        self.train_strokes = tf.keras.preprocessing.sequence.pad_sequences(self.train_strokes, dtype='float32',
                                                                           padding='post', value=0.0)
        self.valid_strokes = tf.keras.preprocessing.sequence.pad_sequences(self.valid_strokes, dtype='float32',
                                                                           padding='post', value=0.0)
        
        self.train_strokes = tf.data.Dataset.from_tensor_slices(self.train_strokes)
        self.train_strokes = self.train_strokes.batch(self.batch_size, drop_remainder=self.drop_remainder)

        self.valid_strokes = tf.data.Dataset.from_tensor_slices(self.valid_strokes)
        self.valid_strokes = self.valid_strokes.batch(self.batch_size, drop_remainder=self.drop_remainder)

        # build sentences datasets
        self.train_sentences = [self.one_hot_encode(sentence, self.max_train_sentence_length)
                                for sentence in self.train_sentences]
        self.valid_sentences = [self.one_hot_encode(sentence, self.max_valid_sentence_length)
                                for sentence in self.valid_sentences]

        self.train_sentences = tf.data.Dataset.from_tensor_slices(self.train_sentences)
        self.train_sentences = self.train_sentences.batch(self.batch_size,
                                                                 drop_remainder=self.drop_remainder)

        self.valid_sentences = tf.data.Dataset.from_tensor_slices(self.valid_sentences)
        self.valid_sentences = self.valid_sentences.batch(self.batch_size,
                                                                 drop_remainder=self.drop_remainder)

    def one_hot_encode(self, sentence, max_length):
        one_hot = np.zeros((max_length, self.num_characters), dtype='float32')
        for idx, char in enumerate(sentence):
            if char in self.char_to_index:
                one_hot[idx][self.char_to_index[char]] = 1.0
            else:
                one_hot[idx][0] = 1.0
        return one_hot

    # loads datasets for training
    def load_datasets(self, include_sentences=False):
        if not include_sentences:
            self.train_dataset = self.train_strokes
            self.valid_dataset = self.valid_strokes
        else:
            self.train_dataset = tf.data.Dataset.zip((self.train_strokes, self.train_sentences))
            self.valid_dataset = tf.data.Dataset.zip((self.valid_strokes, self.valid_sentences))

        # shuffle data
        self.train_dataset = self.train_dataset.shuffle(buffer_size=self.buffer_size)
        self.valid_dataset = self.valid_dataset.shuffle(buffer_size=self.buffer_size)
