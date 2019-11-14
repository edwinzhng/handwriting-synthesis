from pathlib import Path

import numpy as np
import tensorflow as tf


def one_hot_encode(sentence, num_characters, char_to_index):
    one_hot = np.zeros((len(sentence), num_characters), dtype='float32')
    for idx, char in enumerate(sentence):
        if char in char_to_index:
            one_hot[idx][char_to_index[char]] = 1.0
        else:
            one_hot[idx][0] = 1.0
    return one_hot

def char_to_index():
    # removed any characters belonging in ' 0123456789\n+/#():;?!'
    characters = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.,"\''
    char_to_index = {}
    num_characters = len(characters) + 1

    assert num_characters == 57

    # first index reserved for unknown character
    for idx, char in enumerate(characters):
        char_to_index[char] = idx + 1

    return char_to_index, num_characters

class Dataloader:
    def __init__(self,
                 batch_size=16,
                 buffer_size=100,
                 drop_remainder=False,
                 max_sequence_length=800):
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.drop_remainder = drop_remainder
        self.max_sequence_length = max_sequence_length
        self.init_datasets()

    def init_datasets(self):
        pad_sequences = tf.keras.preprocessing.sequence.pad_sequences
        base_path = Path(__file__).parent
        strokes_path = str((base_path / "../data/data.npy").resolve())
        sentences_path = str((base_path / "../data/sentences.txt").resolve())

        strokes = np.load(strokes_path, allow_pickle=True)
        with open(sentences_path) as f:
            sentences = f.readlines()

        self.train_strokes = []
        self.train_next_strokes = []
        self.train_sentences = []

        self.char_to_index, self.num_characters = char_to_index()

        # filter out long sequences
        for stroke, sentence in zip(strokes, sentences):
            # add one extra point to max length for calculating loss using x_{t+1}
            if len(stroke) <= self.max_sequence_length + 1:
                self.train_strokes.append(stroke[:-1])
                self.train_next_strokes.append(stroke[1:])
                self.train_sentences.append(sentence)

        self.num_train_batches = np.math.ceil(len(self.train_strokes) // self.batch_size)

        # lengths for creating custom sequence masks for stroke sequences
        self.train_stroke_lengths = [len(stroke) for stroke in self.train_strokes]

        # build strokes datasets
        self.train_strokes = pad_sequences(self.train_strokes, dtype='float32', padding='post')
        self.train_next_strokes = pad_sequences(self.train_next_strokes, dtype='float32', padding='post')

        self.train_strokes = tf.data.Dataset.from_tensor_slices(self.train_strokes) \
                                .batch(self.batch_size, drop_remainder=self.drop_remainder)
        self.train_next_strokes = tf.data.Dataset.from_tensor_slices(self.train_next_strokes) \
                                    .batch(self.batch_size, drop_remainder=self.drop_remainder)
        self.train_stroke_lengths = tf.data.Dataset.from_tensor_slices(self.train_stroke_lengths) \
                                        .batch(self.batch_size, drop_remainder=self.drop_remainder)

        # build sentences datasets
        self.train_sentences = [one_hot_encode(sentence, self.num_characters, self.char_to_index)
                                for sentence in self.train_sentences]
        self.train_sentence_lengths = [len(sentence) for sentence in self.train_sentences]
        self.max_sentence_length = max(self.train_sentence_lengths)

        self.train_sentences = pad_sequences(self.train_sentences, dtype='float32', padding='post')

        self.train_sentences = tf.data.Dataset.from_tensor_slices(self.train_sentences) \
                                .batch(self.batch_size, drop_remainder=self.drop_remainder)
        self.train_sentence_lengths = tf.data.Dataset.from_tensor_slices(self.train_sentence_lengths) \
                                        .batch(self.batch_size, drop_remainder=self.drop_remainder)

    def load_datasets(self, include_sentences=False):
        if include_sentences:
            self.train_dataset = tf.data.Dataset.zip((self.train_strokes, self.train_next_strokes, self.train_stroke_lengths,
                                                      self.train_sentences, self.train_sentence_lengths))
        else:
            self.train_dataset = tf.data.Dataset.zip((self.train_strokes, self.train_next_strokes, self.train_stroke_lengths))

        # shuffle data
        self.train_dataset = self.train_dataset.shuffle(buffer_size=self.buffer_size)
