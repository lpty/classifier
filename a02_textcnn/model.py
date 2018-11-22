import tensorflow as tf
import numpy as np


class Model:

    def __init__(self, sentence_size, class_num, vocab_size, embed_size,
                 filters):
        self.sentence_size = sentence_size
        self.class_num = class_num
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.filters = filters
        self.filter_num = self.filters.__len__()

    def build(self):
        self.initial_params()
        self.build_forward()

    def initial_params(self):
        self.x = tf.placeholder(name='x', shape=(None, self.sentence_size), dtype=tf.int32)
        self.y = tf.placeholder(name='y', shape=(None, self.class_num), dtype=tf.int32)

        self.w = tf.get_variable(name='w', shape=(self.filter_num, self.class_num), dtype=tf.float32)
        self.b = tf.get_variable(name='b', shape=(self.class_num,), dtype=tf.float32)
        self.embed = tf.get_variable(name='embed', shape=(self.vocab_size, self.embed_size), dtype=tf.float32)

    def build_forward(self):
        self.sentence = tf.nn.embedding_lookup(self.embed, self.x)

    def build_cnn(self):
        for index, filter_size in enumerate(self.filters):
            with tf.get_variable_scope(f'cnn_pooling_{index}_{filter_size}'):
                filter = tf.get_variable()

def _test():
    pass


if __name__ == '__main__':
    _test()
