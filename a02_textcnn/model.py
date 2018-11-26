import tensorflow as tf
import numpy as np


class Model:

    def __init__(self, sentence_size, class_num, vocab_size, embed_size,
                 filters, filter_num, channel_size):
        self.sentence_size = sentence_size
        self.class_num = class_num
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.filters = filters
        self.filter_num = filter_num
        self.channel_size = channel_size

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
        self.sentence_expand = tf.expand_dims(self.sentence, -1)

    def build_cnn(self):
        for index, filter_size in enumerate(self.filters):
            with tf.get_variable_scope(f'cnn_pooling_{filter_size}'):
                filter = tf.get_variable(name=f'filter_size_{filter_size}',
                                         shape=[filter_size, self.embed_size, self.channel_size, self.filter_num])
                conv = tf.nn.conv2d(self.sentence_expand, filter, [1, 1, 1, 1], padding='VALID')
                pool = tf.nn.max_pool(conv)
                dense = tf.concat(pool)

def _test():
    pass


if __name__ == '__main__':
    _test()
