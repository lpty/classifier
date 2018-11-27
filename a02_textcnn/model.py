import tensorflow as tf
import numpy as np


class Model:

    def __init__(self, sentence_size, class_num, vocab_size, embed_size,
                 filters, filter_num, channel_size, keep_prob, learning_rate,
                 decay_step, decay_rate):
        self.sentence_size = sentence_size
        self.class_num = class_num
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.filters = filters
        self.filter_num = filter_num
        self.channel_size = channel_size
        self.keep_prob = keep_prob
        self.learning_rate = learning_rate
        self.decay_step = decay_step
        self.decay_rate = decay_rate
        self.build()

    def build(self):
        self.initial_params()
        self.build_epoch_increment()
        self.build_forward()
        self.build_loss()
        self.build_optimize()
        self.build_predict()
        self.build_accuracy()

    def initial_params(self):
        self.global_step = tf.Variable(name='global_step', initial_value=0, trainable=False)
        self.epoch_step = tf.Variable(name='epoch_step', initial_value=0, trainable=False)

        self.x = tf.placeholder(name='x', shape=(None, self.sentence_size), dtype=tf.int32)
        self.y = tf.placeholder(name='y', shape=(None,), dtype=tf.int32)

        self.w = tf.get_variable(name='w', shape=(self.filter_num * len(self.filters), self.class_num),
                                 dtype=tf.float32)
        self.b = tf.get_variable(name='b', shape=(self.class_num,), dtype=tf.float32)
        self.embed = tf.get_variable(name='embed', shape=(self.vocab_size, self.embed_size), dtype=tf.float32)

    def build_epoch_increment(self):
        self.epoch_increment = tf.assign(self.epoch_step, tf.add(self.epoch_step, tf.constant(1)))

    def build_forward(self):
        sentence = tf.nn.embedding_lookup(self.embed, self.x)
        self.sentence_expand = tf.expand_dims(sentence, -1)
        self.build_cnn()
        pool_drop = tf.nn.dropout(self.pool, keep_prob=self.keep_prob)
        self.logits = tf.matmul(pool_drop, self.w) + self.b

    def build_cnn(self):
        pools = []
        for index, filter_size in enumerate(self.filters):
            with tf.name_scope(name=f'cnn_pooling_{filter_size}'):
                _filter = tf.get_variable(name=f'filter_{filter_size}',
                                          shape=(filter_size, self.embed_size, self.channel_size, self.filter_num))
                conv = tf.nn.conv2d(name='conv', input=self.sentence_expand, filter=_filter,
                                    strides=[1, 1, 1, 1], padding='VALID')
                b = tf.get_variable(name=f'b_{filter_size}', shape=(self.filter_num,))
                h = tf.nn.relu(name='relu', features=tf.nn.bias_add(conv, b))
                pool = tf.nn.max_pool(name='pool', value=h, ksize=[1, self.sentence_size - filter_size + 1, 1, 1],
                                      strides=[1, 1, 1, 1], padding='VALID')
                pools.extend([pool])
        self.pool = tf.reshape(tf.concat(pools, axis=3), shape=(-1, self.filter_num * len(self.filters)))

    def build_loss(self):
        with tf.name_scope('loss'):
            labels_one_hot = tf.one_hot(self.y, self.class_num)
            loss = tf.nn.softmax_cross_entropy_with_logits(labels=labels_one_hot,
                                                           logits=self.logits)
            self.loss = tf.reduce_sum(loss)

    def build_optimize(self):
        learning_rate = tf.train.exponential_decay(self.learning_rate, self.global_step, self.decay_step,
                                                   self.decay_rate, staircase=True)
        self.optimize = tf.contrib.layers.optimize_loss(self.loss, global_step=self.global_step,
                                                        learning_rate=learning_rate, optimizer="Adam")

    def build_predict(self):
        self.predict = tf.argmax(self.logits, axis=1, name="predictions")

    def build_accuracy(self):
        correct_prediction = tf.equal(tf.cast(self.predict, tf.int32), self.y)
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="Accuracy")


def _test():
    batch_size = 8
    embed_size = 100
    class_num = 2
    vocab_size = 10000
    sentence_size = 5
    filters = [2, 3, 4]
    filter_num = 5
    channel_size = 1
    keep_prob = 0.9
    learning_rate = 0.01
    decay_step = 1000
    decay_rate = 0.9
    model = Model(sentence_size, class_num, vocab_size, embed_size, filters, filter_num, channel_size, keep_prob,
                  learning_rate, decay_step, decay_rate)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(100):
            input_x = np.zeros((batch_size, sentence_size), dtype=np.int32)
            input_y = np.array([1, 0, 1, 1, 1, 0, 1, 1], dtype=np.int32)
            loss, _, predict, acc = sess.run(
                [model.loss, model.optimize, model.predict, model.accuracy],
                feed_dict={model.x: input_x, model.y: input_y})
            print("loss:", loss, "acc:", acc, "label:", input_y, "prediction:", predict)


if __name__ == '__main__':
    _test()
