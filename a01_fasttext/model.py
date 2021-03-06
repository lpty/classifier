import tensorflow as tf
import numpy as np


class Model:

    def __init__(self, batch_size, embed_size, class_size, vocab_size, sentence_size,
                 is_train, sample_size, learning_rate, decay_step, decay_rate):
        self.batch_size = batch_size
        self.embed_size = embed_size
        self.class_size = class_size
        self.vocab_size = vocab_size
        self.sentence_size = sentence_size
        self.sample_size = sample_size
        self.is_train = is_train
        self.learning_rate = learning_rate
        self.decay_step = decay_step
        self.decay_rate = decay_rate
        self.build()

    def _is_neg_sample(self):
        return self.class_size >= 10 and self.is_train

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

        self.w = tf.get_variable(name='w', shape=(self.embed_size, self.class_size), dtype=tf.float32)
        self.b = tf.get_variable(name='b', shape=(self.class_size,), dtype=tf.float32)
        self.embed = tf.get_variable(name='embed', shape=(self.vocab_size, self.embed_size), dtype=tf.float32)

    def build_epoch_increment(self):
        self.epoch_increment = tf.assign(self.epoch_step, tf.add(self.epoch_step, tf.constant(1)))

    def build_forward(self):
        sentence = tf.nn.embedding_lookup(self.embed, self.x)
        self.sentence = tf.reduce_mean(sentence, axis=1)
        self.logit = tf.matmul(self.sentence, self.w) + self.b

    def build_loss(self):
        if self._is_neg_sample():
            labels = tf.reshape(self.y, [-1])
            labels = tf.expand_dims(labels, 1)
            loss = tf.nn.nce_loss(weights=tf.transpose(self.w),
                                  biases=self.b,
                                  labels=labels,
                                  inputs=self.sentence,
                                  num_sampled=self.sample_size,
                                  num_classes=self.class_size, partition_strategy="div")
            self.loss = tf.reduce_mean(loss)
        else:
            labels_one_hot = tf.one_hot(self.y, self.class_size)
            loss = tf.nn.softmax_cross_entropy_with_logits(labels=labels_one_hot,
                                                           logits=self.logit)
            self.loss = tf.reduce_sum(loss)

    def build_optimize(self):
        learning_rate = tf.train.exponential_decay(self.learning_rate, self.global_step, self.decay_step,
                                                   self.decay_rate, staircase=True)
        self.optimize = tf.contrib.layers.optimize_loss(self.loss, global_step=self.global_step,
                                                        learning_rate=learning_rate, optimizer="Adam")

    def build_predict(self):
        self.predict = tf.argmax(self.logit, axis=1, name="predictions")

    def build_accuracy(self):
        correct_prediction = tf.equal(tf.cast(self.predict, tf.int32), self.y)
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="Accuracy")


def _test():
    batch_size = 8
    embed_size = 100
    class_size = 2
    vocab_size = 10000
    sentence_size = 5
    is_train = True
    sample_size = 10
    learning_rate = 0.01
    decay_step = 1000
    decay_rate = 0.9
    model = Model(batch_size, embed_size, class_size, vocab_size, sentence_size, is_train,
                  sample_size, learning_rate, decay_step, decay_rate)
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
