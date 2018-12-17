import tensorflow as tf
import numpy as np


class Model:

    def __init__(self, sentence_size, class_num, vocab_size, embed_size,
                 filters, filter_num, channel_size, keep_prob, learning_rate,
                 decay_step, decay_rate, k1, k_top):
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
        self.k1 = k1
        self.k_top = k_top
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
        self.num = int(self.k_top*self.embed_size/4*len(self.filters)**2*self.filter_num[-1])

        self.x = tf.placeholder(name='x', shape=(None, self.sentence_size), dtype=tf.int32)
        self.y = tf.placeholder(name='y', shape=(None,), dtype=tf.int32)

        self.w1 = tf.get_variable(name='w1', shape=(self.num, self.embed_size),
                                  dtype=tf.float32)
        self.b1 = tf.get_variable(name='b1', shape=(self.embed_size,), dtype=tf.float32)

        self.w2 = tf.get_variable(name='w2', shape=(self.embed_size, self.class_num),
                                  dtype=tf.float32)
        self.b2 = tf.get_variable(name='b2', shape=(self.class_num,), dtype=tf.float32)
        self.embed = tf.get_variable(name='embed', shape=(self.vocab_size, self.embed_size), dtype=tf.float32)

    def build_epoch_increment(self):
        self.epoch_increment = tf.assign(self.epoch_step, tf.add(self.epoch_step, tf.constant(1)))

    def build_forward(self):
        sentence = tf.nn.embedding_lookup(self.embed, self.x)
        sentence_expand = tf.expand_dims(sentence, axis=-1)
        convs_1 = self.build_per_dim_cnn([sentence_expand], layer=0)
        pools_1 = self.build_fold_k_max_pool(convs_1, self.k1)

        convs_2 = self.build_per_dim_cnn(pools_1, layer=1)
        pools_2 = self.build_fold_k_max_pool(convs_2, self.k_top)

        pool = tf.reshape(tf.concat(pools_2, axis=3), shape=(-1, self.num))
        pool_drop = tf.nn.dropout(pool, keep_prob=self.keep_prob)

        self.logits = self.build_full_connect(pool_drop)

    def build_per_dim_cnn(self, inputs, layer):
        convs = []
        for index_x, x in enumerate(inputs):
            for index_f, filter_size in enumerate(self.filters):
                with tf.name_scope(name=f'per_dim_conv_{layer}_{index_x}_{index_f}'):
                    f_value = tf.random_normal(shape=[filter_size, 1, x.shape[-1].value, self.filter_num[layer]])
                    f = tf.Variable(name=f'filter_{filter_size}', initial_value=f_value)
                    conv = tf.nn.conv2d(name='conv', input=x, filter=f,
                                        strides=[1, 1, 1, 1], padding='SAME')
                    b_value = tf.random_normal(shape=[self.filter_num[layer], ])
                    b = tf.Variable(name=f'b_{filter_size}', initial_value=b_value)
                    h = tf.nn.relu(name='relu', features=tf.nn.bias_add(conv, b))
                    convs.extend([h])
        return convs

    def build_fold_k_max_pool(self, convs, k):
        pools = []
        with tf.name_scope(name=f'fold_k_max_{k}'):
            for conv in convs:
                fold = tf.add(name='fold', x=conv[:, :, ::2, :], y=conv[:, :, 1::2, :])
                t_fold = tf.transpose(fold, perm=[0, 3, 2, 1])
                t_pool = tf.nn.top_k(name=f'{k}_max_pool', input=t_fold, k=k, sorted=False).values
                pool = tf.transpose(t_pool, perm=[0, 3, 2, 1])
                pools.extend([pool])
        return pools

    def build_full_connect(self, pool):
        with tf.name_scope('full_connect'):
            f1 = tf.tanh(tf.matmul(pool, self.w1) + self.b1)
            logits = tf.matmul(f1, self.w2) + self.b2
        return logits

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
    sentence_size = 10
    filters = [2, 3, 4]
    filter_num = [3, 6]
    channel_size = 1
    keep_prob = 0.5
    learning_rate = 0.01
    decay_step = 1000
    decay_rate = 0.9
    k1 = 5
    k_top = 2
    model = Model(sentence_size, class_num, vocab_size, embed_size, filters, filter_num, channel_size, keep_prob,
                  learning_rate, decay_step, decay_rate, k1, k_top)
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
