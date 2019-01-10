import tensorflow as tf
import numpy as np


class Model:

    def __init__(self, sentence_size, vocab_size, embed_size,
                 class_num, learning_rate, decay_step, decay_rate,
                 layer_size, multi_channel_size):
        self.sentence_size = sentence_size
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.class_num = class_num
        self.learning_rate = learning_rate
        self.decay_step = decay_step
        self.decay_rate = decay_rate
        self.manual_position = False
        self.layer_size = layer_size
        self.multi_channel_size = multi_channel_size
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
        self.batch_size = tf.placeholder(name='batch_size', shape=(), dtype=tf.int32)

        self.encoder_input = tf.placeholder(name='encoder_input', shape=(None, self.sentence_size), dtype=tf.int32)
        self.y_label = tf.placeholder(name='y_label', shape=(None,), dtype=tf.int32)

        self.encoder_embed = tf.get_variable(name='encoder_embed', shape=(self.vocab_size, self.embed_size),
                                             dtype=tf.float32)

        self.W_projection = tf.get_variable(name='W_projection', shape=(self.sentence_size*self.embed_size, self.class_num), dtype=tf.float32)
        self.b_projection = tf.get_variable(name='b_projection', shape=(self.class_num,), dtype=tf.float32)

        if self.manual_position:
            self.position_embed = get_position_encoding(self.sentence_size, self.embed_size)
        else:
            self.position_embed = tf.get_variable(name='position_embed', shape=(self.sentence_size, 1),
                                                  dtype=tf.float32)

    def build_epoch_increment(self):
        self.epoch_increment = tf.assign(self.epoch_step, tf.add(self.epoch_step, tf.constant(1)))

    def build_forward(self):
        sentence_embed = tf.nn.embedding_lookup(self.encoder_embed, self.encoder_input)
        input_embed = tf.add(sentence_embed, self.position_embed)
        Q_encoded,_, _ = self.build_encoder(input_embed)
        Q_encoded = tf.reshape(Q_encoded, shape=(self.batch_size, -1))
        self.logits = tf.matmul(Q_encoded, self.W_projection) + self.b_projection

    def build_encoder(self, encoder_input):
        Q, K, V = encoder_input, encoder_input, encoder_input
        for index in range(self.layer_size):
            Q, K, V = self.build_single_layer(Q, K, V, index)
        return Q, K, V

    def build_single_layer(self, Q, K, V, index):
        with tf.variable_scope(f'single_layer_{index}'):
            multi_head_output = self.build_multi_head(Q, K, V, index)
            layer_norm_residual = self.build_layer_norm_residual(multi_head_output, Q, index, 1)
            position_wise_feed_forward = self.build_position_wise_feed_forward(layer_norm_residual, index)
            layer_norm_residual = self.build_layer_norm_residual(position_wise_feed_forward, layer_norm_residual, index, 2)
        return layer_norm_residual, layer_norm_residual, layer_norm_residual

    def build_multi_head(self, Q, K, V, index):
        with tf.variable_scope(f'multi_head_{index}'):
            Q_projected = tf.layers.dense(Q, units=self.embed_size)
            K_projected = tf.layers.dense(K, units=self.embed_size)
            V_projected = tf.layers.dense(V, units=self.embed_size)

            Q_heads = tf.stack(tf.split(Q_projected, self.multi_channel_size, axis=2), axis=1)
            K_heads = tf.stack(tf.split(K_projected, self.multi_channel_size, axis=2), axis=1)
            V_heads = tf.stack(tf.split(V_projected, self.multi_channel_size, axis=2), axis=1)

            dot_product = tf.matmul(Q_heads, K_heads, transpose_b=True)
            dot_product = dot_product * (1.0 / tf.sqrt(tf.cast(self.embed_size, tf.float32)))
            weights = tf.nn.softmax(dot_product)
            output = tf.matmul(weights, V_heads)

            batch_size, h, length, d_k = output.get_shape().as_list()
            dot_product = tf.reshape(output, shape=(-1, length, self.embed_size))

            output = tf.layers.dense(dot_product, units=self.embed_size)
            return output

    def build_layer_norm_residual(self, x, res, index, count):
        filter = x.get_shape()[-1]
        with tf.variable_scope(f"layer_norm_residual_{index}_{count}"):

            mean = tf.reduce_mean(x, axis=-1, keep_dims=True)
            variance = tf.reduce_mean(tf.square(x - mean), axis=-1, keep_dims=True)
            norm_x = (x - mean) * tf.rsqrt(variance + 1e-6)

            scale = tf.get_variable(f"layer_norm_scale_{index}_{count}", [filter], initializer=tf.ones_initializer)
            bias = tf.get_variable(f"layer_norm_bias_{index}_{count}", [filter], initializer=tf.ones_initializer)
            output = norm_x * scale + bias
            # output = output + res
            return output

    def build_position_wise_feed_forward(self, x, index):
        with tf.variable_scope(f'position_wise_feed_forward_{index}'):
            input = tf.expand_dims(x, axis=3)

            filter1 = tf.get_variable(name=f"filter1_{index}", shape=[1, self.embed_size, 1, 1])
            ouput_conv1 = tf.nn.conv2d(name=f"conv1_{index}", input=input, filter=filter1, strides=[1, 1, 1, 1], padding="VALID")

            filter2 = tf.get_variable(name=f"filter2_{index}", shape=[1, 1, 1, self.embed_size])
            output_conv2 = tf.nn.conv2d(name=f"conv2_{index}", input=ouput_conv1, filter=filter2, strides=[1, 1, 1, 1], padding="VALID")
            output = tf.squeeze(output_conv2, squeeze_dims=2)
            return output

    def build_loss(self):
        with tf.name_scope('loss'):
            labels_one_hot = tf.one_hot(self.y_label, self.class_num)
            loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=labels_one_hot)
            self.loss = tf.reduce_sum(loss)

    def build_optimize(self):
        learning_rate = tf.train.exponential_decay(learning_rate=self.learning_rate, global_step=self.global_step,
                                                   decay_steps=self.decay_step, decay_rate=self.decay_rate,
                                                   staircase=True)
        self.optimize = tf.contrib.layers.optimize_loss(self.loss, global_step=self.global_step,
                                                        learning_rate=learning_rate, optimizer="Adam")

    def build_predict(self):
        self.predict = tf.argmax(self.logits, axis=1, name="predictions")

    def build_accuracy(self):
        correct_prediction = tf.equal(tf.cast(self.predict, tf.int32), self.y_label)
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="Accuracy")


import math
def get_position_encoding(length, hidden_size, min_timescale=1.0, max_timescale=1.0e4):
    """Return positional encoding.
    Calculates the position encoding as a mix of sine and cosine functions with
    geometrically increasing wavelengths.
    Defined and formulized in Attention is All You Need, section 3.5.
    Args:
      length: Sequence length.
      hidden_size: Size of the
      min_timescale: Minimum scale that will be applied at each position
      max_timescale: Maximum scale that will be applied at each position
    Returns:
      Tensor with shape [length, hidden_size]
    """
    position = tf.to_float(tf.range(length))
    num_timescales = hidden_size // 2
    log_timescale_increment = (
            math.log(float(max_timescale) / float(min_timescale)) /
            (tf.to_float(num_timescales) - 1))
    inv_timescales = min_timescale * tf.exp(
        tf.to_float(tf.range(num_timescales)) * -log_timescale_increment)
    scaled_time = tf.expand_dims(position, 1) * tf.expand_dims(inv_timescales, 0)
    signal = tf.concat([tf.sin(scaled_time), tf.cos(scaled_time)], axis=1)
    return signal


def _test():
    batch_size = 8
    embed_size = 512
    class_num = 2
    vocab_size = 10000
    sentence_size = 5
    learning_rate = 0.01
    decay_step = 1000
    decay_rate = 0.9
    layer_size = 6
    multi_channel_size = 8
    model = Model(sentence_size, vocab_size, embed_size, class_num,
                  learning_rate, decay_step, decay_rate, layer_size,
                  multi_channel_size)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(100):
            input_x = np.zeros((batch_size, sentence_size), dtype=np.int32)
            input_y = np.array([1, 0, 1, 1, 1, 0, 1, 1], dtype=np.int32)
            loss, _, predict, acc = sess.run(
                [model.loss, model.optimize, model.predict, model.accuracy],
                feed_dict={model.encoder_input: input_x,
                           model.y_label: input_y,
                           model.batch_size: batch_size})
            print("loss:", loss, "acc:", acc, "label:", input_y, "prediction:", predict)


if __name__ == '__main__':
    _test()
