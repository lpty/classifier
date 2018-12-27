import tensorflow as tf
import numpy as np


class Model:

    def __init__(self, sentence_size, vocab_size, embed_size, hidden_size,
                 class_num, learning_rate, decay_step, decay_rate,
                 mode):
        self.sentence_size = sentence_size
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.class_num = class_num
        self.learning_rate = learning_rate
        self.decay_step = decay_step
        self.decay_rate = decay_rate
        self.mode = mode
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
        self.decoder_input = tf.placeholder(name='decoder_input', shape=(None, self.class_num + 1), dtype=tf.int32)
        self.decoder_output = tf.placeholder(name='decoder_output', shape=(None,), dtype=tf.int32)

        self.encoder_embed = tf.get_variable(name='encoder_embed', shape=(self.vocab_size, self.embed_size),
                                             dtype=tf.float32)
        self.decoder_embed = tf.get_variable(name='decoder_embed', shape=(self.class_num + 1, self.embed_size * 2),
                                             dtype=tf.float32)

    def build_epoch_increment(self):
        self.epoch_increment = tf.assign(self.epoch_step, tf.add(self.epoch_step, tf.constant(1)))

    def build_forward(self):
        encoder_input = tf.nn.embedding_lookup(self.encoder_embed, self.encoder_input)
        encoder_output, encoder_state = self.build_encoder(encoder_input)

        decoder_input = tf.nn.embedding_lookup(self.decoder_embed, self.decoder_input)
        decoder_output = self.build_decoder(encoder_output, encoder_state, decoder_input)

        self.logits = decoder_output.rnn_output[:, -1]

    def build_encoder(self, encoder_input):
        with tf.name_scope('encoder'):
            rnn_fw = tf.nn.rnn_cell.BasicRNNCell(num_units=self.hidden_size)
            rnn_bw = tf.nn.rnn_cell.BasicRNNCell(num_units=self.hidden_size)

            outputs, states = tf.nn.bidirectional_dynamic_rnn(rnn_fw, rnn_bw, encoder_input, dtype=tf.float32)
            output = tf.concat(outputs, axis=2)
            state = tf.concat(states, axis=1)
            return output, state

    def build_decoder(self, encoder_output, encoder_state, decoder_input):
        with tf.name_scope('decoder'):
            attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(num_units=self.hidden_size * 2,
                                                                       memory=encoder_output)
            decoder_cell = tf.nn.rnn_cell.BasicRNNCell(num_units=self.hidden_size * 2)
            decoder_cell = tf.contrib.seq2seq.AttentionWrapper(cell=decoder_cell,
                                                               attention_mechanism=attention_mechanism,
                                                               name='Attention_Wrapper')

            initial_state = decoder_cell.zero_state(batch_size=self.batch_size, dtype=tf.float32).clone(
                cell_state=encoder_state)
            output_layer = tf.layers.Dense(self.class_num,
                                           kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))
            if self.mode == 'train':
                sequence_length = tf.ones([self.batch_size, ], tf.int32) * (self.class_num + 1)
                helper = tf.contrib.seq2seq.TrainingHelper(inputs=decoder_input,
                                                           sequence_length=sequence_length,
                                                           time_major=False, name='training_helper')
            else:
                start_tokens = tf.ones([self.batch_size, ], tf.int32) * 0
                helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
                    embedding=self.decoder_embed,
                    start_tokens=start_tokens,
                    end_token=0)
            training_decoder = tf.contrib.seq2seq.BasicDecoder(cell=decoder_cell, helper=helper,
                                                               initial_state=initial_state,
                                                               output_layer=output_layer)
            outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder=training_decoder,
                                                              impute_finished=True,
                                                              maximum_iterations=self.class_num)
            return outputs

    def build_loss(self):
        with tf.name_scope('loss'):
            labels_one_hot = tf.one_hot(self.decoder_output, self.class_num)
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
        correct_prediction = tf.equal(tf.cast(self.predict, tf.int32), self.decoder_output)
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="Accuracy")


def _test():
    batch_size = 8
    embed_size = 100
    class_num = 2
    hidden_size = 64
    vocab_size = 10000
    sentence_size = 5
    learning_rate = 0.01
    decay_step = 1000
    decay_rate = 0.9
    mode = 'train'
    model = Model(sentence_size, vocab_size, embed_size, hidden_size, class_num,
                  learning_rate, decay_step, decay_rate, mode)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(100):
            input_x = np.zeros((batch_size, sentence_size), dtype=np.int32)
            input_y = np.array([1, 0, 1, 1, 1, 0, 1, 1], dtype=np.int32)
            input_yy = np.array([[2, 0, 1], [2, 1, 0], [2, 0, 1], [2, 0, 1],
                                 [2, 0, 1], [2, 1, 0], [2, 0, 1], [2, 0, 1]], dtype=np.int32)
            loss, _, predict, acc = sess.run(
                [model.loss, model.optimize, model.predict, model.accuracy],
                feed_dict={model.encoder_input: input_x,
                           model.decoder_input: input_yy,
                           model.decoder_output: input_y,
                           model.batch_size: batch_size})
            print("loss:", loss, "acc:", acc, "label:", input_y, "prediction:", predict)


if __name__ == '__main__':
    _test()
