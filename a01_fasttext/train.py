import os
import tensorflow as tf
from tensorflow.python.platform import flags
from a01_fasttext.model import Model
from utils.preprocess import build_corpus

FLAGS = flags.FLAGS
flags.DEFINE_integer('batch_size', 256, 'count of each batch for train')
flags.DEFINE_integer('embed_size', 100, 'dims of word embedding')
flags.DEFINE_integer('class_size', 2, 'class num')
flags.DEFINE_bool('is_train', True, 'training or not')
flags.DEFINE_integer('sample_size', 10, 'number of NCE loss sample')
flags.DEFINE_float('learning_rate', 0.9, 'learning rate')
flags.DEFINE_integer('decay_step', 100, 'decay learning rate every decay_step')
flags.DEFINE_float('decay_rate', 0.9, 'decay learning rate with decay_rate')
flags.DEFINE_integer('epoch_num', 500, 'the number of epoch')
flags.DEFINE_integer('epoch_val', 50, 'the freq for test val')
flags.DEFINE_string('check_point', 'checkpoint/', 'checkpoint path')


def main(_):
    train, test, _, sentence_size, vocab_size = build_corpus()
    train_x, train_y = train
    test_x, test_y = test

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        model = Model(FLAGS.batch_size, FLAGS.embed_size, FLAGS.class_size, vocab_size, sentence_size, FLAGS.is_train,
                      FLAGS.sample_size, FLAGS.learning_rate, FLAGS.decay_step, FLAGS.decay_rate)

        saver = tf.train.Saver()
        if os.path.exists(FLAGS.check_point + 'checkpoint'):
            saver.restore(sess, tf.train.latest_checkpoint(FLAGS.checkpoint))
        else:
            sess.run(tf.global_variables_initializer())

        cur_epoch = sess.run(model.epoch_step)
        corpus_size = len(train_x)
        for epoch in range(cur_epoch, FLAGS.epoch_num):
            loss, acc, count = 0, 0, 1
            for start, end in zip(range(0, corpus_size, FLAGS.batch_size),
                                  range(FLAGS.batch_size, corpus_size, FLAGS.batch_size)):
                _loss, _, _, _acc = sess.run(
                    [model.loss, model.optimize, model.predict, model.accuracy],
                    feed_dict={model.x: train_x[start: end], model.y: train_y[start: end]})
                loss, acc, count = loss + _loss, acc + _acc, count + 1
                if count % FLAGS.batch_size == 0:
                    print(f'Train -- epoch: {epoch}, count: {count}, loss: {loss/count}, acc: {acc/count}')
            sess.run(model.epoch_increment)
            if epoch % FLAGS.epoch_val == 0:
                cur_loss, cur_predict, cur_acc = sess.run([model.loss, model.predict, model.accuracy],
                                                          feed_dict={model.x: test_x, model.y: test_y})
                print(f'Eval -- epoch: {epoch}, loss: {cur_loss}, acc: {cur_acc}')

                save_path = FLAGS.check_point + "a01_fasttext.model"
                saver.save(sess, save_path, global_step=model.epoch_step)


if __name__ == '__main__':
    tf.app.run()
