import tensorflow as tf
from tflearn.data_utils import pad_sequences
from utils.preprocess import build_vocab, build_label, build_corpus
from tensorflow.python.platform import flags
from a05_attention.model import Model

FLAGS = flags.FLAGS
flags.DEFINE_integer('batch_size', 2, 'count of each batch for train')
flags.DEFINE_string('mode', 'predict', 'mode of model')
flags.DEFINE_integer('class_num', 2, 'class num')
flags.DEFINE_integer('embed_size', 100, 'dims of word embedding')
flags.DEFINE_float('learning_rate', 0.09, 'learning rate')
flags.DEFINE_integer('hidden_size', 64, 'hidden_size')
flags.DEFINE_integer('decay_step', 100, 'decay learning rate every decay_step')
flags.DEFINE_float('decay_rate', 0.9, 'decay learning rate with decay_rate')
flags.DEFINE_integer('epoch_num', 500, 'the number of epoch')
flags.DEFINE_integer('epoch_val', 50, 'the freq for test val')
flags.DEFINE_string('check_point', 'checkpoint/', 'checkpoint path')


def main(_):
    _, _, _, sentence_size, vocab_size = build_corpus()
    v2i, _ = build_vocab()
    _, i2l = build_label()
    origin_questions = ['今天 天气 不错', '介绍 贵金属 产品']
    questions = [q.split() for q in origin_questions]
    questions = [[0] + [v2i[vocab] for vocab in ques if vocab in v2i] for ques in questions]

    config = tf.ConfigProto()
    with tf.Session(config=config) as sess:
        model = Model(sentence_size, vocab_size, FLAGS.embed_size, FLAGS.hidden_size,
                      FLAGS.class_num, FLAGS.learning_rate, FLAGS.decay_step, FLAGS.decay_rate,
                      FLAGS.mode)
        saver = tf.train.Saver()
        saver.restore(sess, tf.train.latest_checkpoint(FLAGS.check_point))

        questions = pad_sequences(questions, maxlen=sentence_size, value=0)
        feed_dict = {model.encoder_input: questions, model.batch_size: FLAGS.batch_size}

        p = sess.run([model.predict], feed_dict=feed_dict)
        p = p[0].tolist()
    for index in range(len(questions)):
        print(f'{origin_questions[index]} is_business: {i2l[p[index]]}')


if __name__ == '__main__':
    tf.app.run()
