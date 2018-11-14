import tensorflow as tf
from tflearn.data_utils import pad_sequences
from utils.preprocess import build_vocab, build_label

checkpoint_path = 'checkpoint/'
model_name = 'a01_fasttext.model-1.meta'


def test():
    v2i, _ = build_vocab()
    _, i2l = build_label()
    origin_questions = ['今天 天气 不错', '介绍 贵金属 产品']
    questions = [q.split() for q in origin_questions]
    questions = [[v2i[vocab] for vocab in ques if vocab in v2i] for ques in questions]

    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(checkpoint_path + model_name)
        saver.restore(sess, tf.train.latest_checkpoint(checkpoint_path))

        model = tf.get_default_graph()
        x = model.get_tensor_by_name("x:0")
        predict = model.get_tensor_by_name("predictions:0")

        questions = pad_sequences(questions, maxlen=x.shape[1], value=0)
        feed_dict = {x: questions}

        p = sess.run([predict], feed_dict=feed_dict)
        p = p[0].tolist()
    for index in range(len(questions)):
        print(f'{origin_questions[index]} is_business: {i2l[p[index]]}')


if __name__ == '__main__':
    test()
