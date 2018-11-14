import os
import pickle
import pandas as pd
from tflearn.data_utils import pad_sequences, shuffle
from utils.const import Punctuations

df = pd.read_csv('../data/question_classification_data.csv')


def build_vocab():
    vocab_path = '../cache/vocab.pkl'
    if os.path.exists(vocab_path):
        with open(vocab_path, 'rb') as f:
            vocab2index, index2vocab = pickle.load(f)
    else:
        questions = df['question'].values.tolist()
        vocabs = set(' '.join(questions).split(' '))
        vocabs = [vocab for vocab in vocabs if vocab not in Punctuations.PUNCTUATIONS]
        vocab2index = {v: k for k, v in enumerate(vocabs)}
        index2vocab = {k: v for k, v in enumerate(vocabs)}

        with open(vocab_path, 'wb') as f:
            pickle.dump((vocab2index, index2vocab), f)

    return vocab2index, index2vocab


def build_label():
    label_path = '../cache/label.pkl'
    if os.path.exists(label_path):
        with open(label_path, 'rb') as f:
            label2index, index2label = pickle.load(f)
    else:
        labels = set(df['is_business'].values.tolist())
        label2index = {v: k for k, v in enumerate(labels)}
        index2label = {k: v for k, v in enumerate(labels)}

        with open(label_path, 'wb') as f:
            pickle.dump((label2index, index2label), f)

    return label2index, index2label


def build_corpus():
    v2i, _ = build_vocab()
    vocab_size = len(v2i)
    questions = df['question'].values.tolist()
    questions = [q.split() for q in questions]
    questions = [[v2i[vocab] for vocab in ques if vocab in v2i] for ques in questions]
    sentence_size = max([len(ques) for ques in questions])
    corpus = pad_sequences(questions, maxlen=sentence_size, value=0)

    l2i, _ = build_label()
    labels = df['is_business'].values.tolist()
    labels = [l2i[label] for label in labels if label in l2i]

    corpus, labels = shuffle(corpus, labels)
    corpus_num = len(corpus)
    valid_portion = 0.1
    train = (corpus[0:int((1 - valid_portion) * corpus_num)], labels[0:int((1 - valid_portion) * corpus_num)])
    test = (corpus[int((1 - valid_portion) * corpus_num) + 1:], labels[int((1 - valid_portion) * corpus_num) + 1:])
    valid = test
    return train, test, valid, sentence_size, vocab_size


if __name__ == '__main__':
    # v2i, i2v = build_vocab()
    # l2i, i2l = build_label()
    build_corpus()
