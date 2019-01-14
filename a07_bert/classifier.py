from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from a07_bert import config
from a07_bert import modeling
from a07_bert import tokenization
import tensorflow as tf


class BertClassifier(object):

    def __init__(self):
        self.tokenizer = self.get_tokenizer()
        self.initializer_params()
        self.initializer_sess()
        self.probabilities = self.get_probabilities()

    def initializer_params(self):
        self.input_ids = tf.placeholder(name='input_ids', shape=[1, config.max_seq_length], dtype=tf.int32)
        self.input_mask = tf.placeholder(name='input_mask', shape=[1, config.max_seq_length], dtype=tf.int32)
        self.segment_ids = tf.placeholder(name='segment_ids', shape=[1, config.max_seq_length], dtype=tf.int32)
        model = modeling.BertModel(
            config=self.get_bert_config(),
            is_training=False,
            input_ids=self.input_ids,
            input_mask=self.input_mask,
            token_type_ids=self.segment_ids,
            use_one_hot_embeddings=False)
        self.output_layer = model.get_pooled_output()
        hidden_size = self.output_layer.shape[-1].value

        self.output_weights = tf.get_variable(
            "output_weights", [config.class_num, hidden_size],
            initializer=tf.truncated_normal_initializer(stddev=0.02))
        self.output_bias = tf.get_variable(
            "output_bias", [config.class_num], initializer=tf.zeros_initializer())

    def initializer_sess(self):
        tf_config = tf.ConfigProto()
        tf_config.gpu_options.allow_growth = True
        tf_config.log_device_placement = False
        tf_config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1

        self.sess = tf.Session(config=tf_config)

    def get_tokenizer(self):
        tokenization.validate_case_matches_checkpoint(config.do_lower_case,
                                                      config.init_checkpoint)
        tokenizer = tokenization.FullTokenizer(
            vocab_file=config.vocab_file, do_lower_case=config.do_lower_case)
        return tokenizer

    def get_bert_config(self):

        bert_config = modeling.BertConfig.from_json_file(config.bert_config_file)

        if config.max_seq_length > bert_config.max_position_embeddings:
            raise ValueError(
                "Cannot use sequence length %d because the BERT model "
                "was only trained up to sequence length %d" %
                (config.max_seq_length, bert_config.max_position_embeddings))
        return bert_config

    def get_features(self, sentences):
        features = []
        for _, line in enumerate(sentences):
            text_a = tokenization.convert_to_unicode(line[0])
            text_b = tokenization.convert_to_unicode(line[1])
            features.append(self.convert_to_feature(text_a, text_b, config.max_seq_length, self.tokenizer))

        all_input_ids = []
        all_input_mask = []
        all_segment_ids = []
        for feature in features:
            all_input_ids.append(feature[0])
            all_input_mask.append(feature[1])
            all_segment_ids.append(feature[2])
        return all_input_ids, all_input_mask, all_segment_ids

    def get_probabilities(self):
        logits = tf.matmul(self.output_layer, self.output_weights, transpose_b=True)
        logits = tf.nn.bias_add(logits, self.output_bias)
        probabilities = tf.nn.softmax(logits, axis=-1)

        saver = tf.train.Saver()
        saver.restore(self.sess, tf.train.latest_checkpoint(config.data_dir))
        return probabilities

    def predict(self, sentences):
        input_ids, input_mask, segment_ids = self.get_features(sentences)
        prob = self.sess.run([self.probabilities], feed_dict={self.input_ids: input_ids,
                                                              self.input_mask: input_mask,
                                                              self.segment_ids: segment_ids})
        return prob

    def convert_to_feature(self, text_a, text_b, max_seq_length, tokenizer):
        """Converts a single `InputExample` into a single `InputFeatures`."""

        tokens_a = tokenizer.tokenize(text_a)
        tokens_b = tokenizer.tokenize(text_b)

        self._truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)

        tokens = []
        segment_ids = []
        tokens.append("[CLS]")
        segment_ids.append(0)
        for token in tokens_a:
            tokens.append(token)
            segment_ids.append(0)
        tokens.append("[SEP]")
        segment_ids.append(0)

        if tokens_b:
            for token in tokens_b:
                tokens.append(token)
                segment_ids.append(1)
            tokens.append("[SEP]")
            segment_ids.append(1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        return input_ids, input_mask, segment_ids

    def _truncate_seq_pair(self, tokens_a, tokens_b, max_length):
        """Truncates a sequence pair in place to the maximum length."""

        # This is a simple heuristic which will always truncate the longer sequence
        # one token at a time. This makes more sense than truncating an equal percent
        # of tokens from each, since if one sequence is very short then each token
        # that's truncated likely contains more information than a longer sequence.
        while True:
            total_length = len(tokens_a) + len(tokens_b)
            if total_length <= max_length:
                break
            if len(tokens_a) > len(tokens_b):
                tokens_a.pop()
            else:
                tokens_b.pop()


if __name__ == "__main__":
    import time

    bert = BertClassifier()
    while True:
        sent1 = input('sent1:')
        sent2 = input('sent2:')
        inputs = [[sent1, sent2]]
        start = time.time()
        prob = bert.predict(inputs)
        print(f'prob:{prob}')
        print(time.time() - start)
