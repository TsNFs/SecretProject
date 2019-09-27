import sys

sys.path.append('/home/lcy/nlp/SecretProject')
import csv

from itertools import islice

import util.generate_batch as generator
from constant.constant import CONTENT_LOC, ID_LOC, EMBEDDING_SIZE
from constant.system_path import TEST_DATA_FILE
import tensorflow as tf

from nn.nn_model import NN

RESULT_DATA_FILE = 'nn_result.csv'


class Predictor:
    def __init__(self):
        self.dictionary, self.embedding, _ = generator.get_dictionary_and_embedding()
        self.nn_model = NN()
        self.sess = self.load_model()
        self.embedding_size = EMBEDDING_SIZE

    # content -> label
    def predict(self, content):
        x = generator.change_content_to_vector(content, self.embedding, self.dictionary)
        x = x.reshape([1, self.embedding_size])
        result = self.sess.run(self.nn_model.result, feed_dict={self.nn_model.x: x, self.nn_model.keep_prob: 1.0})
        return result

    def load_model(self):
        with self.nn_model.graph.as_default():
            sess = tf.Session(graph=self.nn_model.graph)
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver(max_to_keep=1)
            ckpt = tf.train.get_checkpoint_state('xkf_nn_model')
            saver.restore(sess, ckpt.model_checkpoint_path)

        return sess


if __name__ == '__main__':
    p = Predictor()
    with open(TEST_DATA_FILE, encoding='utf-8') as test_file:
        with open(RESULT_DATA_FILE, 'w', encoding='utf-8') as result_file:
            # for human
            count = 0
            #
            result_file.write('id,label\n')
            df = csv.reader(test_file)
            for i in islice(df, 1, None):
                count += 1
                result_file.write(i[ID_LOC] + ',' + str(p.predict(i[CONTENT_LOC])[0]) + '\n')
                if count % 100 == 0:
                    print("predict count :" + str(count))
