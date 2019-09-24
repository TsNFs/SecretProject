import csv
import random
import jieba

import pickle
import numpy as np
import datetime

from constant.system_path import TRAIN_DATA_FILE, TEST_DATA_FILE, FONT_FILE, DICT_TXT, EMBEDDING_TXT, \
    GENERATE_TRAIN_FILE
from constant.constant import CONTENT_LOC, TRAIN_LABEL_LOC, NUM_OF_CLASS

# param
embedding_size = 128
batch_size = 128


def get_dictionary_and_embedding():
    with open(EMBEDDING_TXT, "rb") as f:
        embedding = pickle.load(f)
    with open(DICT_TXT, "rb") as f:
        word_dictionary = pickle.load(f)

    return word_dictionary, embedding, dict(zip(word_dictionary.values(), word_dictionary.keys()))


def change_content_to_vector(content, embedding, dictionary):
    result = np.zeros(embedding_size)
    count = 0
    for word in list(jieba.cut(content, cut_all=False)):
        if word in dictionary:
            count = count + 1
            result += embedding[dictionary[word]]

    if count != 0:
        result = result / count
    return result


# label : 数据列，one-hot编码之后非零的列
# max : num of class
def change_label_to_one_hot(label, max):
    result = np.zeros([max + 1])
    result[label] = 1

    return result


def build_data_to_x_y(embedding, dictionary):
    i = 0
    data_x = []
    data_y = []

    time = datetime.datetime.now()
    with open(GENERATE_TRAIN_FILE, "r", encoding="UTF-8") as train_data:
        content = [row for row in csv.reader(train_data)][1:]
        for row in content:
            i += 1

            data_x.append(change_content_to_vector(row[CONTENT_LOC], embedding, dictionary))
            data_y.append(int(row[TRAIN_LABEL_LOC]))

            if i % 1000 == 0:
                print("read ", i, "lines")

    result_x = np.ndarray([len(data_x), embedding_size])

    result_y = np.ndarray([len(data_y), NUM_OF_CLASS + 1])
    for i in range(len(data_x)):
        result_x[i] = data_x[i]
        result_y[i] = change_label_to_one_hot(data_y[i], NUM_OF_CLASS)

    print(datetime.datetime.now() - time)
    return result_x, result_y


def generate_batch(data_x, data_y):
    x = np.ndarray([batch_size, len(data_x[0])], dtype=float)
    if len(data_y.shape) > 1:
        y = np.ndarray([batch_size, len(data_y[0])], dtype=int)
    else:
        y = np.ndarray([batch_size], dtype=int)
    # print(len(data_y[0]))

    for i in range(batch_size):
        index = random.randint(0, len(data_x) - 1)
        x[i] = data_x[index]
        y[i] = data_y[index]

    return x, y


# word_dict, embedding, reverse_dictionary = get_dictionary_and_embedding()
# accu_dict, reverse_accu_dict = read_accu()
# a, b = build_data_to_x_y(embedding, word_dict)
# print(a)
# print(len(a[0]))
