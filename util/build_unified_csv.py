import csv
import re
from random import random

import jieba

from itertools import islice

from constant.system_path import TRAIN_DATA_FILE, TRAIN_DATA_LABEL_FILE, TEST_DATA_FILE, PYTORCH_TRAIN_FILE, \
    PYTORCH_TEST_FILE, PYTORCH_DEV_FILE
from constant.constant import ID_LOC, TITLE_LOC, CONTENT_LOC, LABEL_LOC

TRAIN_UNIFIED_FILE = '../data/train.csv'
TEST_UNIFIED_FILE = '../data/test.csv'


##
# This file want to merge [id, title, content] and [id, label] data file into [id, title, content, label] file
##

# check whether train_dataset and train_dataset_label's id are aligned
# now these two group data are not aligned so we had to use dict  sad :(
def check_id_align():
    with open(TRAIN_DATA_FILE, encoding='utf-8') as train_data:
        with open(TRAIN_DATA_LABEL_FILE, encoding='utf-8') as train_data_label:
            data = [row[ID_LOC] for row in csv.reader(train_data)]
            label = [row[ID_LOC] for row in csv.reader(train_data_label)]
            return data.__eq__(label)


# build id -> label dict
def build_id_label_dict():
    with open(TRAIN_DATA_LABEL_FILE, encoding='utf-8') as train_data_label:
        df = csv.reader(train_data_label)
        res = {}
        for row in islice(df, 1, None):
            res[row[ID_LOC]] = row[LABEL_LOC]
    return res


# [id, content, label] file
def build_unified_file(id_label_dict):
    # for human
    count = 0
    #
    with open(TRAIN_DATA_FILE, encoding='utf-8') as train_data:
        with open(TRAIN_UNIFIED_FILE, 'w', encoding='utf-8') as unified_file:
            unified_file.write('id\ttitle\tcontent\tlabel\n')
            data = csv.reader(train_data)
            for row in islice(data, 1, None):
                if row[ID_LOC] in id_label_dict:
                    count += 1
                    unified_file.write(row[ID_LOC] + '\t'
                                       + content_filter(row[TITLE_LOC]) + '\t'
                                       + content_filter(row[CONTENT_LOC]) + '\t'
                                       + id_label_dict[row[ID_LOC]]
                                       + '\n')
    print('total data count: ' + str(count))


def build_test_file():
    # for human
    count = 0
    #
    with open(TEST_DATA_FILE, encoding='utf-8') as test_data:
        with open(TEST_UNIFIED_FILE, 'w', encoding='utf-8') as unified_file:
            unified_file.write('id\ttitle\tcontent\n')
            data = csv.reader(test_data)
            for row in islice(data, 1, None):
                count += 1
                unified_file.write(row[ID_LOC] + '\t'
                                   + content_filter(row[TITLE_LOC]) + '\t'
                                   + content_filter(row[CONTENT_LOC])
                                   + '\n')
    print('total data count: ' + str(count))


# for data clean
def content_filter(content):
    # remove tag
    content = re.sub(re.compile(r"[~<>.{}\[\]\-\'\"\\(\\n)\(\)/;:?|_=+*%\s]", re.S), "", content)
    # remove english and number
    content = re.sub(re.compile(r"[qwertyuiopasdfghjklzxcvbnmQWERTYUIOPASDFGHJKLZXCVBNM1234567890]", re.S), "", content)
    # remove connected ""
    content = re.sub(re.compile(r"(”){2,}", re.S), "", content)
    content = re.sub(re.compile(r"\s”", re.S), "", content)

    return content


# transfer train data to title + label
def transfer_train_data(form_path, to_path):
    with open(form_path, encoding='utf-8') as data:
        with open(to_path, 'w', encoding='utf-8') as file:
            data = csv.reader(data, delimiter='\t')
            for row in islice(data, 1, None):
                # print(row)
                file.write(content_filter(row[1]) + ',' + content_filter(row[2]) + '\t'
                           + row[3] + '\n')


# transfer test data to title + label
def transfer_test_data(form_path, to_path):
    with open(form_path, encoding='utf-8') as data:
        with open(to_path, 'w', encoding='utf-8') as file:
            data = csv.reader(data, delimiter='\t')
            for row in islice(data, 1, None):
                file.write(content_filter(row[1]) + ',' + content_filter(row[2]) + '\t' + '1\n')


def word_cut(content):
    return ' '.join(jieba.cut(content))


def build_valid_data():
    with open(PYTORCH_TRAIN_FILE, encoding='utf-8') as data:
        with open(PYTORCH_DEV_FILE, 'w', encoding='utf-8') as file:
            data = csv.reader(data, delimiter='\t')
            for row in islice(data, 0, None):
                if random() > 0.9:
                    file.write(row[0] + '\t' + row[1] + '\n')


if __name__ == '__main__':
    build_unified_file(build_id_label_dict())
    build_test_file()
    transfer_train_data(TRAIN_UNIFIED_FILE, PYTORCH_TRAIN_FILE)
    transfer_test_data(TEST_UNIFIED_FILE, PYTORCH_TEST_FILE)
    build_valid_data()
