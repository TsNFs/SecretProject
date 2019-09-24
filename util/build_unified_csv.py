import csv

from itertools import islice

from constant.system_path import TRAIN_DATA_FILE, TRAIN_DATA_LABEL_FILE,TEST_DATA_FILE
from constant.constant import ID_LOC, TITLE_LOC, CONTENT_LOC, LABEL_LOC

UNIFIED_FILE = '../data/train.csv'


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
        with open(UNIFIED_FILE, 'w', encoding='utf-8') as unified_file:
            unified_file.write('id,title,content,label\n')
            data = csv.reader(train_data)
            for row in islice(data, 1, None):
                if row[ID_LOC] in id_label_dict:
                    count += 1
                    print(row[TITLE_LOC])
                    if row[TITLE_LOC].startswith('为挽救'):
                        print(row[CONTENT_LOC])
                        print("!!!")
                    print(id_label_dict[row[ID_LOC]])
                    unified_file.write(row[ID_LOC] + ','
                                       + row[TITLE_LOC] + ','
                                       + row[CONTENT_LOC] + ','
                                       + id_label_dict[row[ID_LOC]]
                                       + '\n')
    print('total data count: ' + str(count))


if __name__ == '__main__':
    build_unified_file(build_id_label_dict())
