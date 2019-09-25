import sys

sys.path.append('/home/lcy/nlp/SecretProject')

from constant.constant import EMBEDDING_SIZE
from constant.system_path import TENCENT_EMBEDDING_LOC, EMBEDDING_TXT, DICT_TXT
import numpy as np
import pickle

if __name__ == '__main__':
    count = 0
    dictionary = {}
    embedding = []
    with open(TENCENT_EMBEDDING_LOC, encoding='utf-8') as f:
        # jump the first row
        f.readline()
        line = f.readline()
        while line:
            count += 1
            if count % 1000 == 0:
                print('cur: ' + str(count))
            row = line.split(' ')
            dictionary[row[0]] = len(dictionary)
            cur_embedding = []
            for i in range(1, EMBEDDING_SIZE + 1):
                cur_embedding.append(float(row[i]))
            embedding.append(cur_embedding)
            line = f.readline()

    final_embedding = np.array(embedding)

    # dump the data
    f = open(EMBEDDING_TXT, 'wb')
    pickle.dump(final_embedding, f)
    f.close()

    f = open(DICT_TXT, 'wb')
    pickle.dump(dictionary, f)
    f.close()
