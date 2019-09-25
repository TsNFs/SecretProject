import sys

sys.path.append('/home/lcy/nlp/SecretProject')

from constant.constant import EMBEDDING_SIZE
from constant.system_path import TENCENT_EMBEDDING_LOC, EMBEDDING_TXT, DICT_TXT
import numpy as np
import pickle

if __name__ == '__main__':
    count = 0
    dictionary = {}
    final_embedding = []
    with open(TENCENT_EMBEDDING_LOC, encoding='utf-8') as f:
        # jump the first row
        f.readline()
        line = f.readline()
        while line:
            count += 1
            if count % 1000 == 0:
                print('cur: ' + str(count))
            row = line.split(' ')
            cur_embedding = []
            # for dirty in data
            try:
                for i in range(1, EMBEDDING_SIZE + 1):
                    cur_embedding.append(float(row[i]))
                dictionary[row[0]] = len(dictionary)
                embedding = [cur_embedding]
                if len(final_embedding) == 0:
                    final_embedding = np.array(embedding)
                else:
                    final_embedding = np.row_stack((final_embedding, embedding))
            except:
                print('error at line : ' + str(count) + line)
            line = f.readline()

    # dump the data
    f = open(EMBEDDING_TXT, 'wb')
    pickle.dump(final_embedding, f)
    f.close()

    f = open(DICT_TXT, 'wb')
    pickle.dump(dictionary, f)
    f.close()
