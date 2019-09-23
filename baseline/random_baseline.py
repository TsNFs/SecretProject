import csv
from random import random
from constant.system_path import TEST_DATA_FILE
from itertools import islice

# just random build result set
RESULT_DATA_FILE = "random_result.csv"

with open(TEST_DATA_FILE, encoding='utf-8') as test_file:
    with open(RESULT_DATA_FILE, 'w', encoding='utf-8') as result_file:
        result_file.write('id,label\n')
        df = csv.reader(test_file)
        for i in islice(df, 1, None):
            result_file.write(i[0] + ',' + str(int(random() * 3)) + '\n')
# with open(RESULT_DATA_FILE, 'w') as f:
