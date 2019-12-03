# 以表格形式展示数据
import pandas as pd
from IPython.display import display

CSV_FILE = '../data/train.csv'

data = pd.read_csv(CSV_FILE)
pd.set_option('max_colwidth', 20)

#正面情绪: 0  中性情绪: 1  负面情绪: 2
print(data)