# coding: UTF-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Config(object):
    """配置参数"""

    def __init__(self, dataset, embedding):
        self.model_name = 'NN'
        self.train_path = dataset + '/data/train.txt'  # 训练集
        self.dev_path = dataset + '/data/dev.txt'  # 验证集
        self.test_path = dataset + '/data/test.txt'  # 测试集
        self.class_list = [x.strip() for x in open(
            dataset + '/data/class.txt').readlines()]  # 类别名单
        self.vocab_path = dataset + '/data/vocab.pkl'  # 词表
        self.save_path = dataset + '/saved_dict/' + self.model_name + '.ckpt'  # 模型训练结果
        self.log_path = dataset + '/log/' + self.model_name
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 设备

        self.dropout = 0.5  # 随机失活
        self.require_improvement = 1000  # 若超过1000batch效果还没提升，则提前结束训练
        self.num_classes = len(self.class_list)  # 类别数
        self.n_vocab = 0  # 词表大小，在运行时赋值
        self.num_epochs = 100  # epoch数
        self.batch_size = 128  # mini-batch大小
        self.pad_size = 200  # 每句话处理成的长度(短填长切)
        self.learning_rate = 1e-3  # 学习率
        self.hidden_size = 32  # 隐含层大小
        self.encode_size = 1024  # bert hidden size



'''Neural Networks for Sentence Classification'''


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.dropout = nn.Dropout(config.dropout)
        self.fc1 = nn.Sequential(nn.Linear(config.pad_size * config.encode_size, config.hidden_size), nn.ReLU(True))
        self.fc2 = nn.Linear(config.hidden_size, config.num_classes)
        self.one_layer_fc = nn.Linear(config.encode_size, config.num_classes)
        self.input_size = config.encode_size
        self.batch_size = config.batch_size

    def forward(self, x):
        # print(x[0].shape)
        out = x[0].view(x[0].shape[0], self.input_size)
        out = self.one_layer_fc(out)
        return out
