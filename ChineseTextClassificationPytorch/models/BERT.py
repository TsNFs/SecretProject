import torch
from torch import nn
from torch.nn import MSELoss, CrossEntropyLoss

from transformers import PretrainedConfig
from transformers.modeling_bert import BertLayerNorm


class Config(PretrainedConfig):
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
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')  # 设备
        # self.device = torch.device('cpu')  # 设备

        self.dropout = 0.5  # 随机失活
        self.require_improvement = 10000  # 若超过1000batch效果还没提升，则提前结束训练
        self.num_labels = len(self.class_list)  # 类别数
        self.n_vocab = 0  # 词表大小，在运行时赋值
        self.num_epochs = 20  # epoch数
        self.batch_size = 4  # mini-batch大小
        self.pad_size = 200  # 每句话处理成的长度(短填长切)
        self.learning_rate = 1e-3  # 学习率
        self.encode_size = 1024  # bert hidden size


class BERT_Classifyer(nn.Module):
    def __init__(self, bert):
        super(BERT_Classifyer, self).__init__()
        self.num_labels = 3

        self.bert = bert
        self.dropout = nn.Dropout(0.8)
        self.classifier = nn.Linear(1024, self.num_labels)
        nn.init.xavier_uniform(self.classifier.weight.data)
        nn.init.constant_(self.classifier.bias.data, 0)

        # self.apply(self._init_weights)

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=1.0)
            print(module)
        elif isinstance(module, BertLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
            print(module)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
            print(module)


    def forward(self, input_ids, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None, labels=None):

        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask)

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)
