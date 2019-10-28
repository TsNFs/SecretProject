import torch
from transformers import *

# Transformers has a unified API
# for 8 transformer architectures and 30 pretrained weights.
#          Model          | Tokenizer          | Pretrained weights shortcut
MODELS = [(BertModel, BertTokenizer, './data')]
# (OpenAIGPTModel,  OpenAIGPTTokenizer,  'openai-gpt'),
# (GPT2Model,       GPT2Tokenizer,       'gpt2'),
# (CTRLModel,       CTRLTokenizer,       'ctrl'),
# (TransfoXLModel,  TransfoXLTokenizer,  'transfo-xl-wt103'),
# (XLNetModel,      XLNetTokenizer,      'xlnet-base-cased'),
# (XLMModel,        XLMTokenizer,        'xlm-mlm-enfr-1024'),
# (DistilBertModel, DistilBertTokenizer, 'distilbert-base-uncased'),
# (RobertaModel,    RobertaTokenizer,    'roberta-base')]

# To use TensorFlow 2.0 versions of the models, simply prefix the class names with 'TF', e.g. `TFRobertaModel` is the TF 2.0 counterpart of the PyTorch model `RobertaModel`

# Let's encode some text in a sequence of hidden-states using each model:
# for model_class, tokenizer_class, pretrained_weights in MODELS:
#     # Load pretrained model/tokenizer
#     tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
#     model = model_class.from_pretrained(pretrained_weights)
#
#     # Encode text
#     input_ids = torch.tensor([tokenizer.encode("这是一个句子哈哈哈", add_special_tokens=True)])  # Add special tokens takes care of adding [CLS], [SEP], <s>... tokens in the right way for each model.
#     with torch.no_grad():
#         last_hidden_states = model(input_ids)[0]  # Models outputs are now tuples
#         print(pretrained_weights)
#         print(last_hidden_states.shape)
#         print(last_hidden_states)

# Each architecture is provided with several class for fine-tuning on down-stream tasks, e.g.
BERT_MODEL_CLASSES = [BertModel, BertForPreTraining, BertForMaskedLM, BertForNextSentencePrediction,
                      BertForSequenceClassification, BertForMultipleChoice, BertForTokenClassification,
                      BertForQuestionAnswering]
MY_MODEL = [BertModel]


# All the classes for an architecture can be initiated from pretrained weights for this architecture
# Note that additional weights added for fine-tuning are only initialized
# and need to be trained on the down-stream task
pretrained_weights = './data'
tokenizer = BertTokenizer.from_pretrained(pretrained_weights)
for model_class in MY_MODEL:
    # Load pretrained model/tokenizer
    # model = model_class.from_pretrained(pretrained_weights)

    # Models can return full list of hidden-states & attentions weights at each layer
    model = model_class.from_pretrained(pretrained_weights,
                                        output_hidden_states=True,
                                        output_attentions=True)
    input_ids = torch.tensor([tokenizer.encode(
        "公安部揭开“维权”事件的黑幕,原标题：揭开“维权”事件的黑幕——公安部指挥摧毁一个以北京锋锐律师事务所为平台，“维权”律师、推手、“访民”相互勾连、滋事扰序的涉嫌重大犯罪团伙新华社北京７月11日电（人民日报记者黄庆畅、新华社记者邹伟）黑龙江庆安、江西南昌、山东潍坊、河南郑州、湖南长沙、湖北武汉12一系列热点事件的现场，为何屡屡出现律师挑头闹事、众多“访民”举牌滋事？一系列敏感案件的庭外，为何屡屡出")])

    print(model(input_ids)[1].shape)
    all_hidden_states, all_attentions = model(input_ids)[-2:]
    print(all_hidden_states[0].shape)
    print(all_attentions[0].shape)
    print(all_hidden_states)
    print(all_hidden_states)
