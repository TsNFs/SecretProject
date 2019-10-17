import torch
from transformers import *

# Transformers has a unified API
# for 8 transformer architectures and 30 pretrained weights.
#          Model          | Tokenizer          | Pretrained weights shortcut
MODELS = [(BertModel,       BertTokenizer,       './data')]
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
MY_MODEL = [BertForSequenceClassification]

# All the classes for an architecture can be initiated from pretrained weights for this architecture
# Note that additional weights added for fine-tuning are only initialized
# and need to be trained on the down-stream task
pretrained_weights = './data'
tokenizer = BertTokenizer.from_pretrained(pretrained_weights)
for model_class in MY_MODEL:
    # Load pretrained model/tokenizer
    model = model_class.from_pretrained(pretrained_weights)

    # Models can return full list of hidden-states & attentions weights at each layer
    model = model_class.from_pretrained(pretrained_weights,
                                        output_hidden_states=True,
                                        output_attentions=True)
    input_ids = torch.tensor([tokenizer.encode("这也是一个句子呀")])

    all_hidden_states, all_attentions = model(input_ids)[-2:]
    print(all_hidden_states[0].shape)
    print(all_attentions[0].shape)
    print(all_hidden_states)
    print(all_hidden_states)
