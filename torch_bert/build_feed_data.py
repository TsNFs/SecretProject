from transformers import *

pretrained_model = './data'
# Load pretrained model/tokenizer
model = BertModel.from_pretrained(pretrained_model)
# Models can return full list of hidden-states & attentions weights at each layer
model = BertModel.from_pretrained(pretrained_model,
                                  output_hidden_states=True,
                                  output_attentions=True)

# input: text content
def change_content_to_vector(content):
    pass


def change_label_to_one_hot(label, num):
    pass

