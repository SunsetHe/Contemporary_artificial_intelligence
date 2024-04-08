# -*- coding: utf-8 -*-
import pickle
from transformers import BertTokenizer, BertModel
import torch

# TODO 读取保存的6个变量
with open('train_valid_data.pkl', 'rb') as file:
    train_jpg, valid_jpg, train_text, valid_text, train_label, valid_label = pickle.load(file)

with open('test_data.pkl', 'rb') as file:
    test_guid, test_jpg, test_text = pickle.load(file)

# TODO bert模型，使用gpu加速计算
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased').to(device)

def text_to_bert_vector(text):
    # 使用tokenizer将文本转换为token IDs
    inputs = tokenizer(text, return_tensors="pt").to(device)

    # 使用BERT模型得到输出
    outputs = model(**inputs)

    # 提取最后一层的隐藏状态作为特征向量
    last_hidden_states = outputs.last_hidden_state

    # 取平均值，得到整体文本的特征向量
    avg_pooling = torch.mean(last_hidden_states, dim=1)

    # 分离张量，释放GPU内存
    avg_pooling = avg_pooling.detach()

    return avg_pooling

# TODO 使用bert将文本转换为特征向量
train_text_vector = [text_to_bert_vector(text) for text in train_text]
valid_text_vector = [text_to_bert_vector(text) for text in valid_text]
test_text_vector = [text_to_bert_vector(text) for text in test_text]

# TODO 打印上面所有向量的长度并检查其是否一致
train_text_vector_lengths = [len(vector[0]) for vector in train_text_vector]
valid_text_vector_lengths = [len(vector[0]) for vector in valid_text_vector]


# TODO 将特征向量保存至本地
torch.save(train_text_vector, 'train_text_vector.pt')
torch.save(valid_text_vector, 'valid_text_vector.pt')
torch.save(test_text_vector, 'test_text_vector.pt')

# # TODO 读取本地的特征向量
loaded_train_text_vector = torch.load('train_text_vector.pt')
loaded_valid_text_vector = torch.load('valid_text_vector.pt')
loaded_test_text_vector = torch.load('test_text_vector.pt')

print("Train Image Vectors Length:", train_text_vector[0])
print("Valid Image Vectors Length:", len(valid_text_vector[0]))

print(test_text_vector[0])
