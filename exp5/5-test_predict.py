# -*- coding: utf-8 -*-
import csv
import torch
import torch.nn as nn
import pickle
from torch.utils.data import TensorDataset, DataLoader

test_jpg_vector = torch.load('test_jpg_vector.pt')

test_text_vector = torch.load('test_text_vector.pt')

with open('test_data.pkl', 'rb') as file:
    test_guid, test_jpg, test_text = pickle.load(file)

test_combined = [torch.cat((jpg_vec, text_vec.squeeze(0)), dim=0) for jpg_vec, text_vec in zip(test_jpg_vector, test_text_vector)]

test_combined = torch.stack(test_combined)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class MultiModalTransformer(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MultiModalTransformer, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x

input_size = test_combined[0].shape[0]  # Assuming all tensors in the list have the same size
hidden_size = 64
output_size = 3

# 加载已保存的模型
model = MultiModalTransformer(input_size, hidden_size, output_size)
model.load_state_dict(torch.load('model_epoch_6.pt'))
model.to(device)
model.eval()

# 使用测试集进行预测
test_dataset = TensorDataset(test_combined)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

all_preds = []

with torch.no_grad():
    for inputs in test_loader:
        inputs = inputs[0].to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())

# TODO 转化
class_mapping = {2: "positive", 1: "neutral", 0: "negative"}

# 将预测结果转化为文本
all_preds_text = [class_mapping[pred] for pred in all_preds]

# 读取txt文件
with open('test_without_label.txt', 'r', encoding='utf-8') as file:
    lines = file.readlines()

# 初始化结果列表
result_lines = []

# 处理每一行文本
for line in lines:
    if line.startswith('guid,tag'):  # 跳过第一行
        result_lines.append(line)
        continue

    # 提取guid
    guid = int(line.split(',')[0])

    # 在test_guid中查找guid对应的索引
    try:
        index = test_guid.index(guid)
    except ValueError:
        print(f"Not found guid: {guid}")
        continue

    # 获取预测结果文本
    pred_text = all_preds_text[index]

    # 将null替换为预测结果
    line = line.replace('null', pred_text)

    # 添加到结果列表
    result_lines.append(line)

# 将结果写入新的文件
with open('test_with_label.txt', 'w', encoding='utf-8') as file:
    file.writelines(result_lines)

