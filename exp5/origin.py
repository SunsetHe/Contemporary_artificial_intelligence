# 总4869 训练集4000 测试集511
import csv
from PIL import Image
import io
from sklearn.model_selection import train_test_split

import torch
from torchvision import transforms

from transformers import BertTokenizer, BertModel
import numpy as np

def read_data(data_path, label_path):
    data = []

    with open(label_path, 'r') as file:
        reader = csv.reader(file)
        # 读掉标题行
        header = next(reader)
        for row in reader:
            guid = int(row[0])
            tag = row[1]
            image_path = data_path + f'/{guid}.jpg'
            text_path = data_path + f'/{guid}.txt'

            # 读取图像内容
            with open(image_path, 'rb') as image_file:
                image_data = image_file.read()

            # 将二进制图像数据转换为PIL Image对象
            image = Image.open(io.BytesIO(image_data))

            # 读取文本内容
            with open(text_path, 'r', encoding='gbk', errors='ignore') as text_file:
                text = text_file.read()

            data.append([guid, image, text, tag])

    # 按照guid的大小排序
    data.sort(key=lambda x: x[0])

    return data

# 将情感标签映射为数字
def tag2numlabel(data):
    label_mapping = {"positive": 1, "neutral": 0, "negative": -1}

    data_new = []

    for row in data:
        guid, image, text, tag = row
        numeric_label = label_mapping.get(tag, -1)
        data_new.append([guid, image, text, numeric_label])

    return data_new


data_path = 'data'
train_label_and_index_path = 'train.txt'
test_index_path = 'test_without_label.txt'

train_data_all = read_data(data_path, train_label_and_index_path)
# test_data = read_data(data_path, test_index_path)

train_data_all = tag2numlabel(train_data_all)
# test_data = tag2numlabel(test_data)

# 划分训练集和验证集
train_data, valid_data = train_test_split(train_data_all, test_size=0.2, random_state=42)

train_jpg = [item[1] for item in train_data_all]
valid_jpg = [item[1] for item in valid_data]

train_text = [item[2] for item in train_data_all]
valid_text = [item[2] for item in valid_data]

train_label = [item[3] for item in train_data_all]
valid_label = [item[3] for item in valid_data]

print(train_jpg[0])

#######################################################################################################
# TODO 版本1
#######################################################################################################
# # TODO 使用CNN将图像转换为特征向量
# # 使用CNN将图像转换为特征向量
# def process_image(image_binary, device='cuda' if torch.cuda.is_available() else 'cpu'):
#     transform = transforms.Compose([
#         transforms.Resize((224, 224)),
#         transforms.ToTensor(),
#     ])
#
#     # 使用io.BytesIO将二进制数据转换为Image对象
#     image = Image.open(io.BytesIO(image_binary)).convert('RGB')
#     image = transform(image)
#
#     # 将图像数据移动到GPU上
#     image = image.to(device)
#     return image
#
# train_jpg_vector = [process_image(row[1]) for row in train_data]
# valid_jpg_vector = [process_image(row[1]) for row in valid_data]
#
# # TODO 打印上面所有向量的长度并检查其是否一致
# print("图像向量长度:", len(train_jpg_vector[0]), len(valid_jpg_vector[0]))
#
# # TODO 使用bert将文本转换为特征向量
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# bert_model = BertModel.from_pretrained('bert-base-uncased')
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# bert_model.to(device)
# def process_text(text, device=device):
#     tokens = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
#     # 将tokens移动到GPU上
#     for key in tokens:
#         tokens[key] = tokens[key].to(device)
#     outputs = bert_model(**tokens)
#     # 将结果移动到 CPU 上，然后转换为 NumPy 数组
#     result = torch.mean(outputs.last_hidden_state, dim=1).squeeze().detach().cpu().numpy()
#
#     return result
#
# train_text_vector = [process_text(row[2]) for row in train_data]
# valid_text_vector = [process_text(row[2]) for row in valid_data]
#
# # TODO 打印上面所有向量的长度并检查其是否一致
# print("文本向量长度:", len(train_text_vector[0]), len(valid_text_vector[0]))
#
# # TODO 将图像向量和文本向量拼接起来并对齐
# # train_vector =
# # valid_vector =
# # # 打印上面所有向量的长度并检查其是否一致
# # print("最终向量长度:", len(train_vector[0]), len(valid_vector[0]))
#
# # TODO 将特征向量和数字标签送入transformer进行训练，查看训练集准确率和验证集准确率
#######################################################################################################
# TODO 版本1
#######################################################################################################

#######################################################################################################
# TODO 版本2
#######################################################################################################
# TODO CNN模型定义，使用pytorch，将模型移到gpu上，并使用gpu进行运算
class SimpleCNN(torch.nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=2, kernel_size=3, stride=1, padding=1)
        self.relu1 = torch.nn.ReLU()
        self.pool1 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = torch.nn.Conv2d(in_channels=2, out_channels=1, kernel_size=3, stride=1, padding=1)
        self.relu2 = torch.nn.ReLU()
        self.pool2 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = torch.nn.Flatten()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = self.flatten(x)
        return x

# 将模型移到GPU上
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cnn_model = SimpleCNN().to(device)

# 打印模型结构
print(cnn_model)

# TODO 使用CNN将图像转换为特征向量
def get_image_vectors(model, images, device):
    model.eval()
    transform = transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor()])

    with torch.no_grad():
        # 转换图像为Tensor
        image_tensors = [transform(image).to(device) for image in images if transform(image) is not None]
        if not image_tensors:
            raise ValueError("No valid image tensors found.")
        images_tensor = torch.stack(image_tensors)

        # 获取特征向量
        features = model(images_tensor)
    return features

train_jpg_vector = get_image_vectors(cnn_model, train_jpg, device)
valid_jpg_vector = get_image_vectors(cnn_model, valid_jpg, device)

# TODO 打印上面所有向量的长度并检查其是否一致
print("Train Image Vectors Length:", train_jpg_vector[0])
print("Valid Image Vectors Length:", len(valid_jpg_vector[0]))
print("Train Image Vectors Length:", train_jpg_vector[0].tolist())

# TODO 将特征向量保存至本地
torch.save(train_jpg_vector, 'train_jpg_vector.pt')
torch.save(valid_jpg_vector, 'valid_jpg_vector.pt')

# TODO 读取本地的特征向量
loaded_train_jpg_vector = torch.load('train_jpg_vector.pt')
loaded_valid_jpg_vector = torch.load('valid_jpg_vector.pt')

print("Train Image Vectors Length:", train_jpg_vector[0])
print("Valid Image Vectors Length:", len(valid_jpg_vector[0]))
print("Train Image Vectors Length:", train_jpg_vector[0].tolist())

for i in range(10):
    print(train_jpg_vector[i].tolist())




# # TODO bert模型，使用gpu加速计算
# # 检查GPU是否可用
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print("Using device:", device)
#
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# model = BertModel.from_pretrained('bert-base-uncased').to(device)
#
# def text_to_bert_vector(text):
#     # 使用tokenizer将文本转换为token IDs
#     inputs = tokenizer(text, return_tensors="pt").to(device)
#
#     # 使用BERT模型得到输出
#     outputs = model(**inputs)
#
#     # 提取最后一层的隐藏状态作为特征向量
#     last_hidden_states = outputs.last_hidden_state
#
#     # 取平均值，得到整体文本的特征向量
#     avg_pooling = torch.mean(last_hidden_states, dim=1)
#
#     # 分离张量，释放GPU内存
#     avg_pooling = avg_pooling.detach()
#
#     return avg_pooling
#
# # TODO 使用bert将文本转换为特征向量
# train_text_vector = [text_to_bert_vector(text) for text in train_text]
# valid_text_vector = [text_to_bert_vector(text) for text in valid_text]
#
# # TODO 打印上面所有向量的长度并检查其是否一致
# train_text_vector_lengths = [len(vector[0]) for vector in train_text_vector]
# valid_text_vector_lengths = [len(vector[0]) for vector in valid_text_vector]
#
# # 检查是否所有训练集中的长度都一致
# if all(length == train_text_vector_lengths[0] for length in train_text_vector_lengths):
#     print("All training text vector lengths are consistent.")
# else:
#     print("Training text vector lengths are not consistent.")
#
# # 检查是否所有验证集中的长度都一致
# if all(length == valid_text_vector_lengths[0] for length in valid_text_vector_lengths):
#     print("All validation text vector lengths are consistent.")
# else:
#     print("Validation text vector lengths are not consistent.")
# # TODO 将特征向量和数字标签送入transformer进行训练，查看训练集准确率和验证集准确率

#######################################################################################################
# TODO 版本2
#######################################################################################################