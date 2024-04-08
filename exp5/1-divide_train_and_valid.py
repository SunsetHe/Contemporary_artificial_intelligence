# -*- coding: utf-8 -*-
import csv
from PIL import Image
import io
from sklearn.model_selection import train_test_split
import pickle

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
    label_mapping = {"positive": 2, "neutral": 1, "negative": 0}

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
test_data = read_data(data_path, test_index_path)

train_data_all = tag2numlabel(train_data_all)
test_data = tag2numlabel(test_data)

# 划分训练集和验证集
train_data, valid_data = train_test_split(train_data_all, test_size=0.2, random_state=42)

train_jpg = [item[1] for item in train_data]
valid_jpg = [item[1] for item in valid_data]
test_jpg = [item[1] for item in test_data]

train_text = [item[2] for item in train_data]
valid_text = [item[2] for item in valid_data]
test_text = [item[2] for item in test_data]

train_label = [item[3] for item in train_data]
valid_label = [item[3] for item in valid_data]

# TODO 保存上面6个变量至本地
# 保存变量至本地
with open('train_valid_data.pkl', 'wb') as file:
    pickle.dump((train_jpg, valid_jpg, train_text, valid_text, train_label, valid_label), file)

test_guid = [item[0] for item in test_data]

with open('test_data.pkl', 'wb') as file:
    pickle.dump((test_guid, test_jpg, test_text), file)