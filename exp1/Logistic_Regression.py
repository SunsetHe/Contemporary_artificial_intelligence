import torch
from transformers import BertTokenizer, BertModel
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import time
import json
import pickle
import numpy as np

print('预处理开始')

import json
import random

# 初始化一个字典用于统计每个label的数量
label_counts = {str(i): 0 for i in range(10)}

# 读取a.txt文件，统计每个label的数量
with open('train_data.txt', 'r') as source_file:
    data = source_file.readlines()

for line in data:
    record = json.loads(line)
    label = str(record['label'])
    label_counts[label] += 1

# 打印每个label的数量
for label, count in label_counts.items():
    print(f"Label {label}: {count} 条数据")

# 随机分割数据并将其写入train.txt
train_data = {}
check_data = {}

for label in label_counts.keys():
    data_for_label = [line for line in data if json.loads(line)['label'] == int(label)]
    random.shuffle(data_for_label)
    total_samples = len(data_for_label)
    train_samples = data_for_label[:int(total_samples * 0.75)]
    check_samples = data_for_label[int(total_samples * 0.75):]

    train_data[label] = train_samples
    check_data[label] = check_samples

# 写入train.txt
with open('train.txt', 'w') as train_file:
    for label in train_data:
        train_file.writelines(train_data[label])

# 写入check.txt
with open('Validation.txt', 'w') as check_file:
    for label in check_data:
        check_file.writelines(check_data[label])

# 读取训练数据
data_path = 'train.txt'
with open(data_path, "r") as file:
    train_data = file.read()
train_samples = [json.loads(line) for line in train_data.split('\n') if line.strip()]
print('训练数据载入成功')

# 读取验证数据
validation_data_path = 'Validation.txt'
with open(validation_data_path, "r") as file:
    validation_data = file.read()
validation_samples = [json.loads(line) for line in validation_data.split('\n') if line.strip()]
print('验证数据载入成功')

# 检查GPU是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("正在使用的device为:", device)

# 加载tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
print('tokenizer已加载')

# 将模型加载到GPU（如果可用）或CPU
model = BertModel.from_pretrained("bert-base-uncased").to(device)
print('模型已加载至device')

print('---------------------------------')


# 向量化训练和验证数据
train_texts = []
train_labels = []
for sample in train_samples:
    raw_text = sample['raw']
    label = sample['label']

    # 使用tokenizer将文本转换为BERT模型输入的格式
    inputs = tokenizer(raw_text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    inputs = {key: value.to(device) for key, value in inputs.items()}

    # 使用BERT模型进行向量化
    with torch.no_grad():
        outputs = model(**inputs)

    # 提取隐藏状态
    hidden_states = outputs.last_hidden_state

    # 使用所有隐藏状态的均值作为文本的向量表示
    text_vector = torch.mean(hidden_states, dim=1).to("cpu").numpy()

    train_texts.append(text_vector)
    train_labels.append(label)

train_texts = np.vstack(train_texts)
train_labels = np.array(train_labels)


validation_texts = []
validation_labels = []
for sample in validation_samples:
    raw_text = sample['raw']
    label = sample['label']

    # 使用tokenizer将文本转换为BERT模型输入的格式
    inputs = tokenizer(raw_text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    inputs = {key: value.to(device) for key, value in inputs.items()}

    # 使用BERT模型进行向量化
    with torch.no_grad():
        outputs = model(**inputs)

    # 提取隐藏状态
    hidden_states = outputs.last_hidden_state

    # 使用所有隐藏状态的均值作为文本的向量表示
    text_vector = torch.mean(hidden_states, dim=1).to("cpu").numpy()

    validation_texts.append(text_vector)
    validation_labels.append(label)

validation_texts = np.vstack(validation_texts)
validation_labels = np.array(validation_labels)

# 保存向量化后的训练和验证数据
with open('train_vectors.pkl', 'wb') as f:
    pickle.dump(train_texts, f)
with open('train_labels.pkl', 'wb') as f:
    pickle.dump(train_labels, f)
with open('validation_vectors.pkl', 'wb') as f:
    pickle.dump(validation_texts, f)
with open('validation_labels.pkl', 'wb') as f:
    pickle.dump(validation_labels, f)

print('向量化文本已保存')
print('---------------------------------')

# 加载向量化后的训练和验证数据
with open('train_vectors.pkl', 'rb') as f:
    train_texts = pickle.load(f)
with open('train_labels.pkl', 'rb') as f:
    train_labels = pickle.load(f)
with open('validation_vectors.pkl', 'rb') as f:
    validation_texts = pickle.load(f)
with open('validation_labels.pkl', 'rb') as f:
    validation_labels = pickle.load(f)




start_time = time.time()
# 初始化逻辑回归模型
logistic_regression = LogisticRegression(max_iter=1000, random_state=42)

# 训练逻辑回归模型
logistic_regression.fit(train_texts, train_labels)

end_time = time.time()
print(f'训练时间：{end_time - start_time} 秒')

# 验证逻辑回归模型
start_time = time.time()

# 使用逻辑回归模型进行预测
predictions = logistic_regression.predict(validation_texts)

# 计算验证集准确率
accuracy = accuracy_score(validation_labels, predictions)

end_time = time.time()
print(f'验证时间：{end_time - start_time} 秒')
print(f'验证集准确率: {accuracy}')






# 读取测试数据
test_data_path = 'test.txt'
with open(test_data_path, "r", encoding='utf-8') as file:
    test_data = file.read().split('\n')[1:]  # Skip the first two lines
test_samples = [line.split(',', 1) for line in test_data if line.strip()]
print('测试数据已载入')

# 初始化空的列表以保存结果
results = []

# 进行文本分类预测
# 进行文本分类预测
for sample in test_samples:
    text_id, text = sample[0], sample[1]

    # 使用tokenizer将文本转换为BERT模型输入的格式
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    inputs = {key: value.to(device) for key, value in inputs.items()}

    # 使用BERT模型进行向量化
    with torch.no_grad():
        outputs = model(**inputs)

    # 提取隐藏状态
    hidden_states = outputs.last_hidden_state

    # 使用所有隐藏状态的均值作为文本的向量表示
    text_vector = torch.mean(hidden_states, dim=1).to("cpu").numpy()

    # 使用逻辑回归模型进行预测
    prediction = logistic_regression.predict(text_vector)

    results.append([text_id, prediction])

# 保存结果到result_Logistic.txt文件
result_file_path = 'result_Logistic.txt'
with open(result_file_path, 'w') as result_file:
    result_file.write("id,pred\n")
    for result in results:
        result_file.write(f"{result[0]},{result[1][0]}\n")

print(f'预测结果已保存到 {result_file_path}')
