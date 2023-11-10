import torch
from transformers import BertTokenizer, BertModel
from sklearn.tree import DecisionTreeClassifier  # 使用决策树模型
from sklearn.metrics import accuracy_score
import time
import json
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

# 向量化训练数据
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

validation_data_path = 'Validation.txt'
with open(validation_data_path, "r") as file:
    validation_data = file.read()
validation_samples = [json.loads(line) for line in validation_data.split('\n') if line.strip()]
print('验证数据载入成功')

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

# # 保存向量化后的数据至本地
# train_data_save_path = 'exp1_data/train_vectors.npy'
# validation_data_save_path = 'exp1_data/validation_vectors.npy'
# np.save(train_data_save_path, train_texts)
# np.save(validation_data_save_path, validation_texts)
#
# # 载入已保存的向量化的数据
# loaded_train_texts = np.load(train_data_save_path)
# loaded_validation_texts = np.load(validation_data_save_path)

start_time = time.time()

# 创建并训练决策树模型
decision_tree = DecisionTreeClassifier(max_depth=10)  # 设置决策树的超参数，可以根据需求调整
decision_tree.fit(train_texts, train_labels)

# 验证决策树模型

end_time = time.time()
print(f'训练时间：{end_time - start_time} 秒')

# 使用决策树模型进行预测
predictions = decision_tree.predict(validation_texts)

# 计算验证集准确率
accuracy = accuracy_score(validation_labels, predictions)

end_time = time.time()
print(f'验证时间：{end_time - start_time} 秒')
print(f'验证集准确率: {accuracy}')


# 读取测试数据
test_data_path = 'test.txt'

with open(test_data_path, 'r', encoding='utf-8') as file:
    lines = file.readlines()

# 获取特征和ID
test_ids = []
test_features = []
for line in lines[1:]:
    id, text = line.strip().split(',', 1)
    test_ids.append(id)
    test_features.append(text)

# 使用tokenizer将文本转换为BERT模型输入的格式
test_texts = []
for text in test_features:
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    hidden_states = outputs.last_hidden_state
    text_vector = torch.mean(hidden_states, dim=1).to("cpu").numpy()
    test_texts.append(text_vector)

test_texts = np.vstack(test_texts)

# 使用决策树模型进行预测
test_predictions = decision_tree.predict(test_texts)

# 保存预测结果到result_jueceshu.txt
result_file = open("result_jueceshu.txt", "w")
result_file.write("id,pred\n")
for id, pred in zip(test_ids, test_predictions):
    result_file.write(f"{id},{pred}\n")

result_file.close()
print("预测结果已保存到result_jueceshu.txt")
