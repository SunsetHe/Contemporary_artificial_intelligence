import torch
from transformers import BertTokenizer, BertModel
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import time
import json
import numpy as np
import pickle

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

# 验证数据向量化
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

# 测试数据向量化
test_data_path = 'test.txt'

with open(test_data_path, 'r', encoding='utf-8') as file:
    lines = file.readlines()

# 获取特征和ID
features = []
ids = []
for line in lines[1:]:
    id, text = line.strip().split(',', 1)
    ids.append(id)
    features.append(text)

# 使用tokenizer将文本转换为BERT模型输入的格式
test_texts = []
for text in features:
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    hidden_states = outputs.last_hidden_state
    text_vector = torch.mean(hidden_states, dim=1).to("cpu").numpy()
    test_texts.append(text_vector)

test_texts = np.vstack(test_texts)

# 保存向量化后的训练，验证，测试数据至本地
# with open('train_texts.pkl', 'wb') as f:
#     pickle.dump(train_texts, f)
# with open('validation_texts.pkl', 'wb') as f:
#     pickle.dump(validation_texts, f)
# with open('test_texts.pkl', 'wb') as f:
#     pickle.dump(test_texts, f)
# with open('train_labels.pkl', 'wb') as f:
#     pickle.dump(train_labels, f)
# with open('validation_labels.pkl', 'wb') as f:
#     pickle.dump(validation_labels, f)

# 载入已保存训练，验证，测试的向量化的数据
# with open('train_texts.pkl', 'rb') as f:
#     train_texts = pickle.load(f)
# with open('validation_texts.pkl', 'rb') as f:
#     validation_texts = pickle.load(f)
# with open('test_texts.pkl', 'rb') as f:
#     test_texts = pickle.load(f)
# with open('train_labels.pkl', 'rb') as f:
#     train_labels = pickle.load(f)
# with open('validation_labels.pkl', 'rb') as f:
#     validation_labels = pickle.load(f)



# 使用SVM模型进行训练
start_time = time.time()
svm_model = SVC(C=1.0, kernel='linear', gamma='scale')  # 可根据需要设置超参数
svm_model.fit(train_texts, train_labels)
end_time = time.time()
print(f'训练时间：{end_time - start_time} 秒')

# 验证SVM模型
start_time = time.time()

# 使用SVM模型进行验证
predictions = svm_model.predict(validation_texts)

# 计算验证集准确率
accuracy = accuracy_score(validation_labels, predictions)

end_time = time.time()
print(f'验证时间：{end_time - start_time} 秒')
print(f'验证集准确率: {accuracy}')





# 使用决策树模型进行预测
predictions = svm_model.predict(test_texts)

# 写入结果到result_svm.txt
result_path = 'result_svm.txt'
with open(result_path, 'w', encoding='utf-8') as result_file:
    result_file.write("id,pred\n")
    for i, pred in zip(ids, predictions):
        result_file.write(f"{i},{pred}\n")
print(f"预测结果已写入 {result_path}")