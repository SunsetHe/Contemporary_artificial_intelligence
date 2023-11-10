import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertTokenizer, BertModel
from sklearn.metrics import accuracy_score
import time
import json

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

# 超参数
hidden_layer = 1 # 当前隐藏层数 3
input_dim = 768  # BERT的隐藏状态大小
hidden_dim = 200 # 神经元数量
output_dim = 10  # 类别数
learning_rate = 0.1
epochs = 300


# 定义MLP模型
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        #self.fc2 = nn.Linear(hidden_dim, hidden_dim)  # 第二个隐藏层
        #self.fc3 = nn.Linear(hidden_dim, hidden_dim)  # 第三个隐藏层
        self.fc4 = nn.Linear(hidden_dim, output_dim)  # 输出层

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        #x = torch.relu(self.fc2(x))
        #x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x

print('---------------------------------')



# 提取训练和验证数据的文本向量化代码，只运行一次
# 准备训练数据
start_time = time.time() # 记录文本向量化的时间
train_vectors = []  # 用于存储训练集的文本向量
train_labels = []  # 用于存储训练集的标签

for sample in train_samples:
    raw_text = sample['raw']
    inputs = tokenizer(raw_text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    hidden_states = outputs.last_hidden_state
    # 使用BERT的隐藏状态作为文本特征
    text_vector = hidden_states.mean(dim=1).squeeze()  # 使用均值池化
    label = sample['label']
    train_vectors.append(text_vector)
    train_labels.append(label)

train_vectors = torch.stack(train_vectors)
train_labels = torch.tensor(train_labels, dtype=torch.long).to(device)

# 准备验证数据
validation_vectors = []  # 用于存储验证集的文本向量
validation_labels = []  # 用于存储验证集的标签

for sample in validation_samples:
    raw_text = sample['raw']
    inputs = tokenizer(raw_text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    hidden_states = outputs.last_hidden_state
    text_vector = hidden_states.mean(dim=1).squeeze()
    label = sample['label']
    validation_vectors.append(text_vector)
    validation_labels.append(label)

validation_vectors = torch.stack(validation_vectors)
validation_labels = torch.tensor(validation_labels, dtype=torch.long).to(device)

end_time = time.time()
print(f'文本向量化用时：{end_time - start_time} 秒')
print('---------------------------------')
#
#
#
# # 保存向量化后的训练和验证数据
# torch.save(train_vectors, 'train_vectors.pt')
# torch.save(train_labels, 'train_labels.pt')
# torch.save(validation_vectors, 'validation_vectors.pt')
# torch.save(validation_labels, 'validation_labels.pt')
# print('向量化文本已保存')
# print('---------------------------------')




# 读取已存储的向量化后的数据
# train_vectors = torch.load('train_vectors.pt')
# train_labels = torch.load('train_labels.pt')
# validation_vectors = torch.load('validation_vectors.pt')
# validation_labels = torch.load('validation_labels.pt')
# print(f'向量化文本已读取')
# print('---------------------------------')



# 训练和验证
# 打印超参数信息
# 打印超参数信息
print('# 超参数')
print(f'hidden_layer 隐藏层 = {hidden_layer}')
print(f'input_dim = {input_dim}')
print(f'hidden_dim 神经元数量 = {hidden_dim}')
print(f'output_dim = {output_dim}')
print(f'learning_rate 学习率= {learning_rate}')
print(f'epochs 训练周期 = {epochs}')

# 初始化MLP模型
start_time = time.time()
print('start training')
mlp_model = MLP(input_dim, hidden_dim, output_dim).to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(mlp_model.parameters(), lr=learning_rate)

# 开始训练
for epoch in range(epochs):
    mlp_model.train()
    optimizer.zero_grad()
    outputs = mlp_model(train_vectors)
    loss = criterion(outputs, train_labels)
    loss.backward()
    optimizer.step()

end_time = time.time()
print(f'训练时间：{end_time - start_time} 秒')

start_time = time.time()
mlp_model.eval()
with torch.no_grad():
    outputs = mlp_model(validation_vectors)
    _, predicted = torch.max(outputs, 1)
    accuracy = accuracy_score(validation_labels.cpu().numpy(), predicted.cpu().numpy())

end_time = time.time()
print(f'验证时间：{end_time - start_time} 秒')
print(f'验证集准确率: {accuracy}')
print('---------------------------------')


# 读取测试数据
test_data_path = 'test.txt'
with open(test_data_path, "r", encoding="utf-8") as file:
    lines = file.readlines()

# 提取测试文本
test_texts = []
test_ids = []  # 用于存储测试数据的ID

for i, line in enumerate(lines[1:]):  # 从第二行开始读取文本
    parts = line.strip().split(',', 1)
    if len(parts) == 2:
        text_id = parts[0]
        text = parts[1]
        test_ids.append(text_id)
        test_texts.append(text)
print('训练数据载入成功')
# 向量化测试文本
test_vectors = []  # 用于存储测试集的文本向量

for text in test_texts:
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    hidden_states = outputs.last_hidden_state
    # 使用BERT的隐藏状态作为文本特征
    text_vector = hidden_states.mean(dim=1).squeeze()  # 使用均值池化
    test_vectors.append(text_vector)

test_vectors = torch.stack(test_vectors)

# 保存向量化之后的文本，让下次训练后的测试阶段可以直接读取，不需要重新向量化
#torch.save(test_vectors, 'test_vectors.pt')
#print('向量化后的测试文本已保存')

# 读取已存储的向量化文本
#test_vectors = torch.load('test_vectors.pt')

# 使用训练好的模型进行预测
mlp_model.eval()
with torch.no_grad():
    test_outputs = mlp_model(test_vectors)
    _, predicted_labels = torch.max(test_outputs, 1)

# 保存预测结果到result.txt
result_path = "result_MLP.txt"
with open(result_path, "w", encoding="utf-8") as result_file:
    result_file.write("id,pred\n")
    for i, label in enumerate(predicted_labels):
        result_file.write(f"{i},{label.item()}\n")

print("预测结果已保存到 result_MLP.txt")