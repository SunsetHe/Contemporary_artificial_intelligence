import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import nltk
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer


# 定义 RNN 模型
class RNNModel(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_size=512, num_layers=2):
        super(RNNModel, self).__init__()
        self.embedding = nn.Embedding(input_dim, hidden_size)
        self.rnn_encoder = nn.RNN(hidden_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.rnn_decoder = nn.RNN(hidden_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_dim)

    def forward(self, src, tgt):
        src = self.embedding(src)
        tgt = self.embedding(tgt)

        _, hidden = self.rnn_encoder(src)
        output, _ = self.rnn_decoder(tgt, hidden)

        output = self.fc(output)
        return output


class CustomDataset(Dataset):
    def __init__(self, df, max_seq_length):
        self.df = df
        self.max_seq_length = max_seq_length  # 设定一个最大序列长度

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # 返回输入和输出序列的张量，确保它们具有相同的长度
        input_sequence = [int(token) for token in self.df['description'].iloc[idx].split()]
        output_sequence = [int(token) for token in self.df['diagnosis'].iloc[idx].split()]

        # 使用padding确保长度一致
        input_sequence = input_sequence[:self.max_seq_length] + [0] * (self.max_seq_length - len(input_sequence))
        output_sequence = output_sequence[:self.max_seq_length] + [0] * (self.max_seq_length - len(output_sequence))

        return torch.tensor(input_sequence), torch.tensor(output_sequence)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("PyTorch is using:", device)

# 读取 CSV 文件
csv_path = "train.csv"  # 替换为实际的文件路径
df = pd.read_csv(csv_path)

# 划分数据集
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

# 创建数据集和 DataLoader，设置max_seq_length
max_seq_length = 150  # 根据实际情况设置最大序列长度
train_dataset = CustomDataset(train_df, max_seq_length)
val_dataset = CustomDataset(val_df, max_seq_length)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# 初始化模型和优化器
input_dim = len(set(df['description'].str.cat().split()))  # 根据数据集中的词汇数初始化输入维度
output_dim = len(set(df['diagnosis'].str.cat().split()))  # 根据数据集中的词汇数初始化输出维度

model = RNNModel(input_dim, output_dim)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# 训练模型
num_epochs = 1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model.to(device)
criterion.to(device)

for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    for input_seq, target_seq in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
        input_seq, target_seq = input_seq.to(device), target_seq.to(device)
        optimizer.zero_grad()
        output = model(input_seq, target_seq)
        loss = criterion(output.view(-1, output_dim), target_seq.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    average_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {average_loss}")

# 在验证集上评估模型
# 在验证集上评估模型
model.eval()
total_val_loss = 0
predictions = []
references = []

with torch.no_grad():
    for input_seq, target_seq in tqdm(val_loader, desc="Validation"):
        input_seq, target_seq = input_seq.to(device), target_seq.to(device)

        # 使用模型生成序列（关闭teacher forcing）
        output = model(input_seq, torch.zeros_like(target_seq))

        # 计算损失
        loss = criterion(output[:, 1:].contiguous().view(-1, output_dim), target_seq[:, 1:].contiguous().view(-1))
        total_val_loss += loss.item()

        # 记录模型的预测和真实参考
        predictions.extend(output.argmax(dim=2)[:, 1:].tolist())
        references.extend(target_seq[:, 1:].tolist())

average_val_loss = total_val_loss / len(val_loader)
print(f"Validation Loss: {average_val_loss}")

# Function to calculate BLEU score
def calculate_bleu_score(predictions, references):
    bleu_scores = []
    for i in range(len(predictions)):
        reference = [str(ref) for ref in references[i]]
        prediction = [str(pred) for pred in predictions[i]]
        score = sentence_bleu([reference], prediction)
        bleu_scores.append(score)
    return sum(bleu_scores) / len(bleu_scores)

# Evaluate the model on the validation set
bleu_scores = calculate_bleu_score(predictions, references)
print(f"BLEU Score on Validation Set: {bleu_scores}")