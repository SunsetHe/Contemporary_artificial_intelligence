import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import pandas as pd
import nltk
from nltk.translate.bleu_score import sentence_bleu

# Define RNN-Transformer Model
class RNNEncoder(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, num_layers):
        super(RNNEncoder, self).__init__()
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_dim, num_layers, batch_first=True)

    def forward(self, x):
        embedded = self.embedding(x)
        output, hidden = self.rnn(embedded)
        return output, hidden

class TransformerDecoder(nn.Module):
    def __init__(self, output_dim, embedding_dim, hidden_dim, num_layers):
        super(TransformerDecoder, self).__init__()
        self.embedding = nn.Embedding(output_dim, embedding_dim)
        self.transformer = nn.Transformer(embedding_dim, nhead=8, num_encoder_layers=6, num_decoder_layers=6)
        self.fc = nn.Linear(embedding_dim, output_dim)

    def forward(self, src, tgt):
        src = self.embedding(src)
        tgt = self.embedding(tgt)
        output = self.transformer(src, tgt)
        output = self.fc(output)
        return output

class Seq2SeqModel(nn.Module):
    def __init__(self, input_dim, output_dim, embedding_dim, hidden_dim, num_layers):
        super(Seq2SeqModel, self).__init__()
        self.encoder = RNNEncoder(input_dim, embedding_dim, hidden_dim, num_layers)
        self.decoder = TransformerDecoder(output_dim, embedding_dim, hidden_dim, num_layers)

    def forward(self, src, tgt):
        encoder_output, encoder_hidden = self.encoder(src)
        decoder_output = self.decoder(encoder_output, tgt)
        return decoder_output

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

# Custom Dataset and DataLoader remain the same
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


# Initialize model and optimizer
embedding_dim = 512
hidden_dim = 256
num_layers = 2

model = Seq2SeqModel(input_dim, output_dim, embedding_dim, hidden_dim, num_layers)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Training and validation loops remain the same

# Rest of the code remains the same
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
model.eval()
total_val_loss = 0
predictions = []
references = []

with torch.no_grad():
    for input_seq, target_seq in tqdm(val_loader, desc="Validation"):
        input_seq, target_seq = input_seq.to(device), target_seq.to(device)
        output = model(input_seq, target_seq)
        loss = criterion(output.view(-1, output_dim), target_seq.view(-1))
        total_val_loss += loss.item()

        # 记录模型的预测和真实参考
        predictions.extend(output.argmax(dim=2).tolist())
        references.extend(target_seq.tolist())

average_val_loss = total_val_loss / len(val_loader)
print(f"Validation Loss: {average_val_loss}")





# 设置nltk的SmoothingFunction
smooth_func = nltk.translate.bleu_score.SmoothingFunction().method1

# 将预测和参考转换为字符串
predictions_str = [' '.join(map(str, seq)) for seq in predictions]
references_str = [' '.join(map(str, seq)) for seq in references]

# 计算BLEU-4得分
bleu_scores = [sentence_bleu([reference.split()], prediction.split(), smoothing_function=smooth_func) for
               reference, prediction in zip(references_str, predictions_str)]

# 输出平均BLEU-4得分
average_bleu_score = sum(bleu_scores) / len(bleu_scores)
print(f"Average BLEU-4 Score: {average_bleu_score}")





from rouge_score import rouge_scorer

# 定义ROUGE评估函数
def rouge_evaluation(predictions, references):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge1_scores = []
    rouge2_scores = []
    rougeL_scores = []

    for pred, ref in zip(predictions, references):
        pred_text = ' '.join(map(str, pred)).strip()
        ref_text = ' '.join(map(str, ref)).strip()

        scores = scorer.score(pred_text, ref_text)

        rouge1_scores.append(scores['rouge1'].fmeasure)
        rouge2_scores.append(scores['rouge2'].fmeasure)
        rougeL_scores.append(scores['rougeL'].fmeasure)

    average_rouge1 = sum(rouge1_scores) / len(rouge1_scores)
    average_rouge2 = sum(rouge2_scores) / len(rouge2_scores)
    average_rougeL = sum(rougeL_scores) / len(rougeL_scores)

    return average_rouge1, average_rouge2, average_rougeL

# 使用ROUGE评估
rouge1, rouge2, rougeL = rouge_evaluation(predictions, references)

print(f"ROUGE-1 Score: {rouge1}")
print(f"ROUGE-2 Score: {rouge2}")
print(f"ROUGE-L Score: {rougeL}")
