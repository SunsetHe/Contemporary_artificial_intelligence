import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from transformers import T5Tokenizer, T5ForConditionalGeneration
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
import nltk


# 定义 T5 模型
class T5Model(nn.Module):
    def __init__(self, model_name):
        super(T5Model, self).__init__()
        self.t5 = T5ForConditionalGeneration.from_pretrained(model_name)

    def forward(self, input_ids, labels=None):
        if labels is not None:
            # During training, return the loss
            outputs = self.t5(input_ids, labels=labels)
            return outputs.loss
        else:
            # During inference, return the logits
            return self.t5(input_ids).logits


class CustomDataset(Dataset):
    def __init__(self, df, max_seq_length, tokenizer):
        self.df = df
        self.max_seq_length = max_seq_length
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        input_text = self.df['description'].iloc[idx]
        output_text = self.df['diagnosis'].iloc[idx]

        # 使用tokenizer编码文本
        input_ids = self.tokenizer.encode(input_text, return_tensors='pt', max_length=self.max_seq_length,
                                          truncation=True)
        output_ids = self.tokenizer.encode(output_text, return_tensors='pt', max_length=self.max_seq_length,
                                           truncation=True)

        return input_ids.squeeze(), output_ids.squeeze()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("PyTorch is using:", device)

# 读取 CSV 文件
csv_path = "train.csv"  # 替换为实际的文件路径
df = pd.read_csv(csv_path)

# 使用 T5 的 tokenizer
tokenizer = T5Tokenizer.from_pretrained('t5-small')

# 划分数据集
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

# 创建数据集和 DataLoader，设置max_seq_length
max_seq_length = 150  # 根据实际情况设置最大序列长度
# 创建数据集和 DataLoader，设置max_seq_length
train_dataset = CustomDataset(train_df, max_seq_length, tokenizer)
val_dataset = CustomDataset(val_df, max_seq_length, tokenizer)

# 使用 pad_sequence 来处理变长序列
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=lambda x: (
torch.nn.utils.rnn.pad_sequence([item[0] for item in x], batch_first=True),
torch.nn.utils.rnn.pad_sequence([item[1] for item in x], batch_first=True)))
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=lambda x: (
torch.nn.utils.rnn.pad_sequence([item[0] for item in x], batch_first=True),
torch.nn.utils.rnn.pad_sequence([item[1] for item in x], batch_first=True)))

# 初始化模型和优化器
model = T5Model('t5-base')
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 训练模型
num_epochs = 1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model.to(device)
# criterion.to(device)

for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    for input_ids, target_ids in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
        input_ids, target_ids = input_ids.to(device), target_ids.to(device)
        optimizer.zero_grad()

        # 计算损失
        loss = model(input_ids, labels=target_ids)
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
    for input_ids, target_ids in tqdm(val_loader, desc="Validation"):
        input_ids, target_ids = input_ids.to(device), target_ids.to(device)
        output_ids = model(input_ids, labels=target_ids)
        loss = model(input_ids, labels=target_ids)
        total_val_loss += loss.item()

        # 记录模型的预测和真实参考
        predictions.extend(output_ids.argmax(dim=2).tolist())
        references.extend(target_ids.tolist())

average_val_loss = total_val_loss / len(val_loader)
print(f"Validation Loss: {average_val_loss}")

# 设置nltk的SmoothingFunction
smooth_func = SmoothingFunction().method1

# 将预测和参考转换为字符串
predictions_str = [' '.join(map(str, seq)) for seq in predictions]
references_str = [' '.join(map(str, seq)) for seq in references]

# 计算BLEU-4得分
bleu_scores = [sentence_bleu([reference], prediction, smoothing_function=smooth_func) for
               reference, prediction in zip(references_str, predictions_str)]

# 输出平均BLEU-4得分
average_bleu_score = sum(bleu_scores) / len(bleu_scores)
print(f"Average BLEU-4 Score: {average_bleu_score}")

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