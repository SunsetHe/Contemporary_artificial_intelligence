import argparse
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from rouge import Rouge
from transformers import T5Tokenizer, T5ForConditionalGeneration

# argparse相关
parser = argparse.ArgumentParser(description="Add two numbers")
parser.add_argument("--model", type=str, required=True, choices=["rnn_rnn", "LSTM_LSTM", "GRU_GRU", "transf_transf","T5_T5"])
parser.add_argument("--epochs", type=int, required=True)
parser.add_argument("--lr", type=float, required=True)
args = parser.parse_args()

model = args.model
learning_rate = args.lr
epochs = args.epochs

if model == 'rnn_rnn':

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
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    # 训练模型
    num_epochs = epochs
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


elif model == 'LSTM_LSTM':


    # 定义 Transformer 模型
    class Seq2SeqModel(nn.Module):
        def __init__(self, input_dim, output_dim, hidden_dim, num_layers):
            super(Seq2SeqModel, self).__init__()
            self.embedding = nn.Embedding(input_dim, 512)
            self.encoder = nn.LSTM(512, hidden_dim, num_layers)
            self.decoder = nn.LSTM(512, hidden_dim, num_layers)
            self.fc = nn.Linear(hidden_dim, output_dim)

        def forward(self, src, tgt):
            src = self.embedding(src)
            tgt = self.embedding(tgt)

            encoder_output, (encoder_hidden, encoder_cell) = self.encoder(src)
            decoder_output, _ = self.decoder(tgt, (encoder_hidden, encoder_cell))

            output = self.fc(decoder_output)
            return output


    class CustomDataset(Dataset):
        def __init__(self, df, max_seq_length):
            self.df = df
            self.max_seq_length = max_seq_length  # 设定一个最大序列长度

        def __len__(self):
            return len(self.df)

        def __getitem__(self, idx):
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
    hidden_dim = 256  # 根据实际情况设置隐藏层维度和层数
    num_layers = 2

    model = Seq2SeqModel(input_dim, output_dim, hidden_dim, num_layers)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    # 训练模型
    num_epochs = epochs
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



elif model == 'GRU_GRU':

    class Seq2SeqModel(nn.Module):
        def __init__(self, input_dim, output_dim, embedding_dim, hidden_dim, num_layers):
            super(Seq2SeqModel, self).__init__()
            self.embedding = nn.Embedding(input_dim, embedding_dim)
            self.encoder = nn.GRU(embedding_dim, hidden_dim, num_layers=num_layers, batch_first=True)
            self.decoder = nn.GRU(embedding_dim, hidden_dim, num_layers=num_layers, batch_first=True)
            self.fc = nn.Linear(hidden_dim, output_dim)

        def forward(self, src, tgt):
            src = self.embedding(src)
            tgt = self.embedding(tgt)

            _, encoder_hidden = self.encoder(src)

            decoder_output, _ = self.decoder(tgt, encoder_hidden)

            output = self.fc(decoder_output)
            return output


    class CustomDataset(Dataset):
        def __init__(self, df, max_seq_length):
            self.df = df
            self.max_seq_length = max_seq_length

        def __len__(self):
            return len(self.df)

        def __getitem__(self, idx):
            input_sequence = [int(token) for token in self.df['description'].iloc[idx].split()]
            output_sequence = [int(token) for token in self.df['diagnosis'].iloc[idx].split()]

            input_sequence = input_sequence[:self.max_seq_length] + [0] * (self.max_seq_length - len(input_sequence))
            output_sequence = output_sequence[:self.max_seq_length] + [0] * (self.max_seq_length - len(output_sequence))

            return torch.tensor(input_sequence), torch.tensor(output_sequence)


    # 读取 CSV 文件
    csv_path = "train.csv"
    df = pd.read_csv(csv_path)

    # 划分数据集
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

    # 创建数据集和 DataLoader，设置max_seq_length
    max_seq_length = 150
    train_dataset = CustomDataset(train_df, max_seq_length)
    val_dataset = CustomDataset(val_df, max_seq_length)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # 初始化模型和优化器
    input_dim = len(set(df['description'].str.cat().split()))
    output_dim = len(set(df['diagnosis'].str.cat().split()))
    embedding_dim = 256  # 调整为合适的嵌入维度
    hidden_dim = 512  # 调整为合适的隐藏层维度
    num_layers = 2  # 调整为合适的层数

    model = Seq2SeqModel(input_dim, output_dim, embedding_dim, hidden_dim, num_layers)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    # 训练模型
    num_epochs = epochs
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



elif model == 'transf_transf':


    # 定义 Transformer 模型
    class TransformerModel(nn.Module):
        def __init__(self, input_dim, output_dim):
            super(TransformerModel, self).__init__()

            self.embedding = nn.Embedding(input_dim, 512)
            self.transformer = nn.Transformer(512, nhead=8, num_encoder_layers=6, num_decoder_layers=6)
            self.fc = nn.Linear(512, output_dim)

        def forward(self, src, tgt):
            src = self.embedding(src)
            tgt = self.embedding(tgt)
            output = self.transformer(src, tgt)
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

    model = TransformerModel(input_dim, output_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    # 训练模型
    num_epochs = epochs
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

            # 使用模型生成序列
            output = model(input_seq, target_seq)  # 关闭teacher forcing

            # 计算损失
            loss = criterion(output[:, 1:].contiguous().view(-1, output_dim), target_seq[:, 1:].contiguous().view(-1))
            total_val_loss += loss.item()

            # 记录模型的预测和真实参考
            predictions.extend(output.argmax(dim=2)[:, 1:].tolist())
            references.extend(target_seq[:, 1:].tolist())

    average_val_loss = total_val_loss / len(val_loader)
    print(f"Validation Loss: {average_val_loss}")



elif model == 'T5_T5':

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
    csv_path = "/kaggle/input/data111/train.csv"  # 替换为实际的文件路径
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
    model = T5Model('t5-small')
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # 训练模型
    num_epochs = epochs
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

            # 生成序列而不是返回 logits
            generated_ids = model.t5.generate(input_ids, max_length=50, num_beams=2, length_penalty=0.6)

            # 计算损失
            loss = model(input_ids, labels=target_ids)
            total_val_loss += loss.item()

            # 记录模型的预测和真实参考
            predictions.extend(generated_ids.tolist())
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