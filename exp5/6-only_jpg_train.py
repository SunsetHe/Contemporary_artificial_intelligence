# -*- coding: utf-8 -*-
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score

train_jpg_vector = torch.load('train_jpg_vector.pt')
valid_jpg_vector = torch.load('valid_jpg_vector.pt')

train_text_vector = torch.load('train_text_vector.pt')
valid_text_vector = torch.load('valid_text_vector.pt')

with open('train_valid_data.pkl', 'rb') as file:
    train_jpg, valid_jpg, train_text, valid_text, train_label, valid_label = pickle.load(file)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

print(train_text_vector[0].shape)
print(train_text_vector[0])
print(len(train_text_vector))

# TODO 构造一个新的训练集的text的特征向量，但是让被拼接的text的向量变为空向量，要求形状与原来一样

new_train_text_vector = [torch.zeros_like(text_vec) for text_vec in train_text_vector]
train_text_vector = new_train_text_vector

print(train_text_vector[0].shape)
print(train_text_vector[0])
print(len(train_text_vector))

# 拼接jpg向量和text向量
# TODO 仍然需要拼接训练jpg和text的向量，
train_combined = [torch.cat((jpg_vec, text_vec.squeeze(0)), dim=0) for jpg_vec, text_vec in zip(train_jpg_vector, train_text_vector)]
valid_combined = [torch.cat((jpg_vec, text_vec.squeeze(0)), dim=0) for jpg_vec, text_vec in zip(valid_jpg_vector, valid_text_vector)]


# TODO 定义模型
class MultiModalTransformer(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MultiModalTransformer, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x

input_size = train_combined[0].shape[0]  # Assuming all tensors in the list have the same size
hidden_size = 64
output_size = 3


train_combined = torch.stack(train_combined)
valid_combined = torch.stack(valid_combined)

model = MultiModalTransformer(input_size, hidden_size, output_size)
model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


train_dataset = TensorDataset(train_combined, torch.tensor(train_label))
valid_dataset = TensorDataset(valid_combined, torch.tensor(valid_label))

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=64, shuffle=False)


# TODO 进行训练
num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    correct_predictions_train = 0
    total_samples_train = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(outputs.data, 1)
        total_samples_train += labels.size(0)
        correct_predictions_train += (predicted == labels).sum().item()


        total_loss += loss.item()

    average_loss = total_loss / len(train_loader)
    accuracy_train = correct_predictions_train / total_samples_train

    # 验证
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in valid_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy_valid = accuracy_score(all_labels, all_preds)

    print(f'Epoch {epoch + 1}/{num_epochs}, Training Loss: {average_loss:.4f}, '
          f'Training Accuracy: {accuracy_train * 100:.2f}%, '
          f'Validation Accuracy: {accuracy_valid * 100:.2f}%')

# Step 5: 保存模型
torch.save(model.state_dict(), f'train_only_jpg_model_epoch_{num_epochs}.pt')

