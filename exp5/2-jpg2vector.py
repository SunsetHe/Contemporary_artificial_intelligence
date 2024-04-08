# -*- coding: utf-8 -*-
import pickle
import torch
from torchvision import transforms

# TODO 读取保存的6个变量
# 读取保存的变量
with open('train_valid_data.pkl', 'rb') as file:
    train_jpg, valid_jpg, train_text, valid_text, train_label, valid_label = pickle.load(file)

with open('test_data.pkl', 'rb') as file:
    test_guid, test_jpg, test_text = pickle.load(file)

class SimpleCNN(torch.nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=2, kernel_size=3, stride=1, padding=1)
        self.relu1 = torch.nn.ReLU()
        self.pool1 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = torch.nn.Conv2d(in_channels=2, out_channels=2, kernel_size=3, stride=1, padding=1)
        self.relu2 = torch.nn.ReLU()
        self.pool2 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = torch.nn.Flatten()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = self.flatten(x)
        return x

# 将模型移到GPU上
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cnn_model = SimpleCNN().to(device)

# 打印模型结构
print(cnn_model)

# TODO 使用CNN将图像转换为特征向量
def get_image_vectors(model, images, device):
    model.eval()
    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])

    with torch.no_grad():
        # 转换图像为Tensor
        image_tensors = [transform(image).to(device) for image in images if transform(image) is not None]
        if not image_tensors:
            raise ValueError("No valid image tensors found.")
        images_tensor = torch.stack(image_tensors)

        # 获取特征向量
        features = model(images_tensor)
    return features

train_jpg_vector = get_image_vectors(cnn_model, train_jpg, device)
valid_jpg_vector = get_image_vectors(cnn_model, valid_jpg, device)
test_jpg_vector = get_image_vectors(cnn_model, test_jpg, device)

# TODO 打印上面所有向量的长度并检查其是否一致
print("Train Image Vectors Length:", train_jpg_vector[0])
print("Valid Image Vectors Length:", len(valid_jpg_vector[0]))
print("Train Image Vectors Length:", train_jpg_vector[0].tolist())

# TODO 将特征向量保存至本地
torch.save(train_jpg_vector, 'train_jpg_vector.pt')
torch.save(valid_jpg_vector, 'valid_jpg_vector.pt')
torch.save(test_jpg_vector, 'test_jpg_vector.pt')

# TODO 读取本地的特征向量
loaded_train_jpg_vector = torch.load('train_jpg_vector.pt')
loaded_valid_jpg_vector = torch.load('valid_jpg_vector.pt')
loaded_test_jpg_vector = torch.load('test_jpg_vector.pt')

print("Train Image Vectors Length:", train_jpg_vector[0])
print("Valid Image Vectors Length:", len(valid_jpg_vector[0]))
print("Train Image Vectors Length:", train_jpg_vector[0].tolist())
print("Test Image Vectors Length:", test_jpg_vector[0])

