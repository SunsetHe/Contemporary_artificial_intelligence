# import struct
# import numpy as np
# import matplotlib.pyplot as plt
#
#
# def load_mnist_images(filename):
#     with open(filename, 'rb') as f:
#         # 读取文件头
#         magic, num_images, rows, cols = struct.unpack('>IIII', f.read(16))
#         # 读取图像数据
#         images = np.fromfile(f, dtype=np.uint8).reshape(num_images, rows, cols)
#     return images
#
# # 加载MNIST训练集图像
# train_images = load_mnist_images('t10k-images.idx3-ubyte')
#
# # 显示前两个图像
# for i in range(8):
#     plt.subplot(1, 8, i + 1)
#     plt.imshow(train_images[i], cmap='gray')
#     plt.title(f"Label: {i}")
#     plt.axis('off')
#
# plt.show()
#
# def load_mnist_labels(filename):
#     with open(filename, 'rb') as f:
#         # 读取文件头
#         magic, num_labels = struct.unpack('>II', f.read(8))
#         # 读取标签数据
#         labels = np.fromfile(f, dtype=np.uint8)
#     return labels
#
# # 加载MNIST训练集标签
# train_labels = load_mnist_labels('t10k-labels.idx1-ubyte')
#
# # 显示前8个图像的标签
# for i in range(8):
#     print(f"Image {i + 1} Label: {train_labels[i]}")
#
# # 输出图像的总个数
# print(f"Total number of images: {train_images.shape[0]}")
#
# # 输出标签的总个数
# print(f"Total number of labels: {train_labels.shape[0]}")











import matplotlib.pyplot as plt

# Model names
models = ['LeNet', 'AlexNet', 'ResNet', 'MobileNet']

# Validation and test accuracies
validation_accuracies = [98.37, 98.74, 97.78, 98.50]
test_accuracies = [98.31, 98.03, 97.91, 78.48]

# Bar width
bar_width = 0.35

# Set up positions for bars
bar_positions_validation = list(range(len(models)))
bar_positions_test = [pos + bar_width for pos in bar_positions_validation]

# Create the bar chart
fig, ax = plt.subplots()
bars_validation = ax.bar(bar_positions_validation, validation_accuracies, width=bar_width, label='Validation Accuracy')
bars_test = ax.bar(bar_positions_test, test_accuracies, width=bar_width, label='Test Accuracy')

# Set axis labels and title
ax.set_xlabel('Model')
ax.set_ylabel('Accuracy (%)')
ax.set_title('Validation and Test Accuracies for Different Models')
ax.set_xticks([pos + bar_width / 2 for pos in bar_positions_validation])
ax.set_xticklabels(models)

# Add annotations for each bar
for bar in bars_validation + bars_test:
    height = bar.get_height()
    ax.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),  # 3 points vertical offset
                textcoords="offset points",
                ha='center', va='bottom')

# Move the legend to the left bottom corner
ax.legend(loc='lower left')

# Show the plot
plt.show()

