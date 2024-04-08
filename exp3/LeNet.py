import struct
import numpy as np

def load_mnist_images(filename):
    with open(filename, 'rb') as f:
        # 读取文件头
        magic, num_images, rows, cols = struct.unpack('>IIII', f.read(16))
        # 读取图像数据
        images = np.fromfile(f, dtype=np.uint8).reshape(num_images, rows, cols)
    return images

def load_mnist_labels(filename):
    with open(filename, 'rb') as f:
        # 读取文件头
        magic, num_labels = struct.unpack('>II', f.read(8))
        # 读取标签数据
        labels = np.fromfile(f, dtype=np.uint8)
    return labels

# 加载MNIST训练集图像
all_images = load_mnist_images('train-images.idx3-ubyte')

# 加载MNIST训练集标签
all_labels = load_mnist_labels('train-labels.idx1-ubyte')

import tensorflow as tf
from sklearn.model_selection import train_test_split

# 划分训练集和验证集
# 划分训练集和验证集
train_images, val_images, train_labels, val_labels = train_test_split(
    all_images, all_labels, test_size=0.2, random_state=42)


# 将标签进行独热编码
train_labels_one_hot = tf.keras.utils.to_categorical(train_labels, 10)
val_labels_one_hot = tf.keras.utils.to_categorical(val_labels, 10)

# 构建LeNet模型
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(6, kernel_size=(5, 5), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.AveragePooling2D(pool_size=(2, 2)),
    tf.keras.layers.Conv2D(16, kernel_size=(5, 5), activation='relu'),
    tf.keras.layers.AveragePooling2D(pool_size=(2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(120, activation='relu'),
    tf.keras.layers.Dropout(0.5),  # 添加 Dropout 层，参数是 Dropout 的比例
    tf.keras.layers.Dense(84, activation='relu'),
    tf.keras.layers.Dropout(0.5),  # 添加 Dropout 层，参数是 Dropout 的比例
    tf.keras.layers.Dense(10, activation='softmax')
])

# 设置学习率
learning_rate = 0.001  # 设置你想要的学习率
epoch = 10
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

# 编译模型，使用指定的优化器
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# 数据预处理
train_images = train_images.reshape(-1, 28, 28, 1).astype('float32') / 255.0
val_images = val_images.reshape(-1, 28, 28, 1).astype('float32') / 255.0

# 训练模型
history = model.fit(train_images, train_labels_one_hot, epochs= epoch, validation_data=(val_images, val_labels_one_hot))

# 输出最终验证集准确率
val_loss, val_acc = model.evaluate(val_images, val_labels_one_hot)
print(f"Final Validation Accuracy: {val_acc * 100:.2f}%")

# 加载MNIST测试集图像
test_images = load_mnist_images('t10k-images.idx3-ubyte')
test_images = tf.expand_dims(test_images, axis=-1)

# 加载MNIST测试集标签
test_labels = load_mnist_labels('t10k-labels.idx1-ubyte')

test_labels_one_hot = tf.keras.utils.to_categorical(test_labels, 10)

# 评估模型在测试集上的性能
test_loss, test_acc = model.evaluate(test_images, test_labels_one_hot)
print(f"Test Accuracy: {test_acc * 100:.2f}%")