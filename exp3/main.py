import argparse
import struct
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split

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

def resnet_block(x, filters, kernel_size=3, stride=1, conv_shortcut=True, name=None):
    bn_axis = 3 if tf.keras.backend.image_data_format() == 'channels_last' else 1

    if conv_shortcut:
        shortcut = layers.Conv2D(filters, 1, strides=stride, name=name + '_0_conv')(x)
        shortcut = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name=name + '_0_bn')(shortcut)
    else:
        shortcut = x

    x = layers.Conv2D(filters, kernel_size, strides=stride, padding='same', name=name + '_1_conv')(x)
    x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name=name + '_1_bn')(x)
    x = layers.Activation('relu', name=name + '_1_relu')(x)

    x = layers.Conv2D(filters, kernel_size, padding='same', name=name + '_2_conv')(x)
    x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name=name + '_2_bn')(x)

    x = layers.add([x, shortcut], name=name + '_add')
    x = layers.Activation('relu', name=name + '_out')(x)
    return x

def resnet_model(input_shape, num_classes=10):
    inputs = tf.keras.Input(shape=input_shape)
    x = layers.Conv2D(64, 7, strides=2, padding='same', name='conv1_conv')(inputs)
    x = layers.BatchNormalization(axis=3, epsilon=1.001e-5, name='conv1_bn')(x)
    x = layers.Activation('relu', name='conv1_relu')(x)
    x = layers.MaxPooling2D(3, strides=2, padding='same', name='pool1_pool')(x)

    # 使用四个残差块
    for i in range(4):
        x = resnet_block(x, 64, conv_shortcut=False, name=f'res2a_branch_{i}')

    x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
    x = layers.Dense(num_classes, activation='softmax', name='fc10')(x)

    model = models.Model(inputs, x, name='resnet')
    return model

# 创建 ArgumentParser 对象
parser = argparse.ArgumentParser(description="Add two numbers")

# 添加命令行参数
parser.add_argument("--model", type=str, required=True, choices=["lenet", "alexnet", "resnet", "moblienet"])
parser.add_argument("--learning_rate", type=float, required=True)
parser.add_argument("--dropout", type=float, required=True)
parser.add_argument("--epochs", type=int, required=True)

# 解析命令行参数
args = parser.parse_args()
model = args.model


# 加载MNIST训练集图像
all_images = load_mnist_images('train-images.idx3-ubyte')
# 加载MNIST训练集标签
all_labels = load_mnist_labels('train-labels.idx1-ubyte')
# 划分训练集和验证集
train_images, val_images, train_labels, val_labels = train_test_split(
    all_images, all_labels, test_size=0.2, random_state=42)
# 将标签进行独热编码
train_labels_one_hot = tf.keras.utils.to_categorical(train_labels, 10)
val_labels_one_hot = tf.keras.utils.to_categorical(val_labels, 10)


if model == 'lenet' :
    print('############################')
    print('#    Using LeNet!!!!!!!!   #')
    print('############################')

    # 设置参数
    learning_rate = args.learning_rate  # 设置你想要的学习率
    epoch = args.epochs
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    dropout = args.dropout

    # 构建LeNet模型
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(6, kernel_size=(5, 5), activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.AveragePooling2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(16, kernel_size=(5, 5), activation='relu'),
        tf.keras.layers.AveragePooling2D(pool_size=(2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(120, activation='relu'),
        tf.keras.layers.Dropout(dropout),  # 添加 Dropout 层，参数是 Dropout 的比例
        tf.keras.layers.Dense(84, activation='relu'),
        tf.keras.layers.Dropout(dropout),  # 添加 Dropout 层，参数是 Dropout 的比例
        tf.keras.layers.Dense(10, activation='softmax')
    ])


    # 编译模型，使用指定的优化器
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    # 数据预处理
    train_images = train_images.reshape(-1, 28, 28, 1).astype('float32') / 255.0
    val_images = val_images.reshape(-1, 28, 28, 1).astype('float32') / 255.0

    # 训练模型
    history = model.fit(train_images, train_labels_one_hot, epochs=epoch,
                        validation_data=(val_images, val_labels_one_hot))

    # 输出最终验证集准确率
    val_loss, val_acc = model.evaluate(val_images, val_labels_one_hot)
    print(f"Final Validation Accuracy of LeNet : {val_acc * 100:.2f}%")

    # 加载MNIST测试集图像
    test_images = load_mnist_images('t10k-images.idx3-ubyte')
    test_images = tf.expand_dims(test_images, axis=-1)

    # 加载MNIST测试集标签
    test_labels = load_mnist_labels('t10k-labels.idx1-ubyte')
    test_labels_one_hot = tf.keras.utils.to_categorical(test_labels, 10)

    # 评估模型在测试集上的性能
    test_loss, test_acc = model.evaluate(test_images, test_labels_one_hot)
    print(f"Test Accuracy of LeNet : {test_acc * 100:.2f}%")


elif model == 'alexnet' :

    print('############################')
    print('#    Using AlexNet!!!!!!!! #')
    print('############################')

    # 设置参数
    learning_rate = args.learning_rate  # 设置你想要的学习率
    epoch = args.epochs
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    dropout = args.dropout

    # 构建AlexNet模型
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(96, kernel_size=(3, 3), strides=(1, 1), activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
        tf.keras.layers.Conv2D(256, kernel_size=(3, 3), padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
        tf.keras.layers.Conv2D(384, kernel_size=(3, 3), padding='same', activation='relu'),
        tf.keras.layers.Conv2D(384, kernel_size=(3, 3), padding='same', activation='relu'),
        tf.keras.layers.Conv2D(256, kernel_size=(3, 3), padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(4096, activation='relu'),
        tf.keras.layers.Dropout(dropout),
        tf.keras.layers.Dense(4096, activation='relu'),
        tf.keras.layers.Dropout(dropout),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    # 编译模型，使用指定的优化器
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    # 数据预处理
    train_images = train_images.reshape(-1, 28, 28, 1).astype('float32') / 255.0
    val_images = val_images.reshape(-1, 28, 28, 1).astype('float32') / 255.0

    # 训练模型
    history = model.fit(train_images, train_labels_one_hot, epochs=epoch,
                        validation_data=(val_images, val_labels_one_hot))

    # 输出最终验证集准确率
    val_loss, val_acc = model.evaluate(val_images, val_labels_one_hot)
    print(f"Final Validation Accuracy of AlexNet : {val_acc * 100:.2f}%")

    # 加载MNIST测试集图像
    test_images = load_mnist_images('t10k-images.idx3-ubyte')
    test_images = tf.expand_dims(test_images, axis=-1)

    # 加载MNIST测试集标签
    test_labels = load_mnist_labels('t10k-labels.idx1-ubyte')
    test_labels_one_hot = tf.keras.utils.to_categorical(test_labels, 10)

    # 评估模型在测试集上的性能
    test_loss, test_acc = model.evaluate(test_images, test_labels_one_hot)
    print(f"Test Accuracy of AlexNet : {test_acc * 100:.2f}%")


elif model == 'resnet' :

    print('############################')
    print('#    Using ResNet!!!!!!!!  #')
    print('############################')

    # 创建ResNet模型
    input_shape = (28, 28, 1)
    num_classes = 10
    resnet_model = resnet_model(input_shape, num_classes)
    epoch = args.epochs

    # 设置参数
    learning_rate = args.learning_rate # 设置你想要的学习率
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    # 编译模型，使用指定的优化器
    resnet_model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    # 训练模型
    history_resnet = resnet_model.fit(train_images, train_labels_one_hot, epochs= epoch,
                                      validation_data=(val_images, val_labels_one_hot))

    # 输出最终验证集准确率
    val_loss_resnet, val_acc_resnet = resnet_model.evaluate(val_images, val_labels_one_hot)
    print(f"Final Validation Accuracy of ResNet : {val_acc_resnet * 100:.2f}%")

    # 评估模型在测试集上的性能
    test_images = load_mnist_images('t10k-images.idx3-ubyte')
    test_images = tf.expand_dims(test_images, axis=-1)

    # 加载MNIST测试集标签
    test_labels = load_mnist_labels('t10k-labels.idx1-ubyte')
    test_labels_one_hot = tf.keras.utils.to_categorical(test_labels, 10)

    test_loss_resnet, test_acc_resnet = resnet_model.evaluate(test_images, test_labels_one_hot)
    print(f"Test Accuracy of ResNet : {test_acc_resnet * 100:.2f}%")


elif model == 'moblienet' :

    print('############################')
    print('#    Using mobileNet!!!!!  #')
    print('############################')

    # 设置参数
    learning_rate = args.learning_rate  # 设置你想要的学习率
    epoch = args.epochs
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    dropout = args.dropout


    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(dropout),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])


    train_images = train_images.reshape(-1, 28, 28, 1).astype('float32') / 255.0
    val_images = val_images.reshape(-1, 28, 28, 1).astype('float32') / 255.0


    history = model.fit(train_images, train_labels_one_hot, epochs=epoch,
                        validation_data=(val_images, val_labels_one_hot))


    val_loss, val_acc = model.evaluate(val_images, val_labels_one_hot)
    print(f"Final Validation Accuracy of MoblieNet : {val_acc * 100:.2f}%")

    test_images = load_mnist_images('t10k-images.idx3-ubyte')
    test_images = tf.expand_dims(test_images, axis=-1)
    test_labels = load_mnist_labels('t10k-labels.idx1-ubyte')
    test_labels_one_hot = tf.keras.utils.to_categorical(test_labels, 10)

    # Evaluate model performance on the test set
    test_loss, test_acc = model.evaluate(test_images, test_labels_one_hot)
    print(f"Test Accuracy of MoblieNet : {test_acc * 100:.2f}%")
