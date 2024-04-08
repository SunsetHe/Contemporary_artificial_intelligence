import struct
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split


def load_mnist_images(filename):
    with open(filename, 'rb') as f:
        magic, num_images, rows, cols = struct.unpack('>IIII', f.read(16))
        images = np.fromfile(f, dtype=np.uint8).reshape(num_images, rows, cols)
    return images

def load_mnist_labels(filename):
    with open(filename, 'rb') as f:
        magic, num_labels = struct.unpack('>II', f.read(8))
        labels = np.fromfile(f, dtype=np.uint8)
    return labels

# 加载MNIST训练集图像
all_images = load_mnist_images('train-images.idx3-ubyte')

# 加载MNIST训练集标签
all_labels = load_mnist_labels('train-labels.idx1-ubyte')


# Split into training and validation sets
train_images, val_images, train_labels, val_labels = train_test_split(
    all_images, all_labels, test_size=0.2, random_state=42)

# One-hot encode labels
train_labels_one_hot = tf.keras.utils.to_categorical(train_labels, 10)
val_labels_one_hot = tf.keras.utils.to_categorical(val_labels, 10)

# Build a custom CNN model for single-channel images
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Set learning rate
learning_rate = 0.001
epoch = 1
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

# Compile model
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Data preprocessing
train_images = train_images.reshape(-1, 28, 28, 1).astype('float32') / 255.0
val_images = val_images.reshape(-1, 28, 28, 1).astype('float32') / 255.0

# Train the model
history = model.fit(train_images, train_labels_one_hot, epochs=epoch, validation_data=(val_images, val_labels_one_hot))

# Output final validation accuracy
val_loss, val_acc = model.evaluate(val_images, val_labels_one_hot)
print(f"Final Validation Accuracy: {val_acc * 100:.2f}%")

test_images = load_mnist_images('t10k-images.idx3-ubyte')
test_images = tf.expand_dims(test_images, axis=-1)
test_labels = load_mnist_labels('t10k-labels.idx1-ubyte')
test_labels_one_hot = tf.keras.utils.to_categorical(test_labels, 10)

# Evaluate model performance on the test set
test_loss, test_acc = model.evaluate(test_images, test_labels_one_hot)
print(f"Test Accuracy: {test_acc * 100:.2f}%")