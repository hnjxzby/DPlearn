# -*- coding: utf-8 -*-
"""
Created on Tue Feb  8 14:57:29 2022
随着深度学习的发展，Dropout在现代卷积架构中，已经逐步被BN（想要了解BN，大家可以参见我之前写的 深度学习基础系列（七）| Batch Normalization 一文，这里不再赘述）取代，BN也同样拥有不亚于Dropout的正则化效果。
纵观无论是VGG、ResNet、Inception、MobileNetV2等模型，都不见了Dropout踪影。唯独在MobileNetV1模型里，还可以找到Dropout，但不是在卷积层；而且在MobileNetV2后，已经不再有全连接层，而是被全局平均池化层所取代
Dropout VS BatchNormalization
　　我们需要做一个简单实验来验证上述理论的成立，实验分五种测试模型：

没有使用Dropout，也没有使用BN;
使用了Dropout，不使用BN，使训练单元为0的概率为0.2；
使用了Dropout，不使用BN，使训练单元为0的概率为0.5；
使用了Dropout，不使用BN，使训练单元为0的概率为0.8；
使用了BN，不使用Dropout
　　代码如下：
  
  
  
  注意版本更新的问题：我将优化器改成了 Adam
  keras.optimizers.Adam(lr=0.001, decay=1e-6)
  原文的rmsprop（）不被支持了。
  
  opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)
@author: Dell_hnjxzby
"""
#import tensorflow as tf
#import tensorflow.keras as keras

from tensorflow import keras
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from matplotlib import pyplot as plt
import numpy as np

# 为保证公平起见，使用相同的随机种子
np.random.seed(7)
batch_size = 32
num_classes = 10
epochs = 40
data_augmentation = True

# The data, split between train and test sets:
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Convert class vectors to binary class matrices.
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

def model(bn=False, dropout=False, level=0.5):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same', input_shape=x_train.shape[1:]))
    if bn:
        model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3)))
    if bn:
        model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    if dropout:
        model.add(Dropout(level))

    model.add(Conv2D(64, (3, 3), padding='same'))
    if bn:
        model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    if bn:
        model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    if dropout:
        model.add(Dropout(level))

    model.add(Flatten())
    model.add(Dense(512))
    if bn:
        model.add(BatchNormalization())
    model.add(Activation('relu'))
    if dropout:
        model.add(Dropout(level))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))
    if bn:
        opt = keras.optimizers.Adam(lr=0.001, decay=1e-6)
    else:
        opt = keras.optimizers.Adam(lr=0.0001, decay=1e-6)

    model.compile(loss='categorical_crossentropy',
                             optimizer=opt,
                             metrics=['accuracy'])

    # 使用数据增强获取更多的训练数据
    datagen = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)
    datagen.fit(x_train)
    history = model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size), epochs=epochs,
                                  validation_data=(x_test, y_test), workers=4)
    return history


no_dropout_bn_history = model(False, False)
dropout_low_history = model(False, True, 0.2)
dropout_medium_history = model(False, True, 0.5)
dropout_high_history = model(False, True, 0.8)
bn_history = model(True, False)

# 比较多种模型的精确度
plt.plot(no_dropout_bn_history.history['val_accuracy'])
plt.plot(dropout_low_history.history['val_accuracy'])
plt.plot(dropout_medium_history.history['val_accuracy'])
plt.plot(dropout_high_history.history['val_accuracy'])
plt.plot(bn_history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Validation Accuracy')
plt.xlabel('Epoch')
plt.legend(['No bn and dropout', 'Dropout with 0.2', 'Dropout with 0.5', 'Dropout with 0.8', 'BN'], loc='lower right')
plt.grid(True)
plt.show()

# 比较多种模型的损失率
plt.plot(no_dropout_bn_history.history['val_loss'])
plt.plot(dropout_low_history.history['val_loss'])
plt.plot(dropout_medium_history.history['val_loss'])
plt.plot(dropout_high_history.history['val_loss'])
plt.plot(bn_history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['No bn and dropout', 'Dropout with 0.2', 'Dropout with 0.5', 'Dropout with 0.8', 'BN'], loc='upper right')
plt.grid(True)
plt.show()