# -*- coding: utf-8 -*-
"""
Created on Tue Feb  8 17:16:04 2022
https://www.cnblogs.com/hutao722/p/9618426.html
本实验用于测试各层参数计算
是学习第一步
深度学习基础系列（一）| 一文看懂用kersa构建模型的各层含义（掌握输出尺寸和可训练参数数量的计算方法）



最后关注的是"param"可训练参数数量，不同的模型层计算方法不一样：

　　对于卷积层而言，假设过滤器尺寸为f * f， 过滤器数量为n， 若开启了bias，则bias数固定为1，输入图片的通道数为c，则param计算公式= (f * f * c + 1) * n；
　　对于池化层、flatten、dropout操作而言，是不需要训练参数的，所以param为0；
　　对于全连接层而言，假设输入的列向量大小为i，输出的列向量大小为o，若开启bias，则param计算公式为=i * o + o
@author: Dell_hnjxzby
"""

from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPool2D


(train_data, train_labels), (test_data, test_labels) = keras.datasets.mnist.load_data()
train_data = train_data.reshape(-1, 28, 28, 1)
print("train data type:{}, shape:{}, dim:{}".format(type(train_data), train_data.shape, train_data.ndim))
# 第一组
model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu', input_shape=(28, 28, 1)))
model.add(Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
# 第二组
model.add(Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu'))
model.add(Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
# 第三组
model.add(Flatten())
model.add(Dense(units=256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(units=10, activation='softmax'))

model.summary()