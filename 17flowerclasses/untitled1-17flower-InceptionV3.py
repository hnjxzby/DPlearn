# -*- coding: utf-8 -*-
"""
Created on Tue Feb  8 14:18:23 2022

https://www.cnblogs.com/hutao722/p/10008581.html
　Global Average Pooling(简称GAP，全局池化层)技术最早提出是在这篇论文（第3.2节）中，被认为是可以替代全连接层的一种新技术。在keras发布的经典模型中，可以看到不少模型甚至抛弃了全连接层，转而使用GAP，而在支持迁移学习方面，各个模型几乎都支持使用Global Average Pooling和Global Max Pooling(GMP)。 然而，GAP是否真的可以取代全连接层？其背后的原理何在呢？本文来一探究竟。


测试的都是迁移学习，这个实验的model是 inception_v3
第一次运行时，当然要联网，从网上下载80M左右的模型参数到本地。
因这个程序 是2018年的了，比较老，所以有的小地方要更新到新版本的支持。如
model = Model(inputs = base_model.input, outputs = prediction)中的input要写成inputs

又如新版本中不支持优化器：：rmsprop,我就将其改成了Adam()
实验成功：
很明显，在InceptionV3模型下，FC、GAP和GMP都表现很好，但可以看出GAP的表现依旧最好，其准确度普遍在90%以上，而另两种的准确度在80～90%之间。
结论
　　从本实验看出，在数据集有限的情况下，采用经典模型进行迁移学习时，GMP表现不太稳定，FC层由于训练参数过多，更易导致过拟合现象的发生，而GAP则表现稳定，优于FC层。当然具体情况具体分析，我们拿到数据集后，可以在几种方式中多训练测试，以寻求最优解决方案。

 

@author: Dell_hnjxzby
"""
import tensorflow as tf
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.layers import Dense, Flatten
from matplotlib import pyplot as plt
import numpy as np

# 为保证公平起见，使用相同的随机种子
np.random.seed(7)
batch_size = 32
# 迭代50次
epochs = 50
# 依照模型规定，图片大小被设定为224
IMAGE_SIZE = 224
# 17种花的分类
NUM_CLASSES = 17

# TRAIN_PATH = '/home/hutao/Documents/tensorflow/images/17flowerclasses/train'
# TEST_PATH = '/home/hutao/Documents/tensorflow/images/17flowerclasses/test'

TRAIN_PATH = 'D:/zbytest/17flowerclasses/train'
TEST_PATH = 'D:/zbytest/17flowerclasses/test'
###这个斜杠的方向不能错啊！！！！hnjxzby@2022-2-8


FLOWER_CLASSES = ['Bluebell', 'ButterCup', 'ColtsFoot', 'Cowslip', 'Crocus', 'Daffodil', 'Daisy',
                  'Dandelion', 'Fritillary', 'Iris', 'LilyValley', 'Pansy', 'Snowdrop', 'Sunflower',
                  'Tigerlily', 'tulip', 'WindFlower']


def model(mode='fc'):
    if mode == 'fc':
        # FC层设定为含有512个参数的隐藏层
        base_model = InceptionV3(input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3), include_top=False, pooling='none')
        x = base_model.output
        x = Flatten()(x)
        x = Dense(512, activation='relu')(x)
        prediction = Dense(NUM_CLASSES, activation='softmax')(x)
    elif mode == 'avg':
        # GAP层通过指定pooling='avg'来设定
        base_model = InceptionV3(input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3), include_top=False, pooling='avg')
        x = base_model.output
        prediction = Dense(NUM_CLASSES, activation='softmax')(x)
    else:
        # GMP层通过指定pooling='max'来设定
        base_model = InceptionV3(input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3), include_top=False, pooling='max')
        x = base_model.output
        prediction = Dense(NUM_CLASSES, activation='softmax')(x)

    model = Model(inputs = base_model.input, outputs = prediction)
    model.summary()
    #opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)
    opt = tf.keras.optimizers.Adam(lr=0.0001, decay=1e-6)
    model.compile(loss='categorical_crossentropy',
                             optimizer=opt,
                             metrics=['accuracy'])

    # 使用数据增强
    train_datagen = ImageDataGenerator()
    train_generator = train_datagen.flow_from_directory(directory=TRAIN_PATH,
                                                        target_size=(IMAGE_SIZE, IMAGE_SIZE),
                                                        classes=FLOWER_CLASSES)
    test_datagen = ImageDataGenerator()
    test_generator = test_datagen.flow_from_directory(directory=TEST_PATH,
                                                      target_size=(IMAGE_SIZE, IMAGE_SIZE),
                                                      classes=FLOWER_CLASSES)
    # 运行模型
    history = model.fit_generator(train_generator, epochs=epochs, validation_data=test_generator)
    return history


fc_history = model('fc')
avg_history = model('avg')
max_history = model('max')


# 比较多种模型的精确度
plt.plot(fc_history.history['val_accuracy'])
plt.plot(avg_history.history['val_accuracy'])
plt.plot(max_history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Validation Accuracy')
plt.xlabel('Epoch')
plt.legend(['FC', 'AVG', 'MAX'], loc='lower right')
plt.grid(True)
plt.show()

# 比较多种模型的损失率
plt.plot(fc_history.history['val_loss'])
plt.plot(avg_history.history['val_loss'])
plt.plot(max_history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['FC', 'AVG', 'MAX'], loc='upper right')
plt.grid(True)
plt.show()


