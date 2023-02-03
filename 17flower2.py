# -*- coding: utf-8 -*-
"""
Created on Tue Feb  8 13:48:12 2022
hnjxzby@2023-2-3重新测试通过
@author: Dell_hnjxzby
"""

#import tensorflow.keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.layers import Dense, Flatten
from matplotlib import pyplot as plt
import numpy as np

from tensorflow import optimizers


# 为保证公平起见，使用相同的随机种子
np.random.seed(7)
batch_size = 32
# 迭代50次
epochs = 50
# 依照模型规定，图片大小被设定为224
IMAGE_SIZE = 224
# 17种花的分类
NUM_CLASSES = 17

#TRAIN_PATH = '/home/yourname/Documents/tensorflow/images/17flowerclasses/train'
#TEST_PATH = '/home/yourname/Documents/tensorflow/images/17flowerclasses/test'

TRAIN_PATH = 'D:/zbytest/17flowerclasses/train'
TEST_PATH = 'D:/zbytest/17flowerclasses/test'


FLOWER_CLASSES = ['Bluebell', 'ButterCup', 'ColtsFoot', 'Cowslip', 'Crocus', 'Daffodil', 'Daisy',
                  'Dandelion', 'Fritillary', 'Iris', 'LilyValley', 'Pansy', 'Snowdrop', 'Sunflower',
                  'Tigerlily', 'tulip', 'WindFlower']


def model(mode='fc'):
    if mode == 'fc':
        # FC层设定为含有512个参数的隐藏层
        base_model = VGG19(input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3), include_top=False, pooling='none')
        x = base_model.output
        x = Flatten()(x)
        x = Dense(512, activation='relu')(x)
        prediction = Dense(NUM_CLASSES, activation='softmax')(x)
    elif mode == 'avg':
        # GAP层通过指定pooling='avg'来设定
        base_model = VGG19(input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3), include_top=False, pooling='avg')
        x = base_model.output
        prediction = Dense(NUM_CLASSES, activation='softmax')(x)
    else:
        # GMP层通过指定pooling='max'来设定
        base_model = VGG19(input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3), include_top=False, pooling='max')
        x = base_model.output
        prediction = Dense(NUM_CLASSES, activation='softmax')(x)
    #model = Model(input=base_model.input, output=prediction)  ##很有意思，要在单词input和output之后加上s才能正确运行 hnjxzby@2023-2-3
    model = Model(inputs=base_model.input, outputs=prediction)
    model.summary()
    opt = optimizers.RMSprop(lr=0.0001, decay=1e-6)
    #opt = keras.optimizers.RMSprop(lr=0.0001,decay=1e-6) ###用keras的优化器不行，要修改成tensorflow的才行！！！今天修改了这两个地方程序运行通过。
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
plt.plot(fc_history.history['val_acc'])
plt.plot(avg_history.history['val_acc'])
plt.plot(max_history.history['val_acc'])
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