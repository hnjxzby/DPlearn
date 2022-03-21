# -*- coding: utf-8 -*-
"""
Created on Tue Feb  8 19:14:05 2022

这个程序很丑，但是可以工作，当然要修改成标准一点程序

输入图片大小进行reshape:
最小分辨率为32*32，这是imagenet的规定。不能设为28*28，会出错。
所以可以测试，32*32，56*56，112*112，224*224，448*448这么些分辨率情况下的准确率对比。
32*32 情况下使用内存：15。9G CPU占用： 5%   GPU占用19%
15.7G   CPU 5%    GPU 33%  GPU温度33度
448*448： 内存用了16.3G  cpu占用：8% GPU占用：51 GPU温度：67度
我想：可能过大的图像输入，效果不一定好？？？




hnjxzby@2022-2-9其实同样的这个程序一点不修改，也可以测试没有均衡处理的数据集：images.csv
只要修改程序pokemon.py中的一行即可(均衡后的数据集是imagesresample.csv)

为了测试程序的正确性，为了节约时间，可以将MYEPOCHS设置为2。
而实际实验时设置为45为好1!!!!

@author: Dell_hnjxzby
"""



import  matplotlib
from    matplotlib import pyplot as plt
"""
matplotlib.rcParams['font.size'] = 18
matplotlib.rcParams['figure.titlesize'] = 18
matplotlib.rcParams['figure.figsize'] = [9, 7]
matplotlib.rcParams['font.family'] = ['KaiTi']
matplotlib.rcParams['axes.unicode_minus']=False
"""
matplotlib.rcParams['font.size'] = 13
matplotlib.rcParams['figure.titlesize'] = 14
matplotlib.rcParams['figure.figsize'] = [7, 7]
matplotlib.rcParams['font.family'] = ['KaiTi']


import  os
import  tensorflow as tf
import  numpy as np
from    tensorflow import keras
from    tensorflow.keras import layers,optimizers,losses
from    tensorflow.keras.callbacks import EarlyStopping

#############################画出混淆图用#########################################
from sklearn.metrics import confusion_matrix


tf.random.set_seed(1234)
np.random.seed(1234)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
assert tf.__version__.startswith('2.')

import time
t0 = time.time()
from pokemon import load_pokemon

MYEPOCHS = 45
batchsz = 32
num_classes = 25


########################################################################################
t1 = time.time()
IMG_SIZE = 32
def preprocess(x,y):
    # x: 图片的路径，y：图片的数字编码,类别
    x = tf.io.read_file(x)
    x = tf.image.decode_jpeg(x, channels=3) # RGBA
    x = tf.image.resize(x, [IMG_SIZE, IMG_SIZE])
    x = tf.image.random_flip_left_right(x)
    x = tf.image.random_flip_up_down(x)
    x = tf.image.random_crop(x, [IMG_SIZE,IMG_SIZE,3])
    x = tf.cast(x, dtype=tf.float32) / 255. 
    
    y = tf.convert_to_tensor(y)
    y = tf.one_hot(y, depth=25)  ###改成25类
    return x, y


# 创建训练集Datset对象
images, labels, table = load_pokemon('pokemon',mode='train')
db_train = tf.data.Dataset.from_tensor_slices((images, labels))
db_train = db_train.shuffle(1000).map(preprocess).batch(batchsz)
# 创建验证集Datset对象
images2, labels2, table = load_pokemon('pokemon',mode='val')
db_val = tf.data.Dataset.from_tensor_slices((images2, labels2))
db_val = db_val.map(preprocess).batch(batchsz)
# 创建测试集Datset对象
images3, labels3, table = load_pokemon('pokemon',mode='test')
db_test = tf.data.Dataset.from_tensor_slices((images3, labels3))
db_test = db_test.map(preprocess).batch(batchsz)
####以上是数据集处理部分对于不同的model都是一样的


IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)

VGG16_MODEL=tf.keras.applications.VGG16(weights='imagenet',input_shape=IMG_SHAPE,include_top=False)                                                                          
VGG16_MODEL.trainable=False
   
global_average_layer = tf.keras.layers.GlobalAveragePooling2D()  #呵呵，这一层将14*14*512已展平了，所以layers.Flatten好象可以不要了，二者有一样的效果啊

model = tf.keras.Sequential()
model.add(VGG16_MODEL)
model.add(global_average_layer)
model.add(tf.keras.layers.Dense(512, activation='relu'))   #我增加了一层，因为VGG16的输出是7*7*512,在此接合一下？
model.add(tf.keras.layers.Dropout(0.5))                    #我增加了一层dropout.
model.add(tf.keras.layers.Dense(25))                       #如果在此去掉了activation,则一定要在model.compile中将from_logits设置为true


model.build(input_shape=(4,IMG_SIZE,IMG_SIZE,3))
model.summary()


model.compile(optimizer=optimizers.Adam(learning_rate=1e-3),
             loss=losses.CategoricalCrossentropy(from_logits=True),
             metrics=["accuracy"])

history_VGG16_32  = model.fit(db_train,  epochs=MYEPOCHS,  validation_data=db_val )

np.save("Model_VGG16_32_val_accuracy.npy",history_VGG16_32.history['val_accuracy'])
np.save("Model_VGG16_32_accuracy.npy",history_VGG16_32.history['accuracy'])
np.save("Model_VGG16_32_val_loss.npy",history_VGG16_32.history['val_loss'])
np.save("Model_VGG16_32_loss.npy",history_VGG16_32.history['loss'])
###上面的数据保存后可以用于绘图32是输入图像的分辨率为32*32。

model.save('Model_VGG16__32_32.h5')

yy_pred = model.predict(db_test)
yy_pred=tf.convert_to_tensor(yy_pred)
yy_pred=(tf.argmax(yy_pred,axis=1)).numpy()
bbbtrue=np.array(labels3)
#print( confusion_matrix(bbbtrue,yy_pred))
#得到的数据存盘后，可以用于以后的绘图用
np.save("Model_VGG16_32_test_pred.npy",yy_pred)
np.save("Model_VGG16_32_test_true.npy",bbbtrue)


#########################################################画出规一化的混淆图
#########################################################画出规一化的混淆图
#########################################################画出规一化的混淆图
y_true = bbbtrue
y_pred = yy_pred

labels = ['Adialer.C', 'Agent.FYI', 'Allaple.A', 'Allaple.L', 'Alueron.gen!J', 'Autorun.K', 'C2LOP.P', 'C2LOP.gen!g', 'Dialplatform.B', 'Dontovo.A', 'Fakerean', 'Instantaccess', 'Lolyda.AA1', 'Lolyda.AA2', 'Lolyda.AA3', 'Lolyda.AT', 'Malex.gen!J', 'Obfuscator.AD', 'Rbot!gen', 'Skintrim.N', 'Swizzor.gen!E', 'Swizzor.gen!I', 'VB.AT', 'Wintrim.BX', 'Yuner.A']
tick_marks = np.array(range(len(labels))) + 0.5
 
def plot_confusion_matrix(cm, title='Confusion Matrix', cmap=plt.cm.binary):
  plt.imshow(cm, interpolation='nearest', cmap=cmap)
  plt.title(title)
  plt.colorbar()
  xlocations = np.array(range(len(labels)))
  plt.xticks(xlocations, labels, rotation=90)
  plt.yticks(xlocations, labels)
  plt.ylabel('True label')
  plt.xlabel('Predicted label')
 
cm = confusion_matrix(y_true, y_pred)
np.set_printoptions(precision=2)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
print( cm_normalized)
plt.figure(figsize=(12, 8), dpi=120)

ind_array = np.arange(len(labels))
x, y = np.meshgrid(ind_array, ind_array)
for x_val, y_val in zip(x.flatten(), y.flatten()):
  c = cm_normalized[y_val][x_val]
  if c > 0.01:
    plt.text(x_val, y_val, "%0.2f" % (c,), color='red', fontsize=7, va='center', ha='center')

# offset the tick
plt.gca().set_xticks(tick_marks, minor=True)
plt.gca().set_yticks(tick_marks, minor=True)
plt.gca().xaxis.set_ticks_position('none')
plt.gca().yaxis.set_ticks_position('none')
plt.grid(True, which='minor', linestyle='-')
plt.gcf().subplots_adjust(bottom=0.15)
 
plot_confusion_matrix(cm_normalized, title='Normalized confusion matrix （ImgSize:32*32）')
# save confusion matrix figure
plt.savefig('confusion_matrix_VGG16_32_32.png', format='png')
plt.show()
#########################################################画出规一化的混淆图
#########################################################画出规一化的混淆图
#########################################################画出规一化的混淆图


validation_steps = 20
loss0,accuracy0 = model.evaluate(db_test, steps = validation_steps)
print("loss: {:.2f}".format(loss0))
print("accuracy: {:.2f}".format(accuracy0))

###统计一下，我们的训练用时多长
t2 = time.time()
t3 =  t2-t1
print("32*32分辨率情况下本次训练用时为：")
print(t3/3600)
############################################################################################







########################################################################################################################
t1 = time.time()
IMG_SIZE = 56
def preprocess(x,y):
    # x: 图片的路径，y：图片的数字编码,类别
    x = tf.io.read_file(x)
    x = tf.image.decode_jpeg(x, channels=3) # RGBA
    x = tf.image.resize(x, [IMG_SIZE, IMG_SIZE])
    x = tf.image.random_flip_left_right(x)
    x = tf.image.random_flip_up_down(x)
    x = tf.image.random_crop(x, [IMG_SIZE,IMG_SIZE,3])
    x = tf.cast(x, dtype=tf.float32) / 255. 
    
    y = tf.convert_to_tensor(y)
    y = tf.one_hot(y, depth=25)  ###改成25类
    return x, y


# 创建训练集Datset对象
images, labels, table = load_pokemon('pokemon',mode='train')
db_train = tf.data.Dataset.from_tensor_slices((images, labels))
db_train = db_train.shuffle(1000).map(preprocess).batch(batchsz)
# 创建验证集Datset对象
images2, labels2, table = load_pokemon('pokemon',mode='val')
db_val = tf.data.Dataset.from_tensor_slices((images2, labels2))
db_val = db_val.map(preprocess).batch(batchsz)
# 创建测试集Datset对象
images3, labels3, table = load_pokemon('pokemon',mode='test')
db_test = tf.data.Dataset.from_tensor_slices((images3, labels3))
db_test = db_test.map(preprocess).batch(batchsz)
####以上是数据集处理部分对于不同的model都是一样的


IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)

VGG16_MODEL=tf.keras.applications.VGG16(weights='imagenet',input_shape=IMG_SHAPE,include_top=False)                                                                          
VGG16_MODEL.trainable=False
global_average_layer = tf.keras.layers.GlobalAveragePooling2D()  #呵呵，这一层将14*14*512已展平了，所以layers.Flatten好象可以不要了，二者有一样的效果啊
model = tf.keras.Sequential()
model.add(VGG16_MODEL)
model.add(global_average_layer)
model.add(tf.keras.layers.Dense(512, activation='relu'))   #我增加了一层，因为VGG16的输出是7*7*512,在此接合一下？
model.add(tf.keras.layers.Dropout(0.5))                    #我增加了一层dropout.
model.add(tf.keras.layers.Dense(25))                       #如果在此去掉了activation,则一定要在model.compile中将from_logits设置为true
model.build(input_shape=(4,IMG_SIZE,IMG_SIZE,3))
model.summary()

model.compile(optimizer=optimizers.Adam(learning_rate=1e-3),
             loss=losses.CategoricalCrossentropy(from_logits=True),
             metrics=["accuracy"])

history_VGG16_56  = model.fit(db_train,  epochs=MYEPOCHS,  validation_data=db_val )

np.save("Model_VGG16_56_val_accuracy.npy",history_VGG16_56.history['val_accuracy'])
np.save("Model_VGG16_56_accuracy.npy",history_VGG16_56.history['accuracy'])
np.save("Model_VGG16_56_val_loss.npy",history_VGG16_56.history['val_loss'])
np.save("Model_VGG16_56_loss.npy",history_VGG16_56.history['loss'])
###上面的数据保存后可以用于绘图,56是输入图像的分辨率为56*56。

model.save('Model_VGG16__56_56.h5')

yy_pred = model.predict(db_test)
yy_pred=tf.convert_to_tensor(yy_pred)
yy_pred=(tf.argmax(yy_pred,axis=1)).numpy()
bbbtrue=np.array(labels3)
#print( confusion_matrix(bbbtrue,yy_pred))
#得到的数据存盘后，可以用于以后的绘图用
np.save("Model_VGG16_56_test_pred.npy",yy_pred)
np.save("Model_VGG16_56_test_true.npy",bbbtrue)

#########################################################画出规一化的混淆图
#########################################################画出规一化的混淆图
#########################################################画出规一化的混淆图
y_true = bbbtrue
y_pred = yy_pred

labels = ['Adialer.C', 'Agent.FYI', 'Allaple.A', 'Allaple.L', 'Alueron.gen!J', 'Autorun.K', 'C2LOP.P', 'C2LOP.gen!g', 'Dialplatform.B', 'Dontovo.A', 'Fakerean', 'Instantaccess', 'Lolyda.AA1', 'Lolyda.AA2', 'Lolyda.AA3', 'Lolyda.AT', 'Malex.gen!J', 'Obfuscator.AD', 'Rbot!gen', 'Skintrim.N', 'Swizzor.gen!E', 'Swizzor.gen!I', 'VB.AT', 'Wintrim.BX', 'Yuner.A']
tick_marks = np.array(range(len(labels))) + 0.5
 
def plot_confusion_matrix(cm, title='Confusion Matrix', cmap=plt.cm.binary):
  plt.imshow(cm, interpolation='nearest', cmap=cmap)
  plt.title(title)
  plt.colorbar()
  xlocations = np.array(range(len(labels)))
  plt.xticks(xlocations, labels, rotation=90)
  plt.yticks(xlocations, labels)
  plt.ylabel('True label')
  plt.xlabel('Predicted label')
 
cm = confusion_matrix(y_true, y_pred)
np.set_printoptions(precision=2)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
print( cm_normalized)
plt.figure(figsize=(12, 8), dpi=120)

ind_array = np.arange(len(labels))
x, y = np.meshgrid(ind_array, ind_array)
for x_val, y_val in zip(x.flatten(), y.flatten()):
  c = cm_normalized[y_val][x_val]
  if c > 0.01:
    plt.text(x_val, y_val, "%0.2f" % (c,), color='red', fontsize=7, va='center', ha='center')

# offset the tick
plt.gca().set_xticks(tick_marks, minor=True)
plt.gca().set_yticks(tick_marks, minor=True)
plt.gca().xaxis.set_ticks_position('none')
plt.gca().yaxis.set_ticks_position('none')
plt.grid(True, which='minor', linestyle='-')
plt.gcf().subplots_adjust(bottom=0.15)
 
plot_confusion_matrix(cm_normalized, title='Normalized confusion matrix （ImgSize:56*56）')
# save confusion matrix figure
plt.savefig('confusion_matrix_VGG16_56_56.png', format='png')
plt.show()
#########################################################画出规一化的混淆图
#########################################################画出规一化的混淆图
#########################################################画出规一化的混淆图

validation_steps = 20
loss0,accuracy0 = model.evaluate(db_test, steps = validation_steps)
print("loss: {:.2f}".format(loss0))
print("accuracy: {:.2f}".format(accuracy0))

###统计一下，我们的训练用时多长
t2 = time.time()
t3 =  t2-t1
print("56*56分辨率情况下本次训练用时为：")
print(t3/3600)
############################################################################################






########################################################################################################################
t1 = time.time()
IMG_SIZE = 112
def preprocess(x,y):
    # x: 图片的路径，y：图片的数字编码,类别
    x = tf.io.read_file(x)
    x = tf.image.decode_jpeg(x, channels=3) # RGBA
    x = tf.image.resize(x, [IMG_SIZE, IMG_SIZE])
    x = tf.image.random_flip_left_right(x)
    x = tf.image.random_flip_up_down(x)
    x = tf.image.random_crop(x, [IMG_SIZE,IMG_SIZE,3])
    x = tf.cast(x, dtype=tf.float32) / 255. 
    
    y = tf.convert_to_tensor(y)
    y = tf.one_hot(y, depth=25)  ###改成25类
    return x, y


# 创建训练集Datset对象
images, labels, table = load_pokemon('pokemon',mode='train')
db_train = tf.data.Dataset.from_tensor_slices((images, labels))
db_train = db_train.shuffle(1000).map(preprocess).batch(batchsz)
# 创建验证集Datset对象
images2, labels2, table = load_pokemon('pokemon',mode='val')
db_val = tf.data.Dataset.from_tensor_slices((images2, labels2))
db_val = db_val.map(preprocess).batch(batchsz)
# 创建测试集Datset对象
images3, labels3, table = load_pokemon('pokemon',mode='test')
db_test = tf.data.Dataset.from_tensor_slices((images3, labels3))
db_test = db_test.map(preprocess).batch(batchsz)
####以上是数据集处理部分对于不同的model都是一样的


IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)

VGG16_MODEL=tf.keras.applications.VGG16(weights='imagenet',input_shape=IMG_SHAPE,include_top=False)                                                                          
VGG16_MODEL.trainable=False
global_average_layer = tf.keras.layers.GlobalAveragePooling2D()  #呵呵，这一层将14*14*512已展平了，所以layers.Flatten好象可以不要了，二者有一样的效果啊
model = tf.keras.Sequential()
model.add(VGG16_MODEL)
model.add(global_average_layer)
model.add(tf.keras.layers.Dense(512, activation='relu'))   #我增加了一层，因为VGG16的输出是7*7*512,在此接合一下？
model.add(tf.keras.layers.Dropout(0.5))                    #我增加了一层dropout.
model.add(tf.keras.layers.Dense(25))                       #如果在此去掉了activation,则一定要在model.compile中将from_logits设置为true
model.build(input_shape=(4,IMG_SIZE,IMG_SIZE,3))
model.summary()

model.compile(optimizer=optimizers.Adam(learning_rate=1e-3),
             loss=losses.CategoricalCrossentropy(from_logits=True),
             metrics=["accuracy"])

history_VGG16_112  = model.fit(db_train,  epochs=MYEPOCHS,  validation_data=db_val )

np.save("Model_VGG16_112_val_accuracy.npy",history_VGG16_112.history['val_accuracy'])
np.save("Model_VGG16_112_accuracy.npy",history_VGG16_112.history['accuracy'])
np.save("Model_VGG16_112_val_loss.npy",history_VGG16_112.history['val_loss'])
np.save("Model_VGG16_112_loss.npy",history_VGG16_112.history['loss'])
###上面的数据保存后可以用于绘图,112是输入图像的分辨率为112*112。

model.save('Model_VGG16__112_112.h5')

yy_pred = model.predict(db_test)
yy_pred=tf.convert_to_tensor(yy_pred)
yy_pred=(tf.argmax(yy_pred,axis=1)).numpy()
bbbtrue=np.array(labels3)
#print( confusion_matrix(bbbtrue,yy_pred))
#得到的数据存盘后，可以用于以后的绘图用
np.save("Model_VGG16_112_test_pred.npy",yy_pred)
np.save("Model_VGG16_112_test_true.npy",bbbtrue)

#########################################################画出规一化的混淆图
#########################################################画出规一化的混淆图
#########################################################画出规一化的混淆图
y_true = bbbtrue
y_pred = yy_pred

labels = ['Adialer.C', 'Agent.FYI', 'Allaple.A', 'Allaple.L', 'Alueron.gen!J', 'Autorun.K', 'C2LOP.P', 'C2LOP.gen!g', 'Dialplatform.B', 'Dontovo.A', 'Fakerean', 'Instantaccess', 'Lolyda.AA1', 'Lolyda.AA2', 'Lolyda.AA3', 'Lolyda.AT', 'Malex.gen!J', 'Obfuscator.AD', 'Rbot!gen', 'Skintrim.N', 'Swizzor.gen!E', 'Swizzor.gen!I', 'VB.AT', 'Wintrim.BX', 'Yuner.A']
tick_marks = np.array(range(len(labels))) + 0.5
 
def plot_confusion_matrix(cm, title='Confusion Matrix', cmap=plt.cm.binary):
  plt.imshow(cm, interpolation='nearest', cmap=cmap)
  plt.title(title)
  plt.colorbar()
  xlocations = np.array(range(len(labels)))
  plt.xticks(xlocations, labels, rotation=90)
  plt.yticks(xlocations, labels)
  plt.ylabel('True label')
  plt.xlabel('Predicted label')
 
cm = confusion_matrix(y_true, y_pred)
np.set_printoptions(precision=2)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
print( cm_normalized)
plt.figure(figsize=(12, 8), dpi=120)

ind_array = np.arange(len(labels))
x, y = np.meshgrid(ind_array, ind_array)
for x_val, y_val in zip(x.flatten(), y.flatten()):
  c = cm_normalized[y_val][x_val]
  if c > 0.01:
    plt.text(x_val, y_val, "%0.2f" % (c,), color='red', fontsize=7, va='center', ha='center')

# offset the tick
plt.gca().set_xticks(tick_marks, minor=True)
plt.gca().set_yticks(tick_marks, minor=True)
plt.gca().xaxis.set_ticks_position('none')
plt.gca().yaxis.set_ticks_position('none')
plt.grid(True, which='minor', linestyle='-')
plt.gcf().subplots_adjust(bottom=0.15)
 
plot_confusion_matrix(cm_normalized, title='Normalized confusion matrix （ImgSize:112*112）')
# save confusion matrix figure
plt.savefig('confusion_matrix_VGG16_112_112.png', format='png')
plt.show()
#########################################################画出规一化的混淆图
#########################################################画出规一化的混淆图
#########################################################画出规一化的混淆图


validation_steps = 20
loss0,accuracy0 = model.evaluate(db_test, steps = validation_steps)
print("loss: {:.2f}".format(loss0))
print("accuracy: {:.2f}".format(accuracy0))

###统计一下，我们的训练用时多长
t2 = time.time()
t3 =  t2-t1
print("112*112分辨率情况下本次训练用时为：")  ###呵呵，这是相对程序第一次启动时的相对时间，不是本次的哦，要将前面的阶段如56*56模型用时送去。。。
print(t3/3600)
############################################################################################




########################################################################################################################
t1 = time.time()
IMG_SIZE = 224
def preprocess(x,y):
    # x: 图片的路径，y：图片的数字编码,类别
    x = tf.io.read_file(x)
    x = tf.image.decode_jpeg(x, channels=3) # RGBA
    x = tf.image.resize(x, [IMG_SIZE, IMG_SIZE])
    x = tf.image.random_flip_left_right(x)
    x = tf.image.random_flip_up_down(x)
    x = tf.image.random_crop(x, [IMG_SIZE,IMG_SIZE,3])
    x = tf.cast(x, dtype=tf.float32) / 255. 
    y = tf.convert_to_tensor(y)
    y = tf.one_hot(y, depth=25)  ###改成25类
    return x, y


# 创建训练集Datset对象
images, labels, table = load_pokemon('pokemon',mode='train')
db_train = tf.data.Dataset.from_tensor_slices((images, labels))
db_train = db_train.shuffle(1000).map(preprocess).batch(batchsz)
# 创建验证集Datset对象
images2, labels2, table = load_pokemon('pokemon',mode='val')
db_val = tf.data.Dataset.from_tensor_slices((images2, labels2))
db_val = db_val.map(preprocess).batch(batchsz)
# 创建测试集Datset对象
images3, labels3, table = load_pokemon('pokemon',mode='test')
db_test = tf.data.Dataset.from_tensor_slices((images3, labels3))
db_test = db_test.map(preprocess).batch(batchsz)
####以上是数据集处理部分对于不同的model都是一样的


IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)

VGG16_MODEL=tf.keras.applications.VGG16(weights='imagenet',input_shape=IMG_SHAPE,include_top=False)                                                                          
VGG16_MODEL.trainable=False
global_average_layer = tf.keras.layers.GlobalAveragePooling2D()  #呵呵，这一层将14*14*512已展平了，所以layers.Flatten好象可以不要了，二者有一样的效果啊
model = tf.keras.Sequential()
model.add(VGG16_MODEL)
model.add(global_average_layer)
model.add(tf.keras.layers.Dense(512, activation='relu'))   #我增加了一层，因为VGG16的输出是7*7*512,在此接合一下？
model.add(tf.keras.layers.Dropout(0.5))                    #我增加了一层dropout.
model.add(tf.keras.layers.Dense(25))                       #如果在此去掉了activation,则一定要在model.compile中将from_logits设置为true
model.build(input_shape=(4,IMG_SIZE,IMG_SIZE,3))
model.summary()

model.compile(optimizer=optimizers.Adam(learning_rate=1e-3),
             loss=losses.CategoricalCrossentropy(from_logits=True),
             metrics=["accuracy"])

history_VGG16_224  = model.fit(db_train,  epochs=MYEPOCHS,  validation_data=db_val )

np.save("Model_VGG16_224_val_accuracy.npy",history_VGG16_224.history['val_accuracy'])
np.save("Model_VGG16_224_accuracy.npy",history_VGG16_224.history['accuracy'])
np.save("Model_VGG16_224_val_loss.npy",history_VGG16_224.history['val_loss'])
np.save("Model_VGG16_224_loss.npy",history_VGG16_224.history['loss'])
###上面的数据保存后可以用于绘图,224是输入图像的分辨率为224*224。

model.save('Model_VGG16__224_224.h5')

yy_pred = model.predict(db_test)
yy_pred=tf.convert_to_tensor(yy_pred)
yy_pred=(tf.argmax(yy_pred,axis=1)).numpy()
bbbtrue=np.array(labels3)
#print( confusion_matrix(bbbtrue,yy_pred))
#得到的数据存盘后，可以用于以后的绘图用
np.save("Model_VGG16_224_test_pred.npy",yy_pred)
np.save("Model_VGG16_224_test_true.npy",bbbtrue)

#########################################################画出规一化的混淆图
#########################################################画出规一化的混淆图
#########################################################画出规一化的混淆图
y_true = bbbtrue
y_pred = yy_pred

labels = ['Adialer.C', 'Agent.FYI', 'Allaple.A', 'Allaple.L', 'Alueron.gen!J', 'Autorun.K', 'C2LOP.P', 'C2LOP.gen!g', 'Dialplatform.B', 'Dontovo.A', 'Fakerean', 'Instantaccess', 'Lolyda.AA1', 'Lolyda.AA2', 'Lolyda.AA3', 'Lolyda.AT', 'Malex.gen!J', 'Obfuscator.AD', 'Rbot!gen', 'Skintrim.N', 'Swizzor.gen!E', 'Swizzor.gen!I', 'VB.AT', 'Wintrim.BX', 'Yuner.A']
tick_marks = np.array(range(len(labels))) + 0.5
 
def plot_confusion_matrix(cm, title='Confusion Matrix', cmap=plt.cm.binary):
  plt.imshow(cm, interpolation='nearest', cmap=cmap)
  plt.title(title)
  plt.colorbar()
  xlocations = np.array(range(len(labels)))
  plt.xticks(xlocations, labels, rotation=90)
  plt.yticks(xlocations, labels)
  plt.ylabel('True label')
  plt.xlabel('Predicted label')
 
cm = confusion_matrix(y_true, y_pred)
np.set_printoptions(precision=2)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
print( cm_normalized)
plt.figure(figsize=(12, 8), dpi=120)

ind_array = np.arange(len(labels))
x, y = np.meshgrid(ind_array, ind_array)
for x_val, y_val in zip(x.flatten(), y.flatten()):
  c = cm_normalized[y_val][x_val]
  if c > 0.01:
    plt.text(x_val, y_val, "%0.2f" % (c,), color='red', fontsize=7, va='center', ha='center')

# offset the tick
plt.gca().set_xticks(tick_marks, minor=True)
plt.gca().set_yticks(tick_marks, minor=True)
plt.gca().xaxis.set_ticks_position('none')
plt.gca().yaxis.set_ticks_position('none')
plt.grid(True, which='minor', linestyle='-')
plt.gcf().subplots_adjust(bottom=0.15)
 
plot_confusion_matrix(cm_normalized, title='Normalized confusion matrix （ImgSize:224*224）')
# save confusion matrix figure
plt.savefig('confusion_matrix_VGG16_224_224.png', format='png')
plt.show()
#########################################################画出规一化的混淆图
#########################################################画出规一化的混淆图
#########################################################画出规一化的混淆图


validation_steps = 20
loss0,accuracy0 = model.evaluate(db_test, steps = validation_steps)
print("loss: {:.2f}".format(loss0))
print("accuracy: {:.2f}".format(accuracy0))

###统计一下，我们的训练用时多长
t2 = time.time()
t3 =  t2-t1
print("224*224分辨率情况下本次训练用时为：")
print(t3/3600)
############################################################################################






########################################################################################################################
t1 = time.time()
IMG_SIZE = 448
def preprocess(x,y):
    # x: 图片的路径，y：图片的数字编码,类别
    x = tf.io.read_file(x)
    x = tf.image.decode_jpeg(x, channels=3) # RGBA
    x = tf.image.resize(x, [IMG_SIZE, IMG_SIZE])
    x = tf.image.random_flip_left_right(x)
    x = tf.image.random_flip_up_down(x)
    x = tf.image.random_crop(x, [IMG_SIZE,IMG_SIZE,3])
    x = tf.cast(x, dtype=tf.float32) / 255. 
    
    y = tf.convert_to_tensor(y)
    y = tf.one_hot(y, depth=25)  ###改成25类
    return x, y


# 创建训练集Datset对象
images, labels, table = load_pokemon('pokemon',mode='train')
db_train = tf.data.Dataset.from_tensor_slices((images, labels))
db_train = db_train.shuffle(1000).map(preprocess).batch(batchsz)
# 创建验证集Datset对象
images2, labels2, table = load_pokemon('pokemon',mode='val')
db_val = tf.data.Dataset.from_tensor_slices((images2, labels2))
db_val = db_val.map(preprocess).batch(batchsz)
# 创建测试集Datset对象
images3, labels3, table = load_pokemon('pokemon',mode='test')
db_test = tf.data.Dataset.from_tensor_slices((images3, labels3))
db_test = db_test.map(preprocess).batch(batchsz)
####以上是数据集处理部分对于不同的model都是一样的


IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)

VGG16_MODEL=tf.keras.applications.VGG16(weights='imagenet',input_shape=IMG_SHAPE,include_top=False)                                                                          
VGG16_MODEL.trainable=False
global_average_layer = tf.keras.layers.GlobalAveragePooling2D()  #呵呵，这一层将14*14*512已展平了，所以layers.Flatten好象可以不要了，二者有一样的效果啊
model = tf.keras.Sequential()
model.add(VGG16_MODEL)
model.add(global_average_layer)
model.add(tf.keras.layers.Dense(512, activation='relu'))   #我增加了一层，因为VGG16的输出是7*7*512,在此接合一下？
model.add(tf.keras.layers.Dropout(0.5))                    #我增加了一层dropout.
model.add(tf.keras.layers.Dense(25))                       #如果在此去掉了activation,则一定要在model.compile中将from_logits设置为true
model.build(input_shape=(4,IMG_SIZE,IMG_SIZE,3))
model.summary()

model.compile(optimizer=optimizers.Adam(learning_rate=1e-3),
             loss=losses.CategoricalCrossentropy(from_logits=True),
             metrics=["accuracy"])

history_VGG16_448  = model.fit(db_train,  epochs=MYEPOCHS,  validation_data=db_val )

np.save("Model_VGG16_448_val_accuracy.npy",history_VGG16_448.history['val_accuracy'])
np.save("Model_VGG16_448_accuracy.npy",history_VGG16_448.history['accuracy'])
np.save("Model_VGG16_448_val_loss.npy",history_VGG16_448.history['val_loss'])
np.save("Model_VGG16_448_loss.npy",history_VGG16_448.history['loss'])
###上面的数据保存后可以用于绘图,448是输入图像的分辨率为448*448。

model.save('Model_VGG16__448_448.h5')

yy_pred = model.predict(db_test)
yy_pred=tf.convert_to_tensor(yy_pred)
yy_pred=(tf.argmax(yy_pred,axis=1)).numpy()
bbbtrue=np.array(labels3)
#print( confusion_matrix(bbbtrue,yy_pred))
#得到的数据存盘后，可以用于以后的绘图用
np.save("Model_VGG16_448_test_pred.npy",yy_pred)
np.save("Model_VGG16_448_test_true.npy",bbbtrue)


#########################################################画出规一化的混淆图
#########################################################画出规一化的混淆图
#########################################################画出规一化的混淆图
y_true = bbbtrue
y_pred = yy_pred

labels = ['Adialer.C', 'Agent.FYI', 'Allaple.A', 'Allaple.L', 'Alueron.gen!J', 'Autorun.K', 'C2LOP.P', 'C2LOP.gen!g', 'Dialplatform.B', 'Dontovo.A', 'Fakerean', 'Instantaccess', 'Lolyda.AA1', 'Lolyda.AA2', 'Lolyda.AA3', 'Lolyda.AT', 'Malex.gen!J', 'Obfuscator.AD', 'Rbot!gen', 'Skintrim.N', 'Swizzor.gen!E', 'Swizzor.gen!I', 'VB.AT', 'Wintrim.BX', 'Yuner.A']
tick_marks = np.array(range(len(labels))) + 0.5
 
def plot_confusion_matrix(cm, title='Confusion Matrix', cmap=plt.cm.binary):
  plt.imshow(cm, interpolation='nearest', cmap=cmap)
  plt.title(title)
  plt.colorbar()
  xlocations = np.array(range(len(labels)))
  plt.xticks(xlocations, labels, rotation=90)
  plt.yticks(xlocations, labels)
  plt.ylabel('True label')
  plt.xlabel('Predicted label')
 
cm = confusion_matrix(y_true, y_pred)
np.set_printoptions(precision=2)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
print( cm_normalized)
plt.figure(figsize=(12, 8), dpi=120)

ind_array = np.arange(len(labels))
x, y = np.meshgrid(ind_array, ind_array)
for x_val, y_val in zip(x.flatten(), y.flatten()):
  c = cm_normalized[y_val][x_val]
  if c > 0.01:
    plt.text(x_val, y_val, "%0.2f" % (c,), color='red', fontsize=7, va='center', ha='center')

# offset the tick
plt.gca().set_xticks(tick_marks, minor=True)
plt.gca().set_yticks(tick_marks, minor=True)
plt.gca().xaxis.set_ticks_position('none')
plt.gca().yaxis.set_ticks_position('none')
plt.grid(True, which='minor', linestyle='-')
plt.gcf().subplots_adjust(bottom=0.15)
 
plot_confusion_matrix(cm_normalized, title='Normalized confusion matrix （ImgSize:448*448）')
# save confusion matrix figure
plt.savefig('confusion_matrix_VGG16_448_448.png', format='png')
plt.show()
#########################################################画出规一化的混淆图
#########################################################画出规一化的混淆图
#########################################################画出规一化的混淆图

validation_steps = 20
loss0,accuracy0 = model.evaluate(db_test, steps = validation_steps)
print("loss: {:.2f}".format(loss0))
print("accuracy: {:.2f}".format(accuracy0))

###统计一下，我们的训练用时多长
t2 = time.time()
t3 =  t2-t1
print("448*448分辨率情况下本次训练用时为：")
print(t3/3600)
############################################################################################




###统计一下，我们的全部训练用时多长
t4 = time.time()
t5=t4-t0
print("5个阶段总的实验时间为：")
print(t5/3600)

###########################################################################################


# 比较多种模型的精确度
plt.plot(history_VGG16_32.history['val_accuracy'])
plt.plot(history_VGG16_56.history['val_accuracy'])
plt.plot(history_VGG16_112.history['val_accuracy'])
plt.plot(history_VGG16_224.history['val_accuracy'])
plt.plot(history_VGG16_448.history['val_accuracy'])
plt.title('Model accuracy - VGG16')
plt.ylabel('Validation Accuracy')
plt.xlabel('Epoch')
plt.legend(['IMGSIZE 32*32', 'IMGSIZE 56*56', 'IMGSIZE 112*112', 'IMGSIZE 224*224', 'IMGSIZE 448*448'], loc='lower right')
plt.grid(True)
plt.savefig('val_acc_VGG16_all.png', format='png')
plt.show()

# 比较多种模型的损失率
plt.plot(history_VGG16_32.history['val_loss'])
plt.plot(history_VGG16_56.history['val_loss'])
plt.plot(history_VGG16_112.history['val_loss'])
plt.plot(history_VGG16_224.history['val_loss'])
plt.plot(history_VGG16_448.history['val_loss'])
plt.title('Model loss - VGG16')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['IMGSIZE 32*32', 'IMGSIZE 56*56', 'IMGSIZE 112*112', 'IMGSIZE 224*224', 'IMGSIZE 448*448'], loc='upper right')
plt.grid(True)
plt.savefig('val_loss_VGG16_all.png', format='png')
plt.show()
###########################################################################################




"""

###从持久化的角度来看，我想将上面的history中的val_accuracy和 val_loss保存起来以后可用
####因为训练要很长时间，所以训练得到的数据存盘后，可以用于以后的绘图用
####测试成功的hnjxzby@2022-2-9

np.save("Model_VGG16_32_val_accuracy.npy",history_VGG16_32.history['val_accuracy'])
np.save("Model_VGG16_32_accuracy.npy",history_VGG16_32.history['accuracy'])
np.save("Model_VGG16_32_val_loss.npy",history_VGG16_32.history['val_loss'])
np.save("Model_VGG16_32_loss.npy",history_VGG16_32.history['loss'])
###上面的数据保存后可以用于绘图32是输入图像的分辨率为32*32。

np.save("Model_VGG16_56_val_accuracy.npy",history_VGG16_56.history['val_accuracy'])
np.save("Model_VGG16_56_accuracy.npy",history_VGG16_56.history['accuracy'])
np.save("Model_VGG16_56_val_loss.npy",history_VGG16_56.history['val_loss'])
np.save("Model_VGG16_56_loss.npy",history_VGG16_56.history['loss'])
###上面的数据保存后可以用于绘图,56是输入图像的分辨率为56*56。

np.save("Model_VGG16_112_val_accuracy.npy",history_VGG16_112.history['val_accuracy'])
np.save("Model_VGG16_112_accuracy.npy",history_VGG16_112.history['accuracy'])
np.save("Model_VGG16_112_val_loss.npy",history_VGG16_112.history['val_loss'])
np.save("Model_VGG16_112_loss.npy",history_VGG16_112.history['loss'])
###上面的数据保存后可以用于绘图,112是输入图像的分辨率为112*112。


np.save("Model_VGG16_224_val_accuracy.npy",history_VGG16_224.history['val_accuracy'])
np.save("Model_VGG16_224_accuracy.npy",history_VGG16_224.history['accuracy'])
np.save("Model_VGG16_224_val_loss.npy",history_VGG16_224.history['val_loss'])
np.save("Model_VGG16_224_loss.npy",history_VGG16_224.history['loss'])
###上面的数据保存后可以用于绘图,224是输入图像的分辨率为224*224。

np.save("Model_VGG16_448_val_accuracy.npy",history_VGG16_448.history['val_accuracy'])
np.save("Model_VGG16_448_accuracy.npy",history_VGG16_448.history['accuracy'])
np.save("Model_VGG16_448_val_loss.npy",history_VGG16_448.history['val_loss'])
np.save("Model_VGG16_448_loss.npy",history_VGG16_448.history['loss'])
###上面的数据保存后可以用于绘图,448是输入图像的分辨率为448*448。



###########################################################################################
####以下我想画出5种输入图像大小的情况下的五种混淆矩阵（或者图或标准化图形）
###########################################################################################
#############################画出混淆图#########################################

# -*-coding:utf-8-*-
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np


y_true = bbbtrue
y_pred = yy_pred

labels = ['Adialer.C', 'Agent.FYI', 'Allaple.A', 'Allaple.L', 'Alueron.gen!J', 'Autorun.K', 'C2LOP.P', 'C2LOP.gen!g', 'Dialplatform.B', 'Dontovo.A', 'Fakerean', 'Instantaccess', 'Lolyda.AA1', 'Lolyda.AA2', 'Lolyda.AA3', 'Lolyda.AT', 'Malex.gen!J', 'Obfuscator.AD', 'Rbot!gen', 'Skintrim.N', 'Swizzor.gen!E', 'Swizzor.gen!I', 'VB.AT', 'Wintrim.BX', 'Yuner.A']
tick_marks = np.array(range(len(labels))) + 0.5
 
def plot_confusion_matrix(cm, title='Confusion Matrix', cmap=plt.cm.binary):
  plt.imshow(cm, interpolation='nearest', cmap=cmap)
  plt.title(title)
  plt.colorbar()
  xlocations = np.array(range(len(labels)))
  plt.xticks(xlocations, labels, rotation=90)
  plt.yticks(xlocations, labels)
  plt.ylabel('True label')
  plt.xlabel('Predicted label')
 
cm = confusion_matrix(y_true, y_pred)
np.set_printoptions(precision=2)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
print( cm_normalized)
plt.figure(figsize=(12, 8), dpi=120)

ind_array = np.arange(len(labels))
x, y = np.meshgrid(ind_array, ind_array)
for x_val, y_val in zip(x.flatten(), y.flatten()):
  c = cm_normalized[y_val][x_val]
  if c > 0.01:
    plt.text(x_val, y_val, "%0.2f" % (c,), color='red', fontsize=7, va='center', ha='center')

# offset the tick
plt.gca().set_xticks(tick_marks, minor=True)
plt.gca().set_yticks(tick_marks, minor=True)
plt.gca().xaxis.set_ticks_position('none')
plt.gca().yaxis.set_ticks_position('none')
plt.grid(True, which='minor', linestyle='-')
plt.gcf().subplots_adjust(bottom=0.15)
 
plot_confusion_matrix(cm_normalized, title='Normalized confusion matrix')
# save confusion matrix figure
plt.savefig('confusion_matrix_VGG16_32_32.png', format='png')
plt.show()
"""