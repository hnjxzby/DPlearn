# -*- coding: utf-8 -*-

import  matplotlib
from    matplotlib import pyplot as plt
matplotlib.rcParams['font.size'] = 18
matplotlib.rcParams['figure.titlesize'] = 18
matplotlib.rcParams['figure.figsize'] = [9, 7]
matplotlib.rcParams['font.family'] = ['KaiTi']
matplotlib.rcParams['axes.unicode_minus']=False

import  os
import  tensorflow as tf
import  numpy as np
from    tensorflow import keras
from    tensorflow.keras import layers,optimizers,losses
from    tensorflow.keras.callbacks import EarlyStopping

tf.random.set_seed(1234)
np.random.seed(1234)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
assert tf.__version__.startswith('2.')

import time
t1 = time.time()
from pokemon import load_pokemon,normalize

def preprocess(x,y):
    # x: 图片的路径，y：图片的数字编码,类别
    x = tf.io.read_file(x)
    x = tf.image.decode_jpeg(x, channels=3) # RGBA
    x = tf.image.per_image_standardization(x) ### 这是增加的规范化的一个办法。hnjxzby@2021-10-08  
    x = tf.image.resize(x, [448, 4448])
    x = tf.image.random_flip_left_right(x)
    x = tf.image.random_flip_up_down(x)
    x = tf.image.random_crop(x, [448,448,3])
    # x: [0,255]=> -1~1
    x = tf.cast(x, dtype=tf.float32) / 255. 
    x = tf.image.resize(x, [448,448])
    #x = normalize(x)
    y = tf.convert_to_tensor(y)
    y = tf.one_hot(y, depth=25)  ###改成25类
    return x, y

#batchsz = 256
batchsz = 32

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
num_classes = 25

IMG_SIZE=448
BATCH_SIZE = 32

IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)
###VGG16_MODEL=tf.keras.applications.VGG16(weights='imagenet',input_shape=IMG_SHAPE,include_top=False)                                          
VGG16_MODEL=tf.keras.applications.VGG16(input_shape=IMG_SHAPE,include_top=False)                                                                          

VGG16_MODEL.trainable=False

fine_tune_at = -3
for layer in VGG16_MODEL.layers[:fine_tune_at]:
    layer.trainable = False
    
global_average_layer = tf.keras.layers.GlobalAveragePooling2D()

model = tf.keras.Sequential()

model.add(VGG16_MODEL)
model.add(global_average_layer)
model.add(tf.keras.layers.Dense(512, activation='relu'))   #我增加了一层
model.add(tf.keras.layers.Dropout(0.5))                    #我增加了一层
model.add(tf.keras.layers.Dense(25, activation="softmax"))

model.build(input_shape=(32,448,448,3))
model.summary()


model.compile(optimizer=optimizers.Adam(lr=1e-3),
             loss=losses.CategoricalCrossentropy(from_logits=True),
             metrics=["accuracy"])

img_, label_ = next(iter(db_train))
pred_ = model.predict(img_)
print(pred_.shape)

img_1, label_1 = next(iter(db_train))
pred_1 = model.predict(img_1)
print(pred_1.shape)


import datetime
####用tensorboard可视化
log_dir=r"d:\\zbylog\\" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

history = model.fit(db_train,epochs=100, validation_data=db_val,callbacks=[tensorboard_callback] )

validation_steps = 20
loss0,accuracy0 = model.evaluate(db_test, steps = validation_steps)
print("loss: {:.2f}".format(loss0))
print("accuracy: {:.2f}".format(accuracy0))

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

history = history.history
print(history.keys())
print(history['val_accuracy'])
print(history['accuracy'])

####因为训练要很长时间，所以训练得到的数据存盘后，可以用于以后的绘图用
np.save("zby_VGG_100_448_loss.npy",history['loss'])
np.save("zby_VGG_100_448_accuracy.npy",history['accuracy'])
np.save("zby_VGG_100_448_val_loss.npy",history['val_loss'])
np.save("zby_VGG_100_448_val_accuracy.npy",history['val_accuracy'])
###上面的数据保存后可以用于绘图

############################################################
#hnjxzby@2021-10-17 画出混淆图

from sklearn.metrics import confusion_matrix

yy_pred = model.predict(db_test)
# yy_pred.shape  是(1868,25),所以要转化成类别
yy_pred=tf.convert_to_tensor(yy_pred)
yy_pred=(tf.argmax(yy_pred,axis=1)).numpy()


bbbtrue=np.array(labels3)
print( confusion_matrix(bbbtrue,yy_pred))


####因为训练要很长时间，所以训练得到的数据存盘后，可以用于以后的绘图用
np.save("zby_VGG_pred_100_32_448_448_3.npy",yy_pred)
np.save("zby_VGG_true_100_32_448_448_3.npy",bbbtrue)
####上面的100表示１００个epoch,448*448是分辨率。



 
#labels表示你不同类别的代号，比如这里的demo中有25类别
#labels = ['0','1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13','14','15','16','17','18','19','20','21','22','23','24']
labels = ['Adialer.C', 'Agent.FYI', 'Allaple.A', 'Allaple.L', 'Alueron.gen!J', 'Autorun.K', 'C2LOP.P', 'C2LOP.gen!g', 'Dialplatform.B', 'Dontovo.A', 'Fakerean', 'Instantaccess', 'Lolyda.AA1', 'Lolyda.AA2', 'Lolyda.AA3', 'Lolyda.AT', 'Malex.gen!J', 'Obfuscator.AD', 'Rbot!gen', 'Skintrim.N', 'Swizzor.gen!E', 'Swizzor.gen!I', 'VB.AT', 'Wintrim.BX', 'Yuner.A']

y_true = bbbtrue
y_pred = yy_pred

np.save("zby_VGG_y_pred.npy",y_pred)
np.save("zby_VGG_y_true.npy",y_true)

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
    plt.text(x_val, y_val, "%0.2f" % (c,), color='yellowgreen', fontsize=7, va='center', ha='center')
# offset the tick
plt.gca().set_xticks(tick_marks, minor=True)
plt.gca().set_yticks(tick_marks, minor=True)
plt.gca().xaxis.set_ticks_position('none')
plt.gca().yaxis.set_ticks_position('none')
plt.grid(True, which='minor', linestyle='-')
plt.gcf().subplots_adjust(bottom=0.15)
 
plot_confusion_matrix(cm_normalized, title='Normalized confusion matrix')
# show confusion matrix
plt.savefig('confusion_matrix_VGG16.png', format='png')
plt.show()

###统计一下用时多长
t2 = time.time()
t3 =  t2-t1
print("本次用时为：")
print(t3/3600)

