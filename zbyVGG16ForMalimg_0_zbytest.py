# -*- coding: utf-8 -*-
"""
Created on Wed Nov 17 16:16:41 2021

@author: Dell_hnjxzby
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Oct  8 23:02:38 2021
用resnet18来测试一下全部数据的分类效果，先前用 densenet121模型分类时，有两个类分不开。
（5类和24类，确实用肉眼也很难分清），但是将样本均衡化和标准化后，分得很好。
我的采样很简单，就是将样本少的类5复制了很多次。

不知道用resnet18怎么样？？？成功，但是对swizzor的两类分类不太成功！！但比densenet121好！！，晚上多训练一下，200次epock，看看效果！！
或者测试一下resnet34???

综上，所以今天我要测试一下VGG16的分类效果。 hnjxzby@2021-10-17  分类效果非常差？？？？

回答： 经过今天实验发现，以前训练要么没有开始，是因为model.fit()出了错。
而准确率与ｌｏｓｓ一直不变化的原因是： 我们是迁移学习，各种参数是从ｉｍａｇｅｎｅｔ直接拿来用的。所以：
下面这一语句很重要：
   VGG16_MODEL.trainable=False

我将它从True设置为flase 以后，程序运行正常，剩下的就交给时间了。。。长时间的等待，用更好的GPU吧 ！！！
 
hnjxzby@2021-10-30  为什么第04类和24类分类效果比较差呢？我用肉眼看了一下，确实很相似。其实，这里训练产生准确率不高的原因是：这些样本的原始分辨率比较大
所以我们实验时不能将样本图缩小得太小，448*448这个大小时我用Ｒｅｓnet１８ 分类可以达到 １００％

因此我准备用VGG１６也用 大的分辨率，看实验结果如何？   


#############################################################################################################
回答：结果非常好，也达到了近100%的准确率，只是可能需要较大的内存和显存空间吧。
我测试的每张图片的分辨率是448*448。本次训练用时为：
8.226230519082812
Epoch 50/50
692/692 [==============================] - 416s 601ms/step - loss: 0.0263 - accuracy: 0.9914 - val_loss: 0.0136 - val_accuracy: 0.9980
20/20 [==============================] - 18s 904ms/step - loss: 0.0135 - accuracy: 0.9969
loss: 0.01
accuracy: 1.00

@author: WIN10
"""


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
    ## x = tf.image.per_image_standardization(x) ### 这是我增加的规范化的一个办法。hnjxzby@2021-10-08  
    x = tf.image.resize(x, [448, 448])
    x = tf.image.random_flip_left_right(x)
    x = tf.image.random_flip_up_down(x)
    x = tf.image.random_crop(x, [448,448,3])
    # x: [0,255]=> -1~1
    ####我准备测试一下在不同分辨率情况下的分类准确率
    ####如[224, 224]，如 [112, 112]，如56*56，等。
    x = tf.cast(x, dtype=tf.float32) / 255. 
    
    #x = tf.image.resize(x, [448,448])
    #x = normalize(x)
    y = tf.convert_to_tensor(y)
    y = tf.one_hot(y, depth=25)  ###改成25类
    return x, y


batchsz = 32
#batchsz = 32

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


##########下面是选择不同的model，以前我用的是Densenet121,而后测试的是ResNet18. 现在我准备测试VGG16,看看效果是否有提高？
#############hnjxzby@2021-10-17  22：56

num_classes = 25

##############作为实验，我想测试一下分辩率分别为不同时的分类效果，如：
###resnet18.build(input_shape=(4,224,224,3))
###resnet18.build(input_shape=(4,112,112,3))
###resnet18.build(input_shape=(4,56,56,3))
###resnet18.build(input_shape=(4,224,224,3))
###resnet18.summary()
###resnet18.compile(optimizer=optimizers.Adam(lr=1e-3),
###               loss=losses.CategoricalCrossentropy(from_logits=True),
###               metrics=['accuracy'])

IMG_SIZE=448
#BATCH_SIZE = 32

IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)
###VGG16_MODEL=tf.keras.applications.VGG16(weights='imagenet',input_shape=IMG_SHAPE,include_top=False)                                          
VGG16_MODEL=tf.keras.applications.VGG16(weights='imagenet',input_shape=IMG_SHAPE,include_top=False)                                                                          

VGG16_MODEL.trainable=False

###捣鼓好多天，发现将上面一行，设置为false, 训练时的准确率直线上升，5个epoch就可以达到81%。
###而将trainable设置为True,怎么训练 acc 和 loss都不变，浪费我好多时间。


###千万注意，要在第一次实验中将我们添加的分类器各层训练好后，再解冻vgg16中的block5中的层，再训练一次才行。
#VGG16_MODEL.trainable=True
#fine_tune_at = -4
#for layer in VGG16_MODEL.layers[:fine_tune_at]:
#    layer.trainable = False
#for layer in VGG16_MODEL.layers:
#    print(layer.name,layer.trainable) 

   
global_average_layer = tf.keras.layers.GlobalAveragePooling2D()  #呵呵，这一层将14*14*512已展平了，所以layers.Flatten好象可以不要了，二者有一样的效果啊

#dense_layer= tf.keras.layers.Dense(512,activation = 'relu')      #我增加了一层，因为VGG16的输出是7*7*512,在此接合一下？
#prediction_layer = tf.keras.layers.Dense(25,activation='softmax')#我将sigmoid改成了softmax应该正确??


model = tf.keras.Sequential()

model.add(VGG16_MODEL)
model.add(global_average_layer)
#model.add(tf.keras.layers.Flatten())  #必须展平
model.add(tf.keras.layers.Dense(512, activation='relu'))   #我增加了一层，因为VGG16的输出是7*7*512,在此接合一下？
model.add(tf.keras.layers.Dropout(0.5))                    #我增加了一层dropout.
#model.add(tf.keras.layers.Dense(25, activation="softmax"))
model.add(tf.keras.layers.Dense(25))  #如果在此去掉了activation,则一定要在model.compile中将from_logits设置为true

"""
model = tf.keras.Sequential([
  VGG16_MODEL,
  global_average_layer,
  dense_layer,                                                    #我增加了一层
  prediction_layer
])
"""


model.build(input_shape=(4,448,448,3))
model.summary()

#####4. 编译模型
model.compile(optimizer=optimizers.Adam(learning_rate=1e-3),
             loss=losses.CategoricalCrossentropy(from_logits=True),
             metrics=["accuracy"])

###以前的训练效果太差，是原因出在这吗？ 是否是应该将 from_logits=False


img_, label_ = next(iter(db_train))
pred_ = model.predict(img_)#此tensorflow版本在训练前，必须要先使用model.predict方法，否则后面训练会报错，原因未知
print(pred_.shape)

img_1, label_1 = next(iter(db_train))
pred_1 = model.predict(img_1)#此tensorflow版本在训练前，必须要先使用model.predict方法，否则后面训练会报错，原因未知
print(pred_1.shape)

#####5. 训练
import datetime
####用tensorboard可视化
log_dir=r"d:\\zbylog\\" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

early_stopping = EarlyStopping(
    monitor='val_accuracy',
    min_delta=0.001,
    patience=10
)


"""
history = model.fit(db_train,
                    epochs=2, 
                    steps_per_epoch=2,
                    validation_steps=2,
                    validation_data=db_val,
					callbacks=[tensorboard_callback])
"""
###用上面的fit来做训练，感觉没有对数据集进行测试，直接跳过去一样快，效果很差。

"""
#history  = resnet.fit(db_train, validation_data=db_val, validation_freq=1, epochs=50,
#           callbacks=[early_stopping])
# 创建Early Stopping类，连续10次不下降则终止

)
"""




history = model.fit(db_train,
                    epochs=50, 
                    validation_data=db_val,
                    callbacks=[early_stopping] )


                    
##用上面这行，训练时间很长，感觉是在真正地干活。。。。。。
##在我家的电脑是运行一个epoch需要时间5分钟。
##虽然在干活，但是好像分类结果仍然很差劲！！！！！！！hnjxzby@2021-10-21 thur 19:45

model.save('Model_VGG16_before_finetune_448_448.h5')

#####6. 评估模型(test)
validation_steps = 20
loss0,accuracy0 = model.evaluate(db_test, steps = validation_steps)
print("loss: {:.2f}".format(loss0))
print("accuracy: {:.2f}".format(accuracy0))

####7. 打印学习曲线
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# 8 打印损失函数
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

print(history['val_loss'])
print(history['loss'])

############################################################
#hnjxzby@2021-10-17 画出混淆图


from sklearn.metrics import confusion_matrix

yy_pred = model.predict(db_test)
# yy_pred.shape  是(1868,25),所以要转化成类别
yy_pred=tf.convert_to_tensor(yy_pred)
yy_pred=(tf.argmax(yy_pred,axis=1)).numpy()

bbbtrue=np.array(labels3)
print( confusion_matrix(bbbtrue,yy_pred))
 
#labels表示你不同类别的代号，比如这里的demo中有25类别
#labels = ['0','1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13','14','15','16','17','18','19','20','21','22','23','24']
labels = ['Adialer.C', 'Agent.FYI', 'Allaple.A', 'Allaple.L', 'Alueron.gen!J', 'Autorun.K', 'C2LOP.P', 'C2LOP.gen!g', 'Dialplatform.B', 'Dontovo.A', 'Fakerean', 'Instantaccess', 'Lolyda.AA1', 'Lolyda.AA2', 'Lolyda.AA3', 'Lolyda.AT', 'Malex.gen!J', 'Obfuscator.AD', 'Rbot!gen', 'Skintrim.N', 'Swizzor.gen!E', 'Swizzor.gen!I', 'VB.AT', 'Wintrim.BX', 'Yuner.A']


#################################################
#我在用DenseNet121做实验时，第5类样本，全部被错误分类成第24类了！！！！！
#用肉眼一看，这两类的图像还真的是很像，傻傻分不清！！有什么办法呢？hnjxzby@2021-10-06
#现在我用VGG来实验，看看效果是否有提高？  结果也一样哦！！！！，也是分不清！！！hnjxzby@2021-10-22
################################################ 
 
'''
具体解释一下re_label.txt和pr_label.txt这两个文件，比如你有100个样本
去做预测，这100个样本中一共有10类，那么首先这100个样本的真实label你一定
是知道的，一共有10个类别，用[0,9]表示，则re_label.txt文件中应该有100
个数字，第n个数字代表的是第n个样本的真实label（100个样本自然就有100个
数字）。
同理，pr_label.txt里面也应该有1--个数字，第n个数字代表的是第n个样本经过
你训练好的网络预测出来的预测label。
这样，re_label.txt和pr_label.txt这两个文件分别代表了你样本的真实label和预测label，然后读到y_true和y_pred这两个变量中计算后面的混淆矩阵。当然，不一定非要使用这种txt格式的文件读入的方式，只要你最后将你的真实
label和预测label分别保存到y_true和y_pred这两个变量中即可。
'''
###################################################


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

###统计一下，我们的训练用时多长
t2 = time.time()
t3 =  t2-t1
print("本次训练用时为：")
print(t3/3600)

