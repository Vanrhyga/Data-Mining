#!/usr/bin/env python 3.7
# -*- coding: utf-8 -*



from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf


#生成权重
def weightVariable(shape):
    #加入少量噪声，打破对称性和0梯度
    initial=tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial)

#生成偏差
def biasVariable(shape):
    initial=tf.constant(0.1,shape=shape)
    return tf.Variable(initial)

#x:需进行卷积的图片，W:卷积核
def conv2d(x,W):
    #strides：卷积核步长，[1,x_movement,y_movement,1]
    #x，y轴都是每隔一个像素移动，步长为1
    #padding为SAME，卷积后图片与原图片大小相同，存在填充
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')

#x:卷积后图片
def maxPool2x2(x):
    #ksize：池化算子，[1,pool_op_length,pool_op_width,1]
    #2x2，长宽均为2
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

#导入数据集
images=input_data.read_data_sets('./images',one_hot=True)

#占位符定义输入
#x:图片集，y:标签，keepProb：dropout概率
x=tf.placeholder(tf.float32,[None,784]) #28X28
y=tf.placeholder(tf.float32,[None,10])
keepProb=tf.placeholder(tf.float32)

#四维张量，[batch,height,width,channels]
xImage=tf.reshape(x,[-1,28,28,1])

#conv1 layer
#卷积核大小为5x5，32个卷积核
Wconv1=weightVariable([5,5,1,32])   #patch 5x5,in size 1,out size 32
bconv1=biasVariable([32])
hconv1=tf.nn.relu(conv2d(xImage,Wconv1)+bconv1) #output size 28x28x32

#pool1 layer
hpool1=maxPool2x2(hconv1)   #output size 14x14x32

#conv2 layer
Wconv2=weightVariable([5,5,32,64])  #patch 5x5,in size 32,out size 64
bconv2=biasVariable([64])
hconv2=tf.nn.relu(conv2d(hpool1,Wconv2)+bconv2) #output size 14x14x64

#pool2_layer
hpool2=maxPool2x2(hconv2)   #output size 7x7x64

#四维张量变为二维张量，第一维是样本数，第二维是输入特征，输入神经元的个数是7*7*64
hpool2Flat=tf.reshape(hpool2,[-1,7*7*64])

#fc1 layer
Wfc1=weightVariable([7*7*64,1024])
bfc1=biasVariable([1024])
hfc1=tf.nn.relu(tf.matmul(hpool2Flat,Wfc1)+bfc1)

#dropout
#防止过拟合
hfc1Drop=tf.nn.dropout(hfc1,keepProb)

#fc2 layer
Wfc2=weightVariable([1024,10])
bfc2=biasVariable([10])

#softmax layer
yconv=tf.nn.softmax(tf.matmul(hfc1Drop,Wfc2)+bfc2)

crossEntropy=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=yconv))    #loss

#Adam优化器
trainStep=tf.train.AdamOptimizer(1e-4).minimize(crossEntropy)

correctPrediction=tf.equal(tf.argmax(yconv,1),tf.argmax(y,1))

accuracy=tf.reduce_mean(tf.cast(correctPrediction,tf.float32))

with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(40000):
        batch=images.train.next_batch(50)
        #每100次测试一次
        if i % 100 == 0:
            #测试时不加dropout
            trainAccuracy=accuracy.eval(feed_dict={x:batch[0],y:batch[1],keepProb:1.0})
            print('step %d,training accuracy %g'%(i,trainAccuracy))
        trainStep.run(feed_dict={x:batch[0],y:batch[1],keepProb:0.5})
    print('test accuracy %g'% accuracy.eval(feed_dict={x:images.test.images,y:images.test.labels,keepProb:1.0}))