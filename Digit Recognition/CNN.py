#!/usr/bin/env python 3.7
# -*- coding: utf-8 -*



import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import os
import time


#  1. 定义神经网络结构相关的参数。
INPUT_NODE = 784 # 输入层的节点数
OUTPUT_NODE = 10# 输出层的节点数
LAYER1_NODE = 500 # 隐藏层的节点数


# #### 2. 通过tf.get_variable函数来获取变量。

# 通过tf. get_variable函数来获取变量：在训练神经网络时会创建这些变量，在测试时会通过保存的模型保存这些变量的取值。现在更加方便的是由于可以在
# 变量加载时将滑动平均变量重命名，所以可以直接通过同样的名字在训练时使用变量本身，而在测试时使用变量的滑动平均值。在这个函数中也会将变量的
# 正则化损失加入损失函数
def get_weight_variable(shape, regularizer): # 此处的shape为[784x500]
    weights = tf.get_variable("weights", shape, initializer=tf.truncated_normal_initializer(stddev=0.1))  # 变量初始化函数：tf.truncated_normal_initializer
    # 当给出了正则化生成函数时，将当前变量的正则化损失加入名字为losses的集合，在这里使用了add_to_collection函数将一个张量加入一个集合，而这个
    # 集合的名称为losses。这是自定义的集合，不在tensorflow自动管理的集合列表内
    if regularizer != None:
        tf.add_to_collection('losses', regularizer(weights))
    return weights


# #### 3. 定义神经网络的前向传播过程。
def inference(input_tensor, regularizer):
    # 声明第一层神经网络的变量并完成前向传播过程
    with tf.variable_scope('layer1'):  # 要通过tf.get_variable获取一个已经创建的变量，需要通过 tf.variable_scope函数来生成一个上下文管理器。
        # 这里通过 tf.get_variable和 tf.variable没有本质的区别，因为在训练或测试中没有在同一个程序中多次调用这个函数。如果在同一个程序中多次调用
        # 在第一次调用后需要将reuse参数设置为true
        weights = get_weight_variable([INPUT_NODE, LAYER1_NODE], regularizer) # 权重
        biases = tf.get_variable("biases", [LAYER1_NODE], initializer=tf.constant_initializer(0.0)) # 偏置
        layer1 = tf.nn.relu(tf.matmul(input_tensor, weights) + biases)  # tf.nn.relu非线性激活函数

    # 类似的声明第二层神经网络的变量并完成前向传播过程
    with tf.variable_scope('layer2'):
        weights = get_weight_variable([LAYER1_NODE, OUTPUT_NODE], regularizer)
        biases = tf.get_variable("biases", [OUTPUT_NODE], initializer=tf.constant_initializer(0.0))
        layer2 = tf.matmul(layer1, weights) + biases

    # 返回最后前向传播的结果
    return layer2


# #### 1. 定义神经网络结构相关的参数。
BATCH_SIZE = 100 # 一个训练batch中的训练数据个数。个数越小越接近随机梯度下降；数字越大时，训练越接近梯度下降
LEARNING_RATE_BASE = 0.8 # 基础的学习率
LEARNING_RATE_DECAY = 0.99 # 学习率的衰减率
REGULARIZATION_RATE = 0.0001 # 描述模型复杂度的正则化项在损失函数中的系数
TRAINING_STEPS = 80000# 训练轮数
MOVING_AVERAGE_DECAY = 0.99  # 滑动平均衰减率
# 模型保存的路径和文件名
MODEL_SAVE_PATH = "./models"
MODEL_NAME = "model.ckpt"


# #### 2. 定义训练过程。
def train(mnist):
    # 定义输入输出placeholder（placeholder机制用于提供输入数据，该占位符中的数据只有在运行时才指定）
    x = tf.placeholder(tf.float32, [None, INPUT_NODE], name='x-input')
    y_ = tf.placeholder(tf.float32, [None, OUTPUT_NODE], name='y-input')
    # 这里使用L2正则化，tf.contrib.layers.l2_regularizer会返回一个函数，这个函数可以计算一个给定参数的L2正则化项的值
    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    # 直接使用mnist_inference.py中定义的前向传播函数
    y = inference(x, regularizer)
    global_step = tf.Variable(0, trainable=False)

    # 定义损失函数、学习率、滑动平均操作以及训练过程。

    # 定义指数滑动平均的类，初始化给点了衰减率0.99和控制衰减率的变量global_step
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables()) # 定义一个更新变量滑动平均的操作
    # 定义交叉熵损失：因为交叉熵一般和softmax回归一起使用，所以 tf.nn.sparse_softmax_cross_entropy_with_logits函数对这两个功能进行了封装。
    # 这里使用该函数进行加速交叉熵的计算，第一个参数是不包括softmax层的前向传播结果。第二个参数是训练数据的正确答案，这里得到的是正确答案的
    # 正确编号。
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
    # 计算当前batch中所有样例的交叉熵平均值
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    # 总损失等于交叉熵损失和正则化损失的和
    loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))
    # 设置指数衰减的学习率
    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,
        global_step,
        mnist.train.num_examples / BATCH_SIZE, LEARNING_RATE_DECAY,
        staircase=True)
    # 这里使用指数衰减的学习率。在minimize中传入global_step将会自动更新global_step参数，从而使学习率得到相应的更新
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
    # 在训练神经网络时，每过一遍数据既需要通过反向传播来更新神经神经网络的参数，又需要更新每一个参数的滑动平均值，这里的 tf.control_dependencies
    with tf.control_dependencies([train_step, variables_averages_op]):
        train_op = tf.no_op(name='train')

    # 初始化TensorFlow持久化类。
    saver = tf.train.Saver()
    with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
        tf.global_variables_initializer().run()
        # 在训练过程中不再测试模型在验证数据上的表现，验证和测试的过程将会有一个独立的程序来完成。
        for i in range(TRAINING_STEPS):
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            _, loss_value, step = sess.run([train_op, loss, global_step], feed_dict={x: xs, y_: ys})
            # 每1000轮保存一次模型
            if i % 1000 == 0:
                # 输出当前的训练情况。这里只输出了模型在当前训练batch上的损失函数大小，通过损失函数的大小可以大概了解训练的情况。在验证数据数据
                # 上的正确率会有一个单独的程序来完成。
                print("After %d training step(s), loss on training batch is %g." % (step, loss_value))
                # 保存当前的模型。这里给出了global_step参数，这样可以让每个被保存模型的文件名末尾加上训练的轮数。
                saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)


EVAL_INTERVAL_SECS = 10


def evaluate(mnist):
    with tf.Graph().as_default() as g:
        # 定义输入输出的格式
        x = tf.placeholder(tf.float32, [None, INPUT_NODE], name='x-input')
        y_ = tf.placeholder(tf.float32, [None, OUTPUT_NODE], name='y-input')
        validate_feed = {x: mnist.validation.images, y_: mnist.validation.labels}
        # 直接通过调用封装好的函数来计算前向传播结果。因为测试时不关注正则化的值，所以这里用于计算正则化损失的函数被设置为none
        y = inference(x, None)
        # 使用前向传播的结果计算正确率。如果需要对未来的样例进行分类，使用tf.argmax（）就可以得到输入样例的预测类别了
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        # 通过变量重命名的方式来加载模型，这样在前向传播的过程中就不需要调用求滑动平均的函数来获取平均值了。这样就可以完全共用mnist_inference.py
        # 中定义的前向传播过程
        variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)

        # 每隔10秒调用一次计算正确率的过程以检测训练过程中正确率的变化
        while True:
            with tf.Session() as sess:
                #  tf.train.get_checkpoint_state函数会通过checkpoint文件自找到目录中最新的文件名
                ckpt = tf.train.get_checkpoint_state("./models")
                # ckpt.model_checkpoint_path：表示模型存储的位置，不需要提供模型的名字，它会去查看checkpoint文件，看看最新的是谁，叫做什么。
                if ckpt and ckpt.model_checkpoint_path:
                    # 加载模型
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    # 通过文件名得到模型保存时迭代的轮数（split('/')[-1].split('-')[-1]：正则表达式）
                    global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                    accuracy_score = sess.run(accuracy, feed_dict=validate_feed)
                    print("After %s training step(s), validation accuracy = %g" % (global_step, accuracy_score))
                else:
                    print('No checkpoint file found')
                    return
            time.sleep(EVAL_INTERVAL_SECS)


# #### 3. 主程序入口。
def main(argv=None):
    # "/home/lilong/desktop/MNIST_data/"
    # mnist = input_data.read_data_sets("../../../datasets/MNIST_data", one_hot=True)
    mnist = input_data.read_data_sets("./images", one_hot=True)
    train(mnist)
    evaluate(mnist)

if __name__ == '__main__':
    main()
