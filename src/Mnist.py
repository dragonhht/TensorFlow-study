#!/usr/bin/env python
# -*- encoding: utf-8 -*-

"""
@File    :   Mnist.py
@Time    :   2020/01/05 18:18:21
@Author  :   Carlos Huang
@Version :   1.0
@Desc    :   None
"""
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import pylab 

mnist = input_data.read_data_sets("data/MNIST_data/", one_hot=True)

tf.reset_default_graph()
# 定义占位符
# MINIST的数据维度 28 * 28 = 784
x = tf.placeholder(tf.float32, [None, 784])
# 数字0~9, 共10个类别
y = tf.placeholder(tf.float32, [None, 10])

# 定义学习参数
w = tf.Variable(tf.random_normal([784, 10]))
b = tf.Variable(tf.zeros([10]))

# 正向传播模型
pred = tf.nn.softmax(tf.matmul(x, w) + b)

# 反向传播模型
cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(pred), reduction_indices=1))

# 定义参数
learning_rate = 0.01
# 使用梯度下降优化器
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# 用于保存模型
saver = tf.train.Saver()
# 模型保存地址
model_path = "log/minist-model.ckpt"

# 训练模型
# 训练迭代次数
training_epochs = 30
# 一次取出用于训练的数据数量
batch_size = 100
# 每一次训练显示中间状态
display_step = 1
# 启动Session
with tf.Session() as sess:
    # 初始化参数
    sess.run(tf.global_variables_initializer())
    # 开始循环训练
    for epoch in range(training_epochs):
        avg_cost = 0
        # 计算数据集循环次数
        total_batch = int(mnist.train.num_examples/batch_size)
        # 循环数据集
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            # 运行优化器
            _, c = sess.run([optimizer, cost], feed_dict={x: batch_xs, y: batch_ys})
            # 计算平均loss值
            avg_cost += c / total_batch
        # 显示训练详情
        if (epoch + 1) % display_step == 0:
            print("Epoch: ", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(avg_cost))
    print("Finished!")

    # 测试模型
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    # 计算准确率
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print("Accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))

    # 保存模型
    save_path = saver.save(sess, model_path)
    print("模型已保存于: %s" % save_path)

# 载入模型
print("开始载入模型")
with tf.Session() as sess:
    # 初始化变量
    sess.run(tf.global_variables_initializer())
    # 恢复模型变量
    saver.restore(sess, model_path)

    # 测试模型
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    # 计算准确率
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print("Accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))

    output = tf.argmax(pred, 1)
    batch_xs, batch_ys = mnist.train.next_batch(2)
    outputval, predv = sess.run([output, pred], feed_dict={x: batch_xs})
    print(outputval, predv, batch_ys)

    im = batch_xs[0]
    im = im.reshape(-1, 28)
    pylab.imshow(im)
    pylab.show()

    im = batch_xs[1]
    im = im.reshape(-1, 28)
    pylab.imshow(im)
    pylab.show()
