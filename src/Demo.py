import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

"""
y约等于2x
"""

plotdata = {"batchsize": [], "loss": []}
def moving_average(a, w=10):
    if len(a) < w:
        return a[:]
    return [val if idx < w else sum(a[(idx-w):idx])/w for idx, val in enumerate(a)]

train_x = np.linspace(-1, 1, 100)
# y = 2x, 但加入了噪音
train_y = 2 * train_x + np.random.randn(*train_x.shape) * 0.3 

# 正向模型
# 创建模型
# 占位符
X = tf.placeholder("float")
Y = tf.placeholder("float")
# 模型参数
w = tf.Variable(tf.random_normal([1]), name="weight")
b = tf.Variable(tf.zeros([1]), name="bias")
# 前向结构
z = tf.multiply(X, w) + b

# 反向模型
cost = tf.reduce_mean(tf.square(Y - z))
learning_rate = 0.01
# 梯度下降
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# 训练模型
# 初始化所有值
init = tf.global_variables_initializer()
# 定义参数
training_epochs = 20
display_step = 2
# 启动session
with tf.Session() as sess:
    sess.run(init)
    # 存放批次值和损失值
    plotdata = {"batchsize": [], "loss": []}
    # 向模型输入数据
    for epoch in range(training_epochs):
        for (x, y) in zip(train_x, train_y):
            sess.run(optimizer, feed_dict={X: x, Y: y})
            # 显示训练的详细信息
            if epoch % display_step == 0:
                loss = sess.run(cost, feed_dict={X: train_x, Y: train_y})
                print("Epoch:", epoch + 1, "cost=", loss, "w=", sess.run(w), "b=", sess.run(b))
                if not (loss == "NA"):
                    plotdata["batchsize"].append(epoch)
                    plotdata["loss"].append(loss)
    print("Finished")
    print("cost=", sess.run(cost, feed_dict={X: train_x, Y: train_y}), "w=", sess.run(w), "b=", sess.run(b))

    # 图形显示
    # 显示模拟数据点
    plt.plot(train_x, train_y, 'ro', label='Original data')
    plt.plot(train_x, sess.run(w) * train_x + sess.run(b), label="Fittedline")
    plt.legend()
    plt.show()

    plotdata["avgloss"] = moving_average(plotdata["loss"])
    plt.figure(1)
    plt.subplot(211)
    plt.plot(plotdata["batchsize"], plotdata["avgloss"], 'b--')
    plt.xlabel('Minibatch number')
    plt.ylabel('Loss')
    plt.title('Minibatch run vs. Training loss')
    plt.show()