# TensorFlow学习笔记

## 了解TensorFlow

###   开发的基本步骤([Demo](./src/Demo.py))

-   定义TensorFlow输入节点

    -   通过占位符定义: `X = tf.placeholder("float")`

    -   通过字典类型: `input = {'x': tf.placeholder("float")}`

    -   直接定义输入节点: `train_x = np.linspace(-1, 1, 100)`

-   定义"学习参数"的变量

    -   直接定义

    ```python
    # 模型参数
    w = tf.Variable(tf.random_normal([1]), name="weight")
    b = tf.Variable(tf.zeros([1]), name="bias")
    ```

    -   字典定义

    ```python
    paradict = {
        'w': tf.Variable(tf.random_normal([1])),
        'b': tf.Variable(tf.zeros([1]))
    }
    ```

-   定义"运算"

    -   定义正向传播模型

    -   定义损失函数: 主要用于计算"输出值"和"目标值"之间的误差，配合反向传播使用

-   优化函数，优化目标

    -   通过反向模型优化学习参数

-   初始化所有变量

    ```python
    # 初始化所有值
    init = tf.global_variables_initializer()
    # 启动session
    with tf.Session() as sess:
        sess.run(init)
    ```

-   迭代更新参数到最优解

    ```python
    for epoch in range(training_epochs):
        for (x, y) in zip(train_x, train_y):
            sess.run(optimizer, feed_dict={X: x, Y: y})
    ```

-   测试模型

    ```python
    print("cost=", sess.run(cost, feed_dict={X: train_x, Y: train_y}), "w=", sess.run(w), "b=", sess.run(b))
    ```

-   使用模型

    ```python
    # 使用模型
    print("x=0.2, z=", sess.run(z, feed_dict={X: 0.2}))
    ```

### 保存模型

```python
# 用于保存模型
saver = tf.train.Saver()
# 模型保存地址
model_path = "log/minist-model.ckpt"

with tf.Session() as sess:
    # 保存模型
    save_path = saver.save(sess, model_path)
```

### 载入模型

```python
saver = tf.train.Saver()
# 模型保存地址
model_path = "log/minist-model.ckpt"

with tf.Session() as sess:
    # 初始化变量
    sess.run(tf.global_variables_initializer())
    # 恢复模型变量
    saver.restore(sess, model_path)
```