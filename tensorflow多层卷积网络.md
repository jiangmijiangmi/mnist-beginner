

# 多层卷积网络

## 权重初始化



![img](http://images2015.cnblogs.com/blog/1015872/201611/1015872-20161110214243420-685796554.png)

使用的是ReLU神经元，因此比较好的做法是用一个较小的正数来初始化偏置项，以避免神经元节点输出恒为0的问题（dead neurons）。

```python
>>> def weight_variable(shape):

...   initial = tf.truncated_normal(shape, stddev=0.1)

...   return tf.Variable(initial)

... 

>>> def bias_variable(shape):

...   initial = tf.constant(0.1, shape=shape)

...   return tf.Variable(initial)


```

在这里使用了

tf.truncated_normal(shape, mean=0.0, stddev=1.0, dtype=tf.float32, seed=None, name=None)

参数:

- shape: 一维的张量，也是输出的张量。
- mean: 正态分布的均值。 
- stddev: 正态分布的标准差。
- dtype: 输出的类型。
- seed: 一个整数，当设置之后，每次生成的随机数都一样。
- name: 操作的名字。

这个函数产生正太分布，均值和标准差自己设定。这是一个截断的产生正太分布的函数，就是说产生正太分布的值如果与均值的差值大于两倍的标准差，那就重新生成。

## 卷积与池化

```python
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')
```

在**相同填充**中，超出边界的部分使用补充0的办法，使得输入输出的图像尺寸相同。

在**有效填充**中，则不使用补充0的方法，不能超出边界，因此往往输入的尺寸大于输出的尺寸。

在这里使用的是步长为1的相同填充。

![img](http://images2015.cnblogs.com/blog/1015872/201611/1015872-20161107225730639-1656228312.gif)

为了引入不变性，同时防止过拟合问题或欠拟合问题、降低计算量，我们常进行池化处理。池化过程如上图所示。因此池化过后，通常图像的宽度和高度会变成原来的1/2。

**平均池化**：计算图像区域的平均值作为该区域池化后的值。

**最大池化**：选图像区域的最大值作为该区域池化后的值。

### 卷积

```python
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
x_image = tf.reshape(x, [-1,28,28,1])#第一维的－1是指我们可以先不指定，如果为rgb彩色图则第四维为3
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)
```

```python
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)
```

第一层由一个卷积接一个max pooling完成。卷积在每个5x5的patch中算出32个特征。

第二层中，每个5x5的patch会得到64个特征。

---

**32和64是指定的。也可以指定为其它特征。如在LeNet的5层结构中，将第一层指定为6个特征。**

**在池化过程中，使用ksize = [1,2,2,1]和[1,2,2,1]的步幅，因此池化后结果为：14x14。所以，在卷积后大小不变；池化后，每个维度缩小一倍。最终在全联接时大小为7x7。**

---

图片尺寸减小到7x7，我们加入一个有1024个神经元的全连接层，用于处理整个图片。我们把池化层输出的张量reshape成一些向量，乘上权重矩阵，加上偏置，然后对其使用ReLU。

```python
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
```

tf.nn.dropout是TensorFlow里面为了防止或减轻过拟合而使用的函数，它**一般用在全连接层**。

完全随机选取经过神经网络流一半的数据来训练，在每次训练过程中用0来替代被丢掉的激活值，其它激活值合理缩放。

```python
keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
```

**tf.nn.dropout(x, keep_prob, noise_shape=None, seed=None,name=None) **

* 第一个参数x：指输入
* 第二个参数keep_prob: 设置神经元被选中的概率,在初始化时keep_prob是一个占位符,keep_prob = tf.placeholder(tf.float32) 。tensorflow在run时设置keep_prob具体的值，例如keep_prob: 0.5
* 第五个参数name：指定该操作的名字。

## 输出

```python
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
```

## 训练和评估

```python
#计算交叉熵的代价函数
cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
#使用优化算法使得代价函数最小化
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
#找出预测正确的标签
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
sess.run(tf.initialize_all_variables())
#每100次迭代输出一次日志，共迭代20000次
for i in range(20000):
  batch = mnist.train.next_batch(50)
  if i%100 == 0:
    train_accuracy = accuracy.eval(feed_dict={
        x:batch[0], y_: batch[1], keep_prob: 1.0})
    print("step %d, training accuracy %g"%(i, train_accuracy))
  train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

print("test accuracy %g"%accuracy.eval(feed_dict={
    x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
```

再次强调一下python3与python2的输出区别。

显卡：Radeon Pro 560，两万次迭代差不多运算了40分钟。

```
step 0, training accuracy 0.08
step 100, training accuracy 0.66
step 200, training accuracy 0.86
step 300, training accuracy 0.82
step 400, training accuracy 0.94
step 500, training accuracy 0.92
step 600, training accuracy 0.94
step 700, training accuracy 0.94
step 800, training accuracy 0.94
step 900, training accuracy 0.98
step 1000, training accuracy 0.94
step 1100, training accuracy 0.96
step 1200, training accuracy 0.96
step 1300, training accuracy 1
step 1400, training accuracy 0.96
step 1500, training accuracy 0.96
step 1600, training accuracy 1
step 1700, training accuracy 0.96
step 1800, training accuracy 0.94
step 1900, training accuracy 1
step 2000, training accuracy 0.96
step 2100, training accuracy 0.96
step 2200, training accuracy 1
step 2300, training accuracy 0.92
step 2400, training accuracy 0.98
step 2500, training accuracy 1
step 2600, training accuracy 1
step 2700, training accuracy 1
step 2800, training accuracy 1
step 2900, training accuracy 1
step 3000, training accuracy 1
step 3100, training accuracy 0.98
step 3200, training accuracy 0.98
step 3300, training accuracy 1
step 3400, training accuracy 1
step 3500, training accuracy 0.96
step 3600, training accuracy 0.94
step 3700, training accuracy 0.98
step 3800, training accuracy 0.98
step 3900, training accuracy 0.98
step 4000, training accuracy 0.98
step 4100, training accuracy 0.98
step 4200, training accuracy 1
```

大概到8000次左右就比较稳定了，精确度基本没有波动。

### 参考资料

http://wiki.jikexueyuan.com/project/tensorflow-zh/tutorials/mnist_pros.html

http://www.cnblogs.com/rgvb178/p/6052541.html

http://www.cnblogs.com/rgvb178/p/6017991.html

http://blog.csdn.net/danieljianfeng/article/details/42433475

https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf

http://www.cnblogs.com/tornadomeet/p/3258122.html

https://www.zhihu.com/question/46889310