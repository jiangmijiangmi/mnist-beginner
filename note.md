# mnist-beginner
# mnist机器学习入门

## 愚蠢的我照着代码打都打不对系列
```
from tensorflow.examples.tutorials.mnist import input_data
>>> mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
```
这边的报错主要是连接超时，然后并不机智的我在小伙伴提醒下才冷静挂上了vpn。
```
Extracting MNIST_data/train-images-idx3-ubyte.gz
Extracting MNIST_data/train-labels-idx1-ubyte.gz
Extracting MNIST_data/t10k-images-idx3-ubyte.gz
Successfully downloaded t10k-labels-idx1-ubyte.gz 4542 bytes.
Extracting MNIST_data/t10k-labels-idx1-ubyte.gz
```
最终终于成功后简直感动地要哭出来。

```
AttributeError: module 'urllib' has no attribute 'urlretrieve'
```
错误是由版本更新导致的，在python3中正确用法是`urllib3.request.urlretrieve`。
终端的默认python版本我没有改，所以愚蠢的我在修改了源码中的路径后，又冷静用python2运行了代码。
不出意外的再次报错了= =
``` 
>>> import tensorflow as tf
>>> x=tf.placeholder("float",[None,784])
>>> W=tf.Variable(tf.zeros([784,10]))
>>> b=tf.Variable(tf.zeros([10]))
>>> y=tf.nn.softmax(tf.matmul(x,W)+b)
>>> y_ = tf.placeholder("float", [None,10])
>>> cross_entropy = -tf.reduce_sum(y_*tf.log(y))
>>> train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
>>> init = tf.initialize_all_variables()
>>> sess = tf.Session()
>>> sess.run(init)
>>> for i in range(1000):
...   batch_xs, batch_ys = mnist.train.next_batch(100)
...   sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
... 
>>> correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
>>> accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
>>> print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
0.895
```
输出部分要注意python3中print要加括号。
循环1000次最终的结果是0.895，确实是很不如人意啊
```
>>>for i in range(10000):               
...         batch_xs, batch_ys = mnist.train.next_batch(100)           
...         sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys}) 
... 
>>> correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
>>> accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
>>> print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
0.9134
>>> 
```
循环一万次后最终输出结果0.9134，和大部分指南上一样。




