# ReLU 激活函数

该模型相对于sigmoid系主要变化有三点：

* 单侧抑制
* 相对宽阔的兴奋边界
* 稀疏激活性



## Rectified Linear Unit(ReLU)

该激活函数表达式为

![img](http://img.blog.csdn.net/20161219175003614)

即在x>=0的情况下，激活函数值取x，在x<0的情况下，函数值取0。

它的作用是如果计算出的值小于0，就让它等于0，否则保持原来的值不变。这是一种简单粗暴地强制某些数据为0的方法，然而经实践证明，训练后的网络完全具备适度的稀疏性。而且训练后的可视化效果和传统方式预训练出的效果很相似，这也说明了ReLU具备引导适度稀疏的能力。

ReLU激活函数的导数表达式为：

![img](http://img.blog.csdn.net/20161219192334649)

相对于sigmoid类函数来说，不存在“梯度消失”现象，即导数从0开始很快又趋近于0。

ReLU在经历预训练和不经历预训练时的效果差不多，而其它激活函数在不用预训练时效果就差多了。ReLU不预训练和sigmoid预训练的效果差不多，甚至还更好。
相比之下，ReLU的速度非常快，而且精确度更高。

因此ReLU在深度网络中已逐渐取代sigmoid而成为主流。

## 稀疏激活性

1. 神经元具有稀疏激活性。
2. 神经元编码工作方式具有稀疏性和分布性。
3. 神经元同时只对输入信号的少部分选择性响应，大量信号被刻意的屏蔽了。

sigmoid系函数同时近乎有一半的神经元被激活，不符合神经科学的研究。

稀疏性有很多优势。但是，过分的强制稀疏处理，会减少模型的有效容量。即特征屏蔽太多，导致模型无法学习到有效特征。

论文中对稀疏性的引入度做了实验，理想稀疏性（强制置0）比率是70%~85%。超过85%，网络就容量就成了问题，导致错误率极高。

## 变体

* noisy ReLUs

可将其包含Gaussian noise得到noisy ReLUs，f(x)=max(0,x+(0,σ(x)))，常用来在机器视觉任务里的restricted Boltzmann machines中。

* leaky ReLUs

即修正了数据分布，又保留了一些负轴的值，使得负轴信息不会全部丢失。

* Parametric ReLUs

take this idea further by making the coefficient of leakage into a parameter that is learned along with the other neural network parameters

* Randomized Leaky ReLU 

是 leaky ReLU 的random 版本 （α 是random的，它首次试在 kaggle 的NDSB 比赛中被提出的。

![此处输入图片的描述](http://7pn4yt.com1.z0.glb.clouddn.com/blog-rrelu.png)



普通的ReLU负数端斜率是0，Leaky ReLU则是负数端有一个比较小的斜率，而PReLU则是在后向传播中学习到斜率。而Randomized Leaky ReLU则是使用一个均匀分布在训练的时候随机生成斜率，在测试的时候使用均值斜率来计算。

