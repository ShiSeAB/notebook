# Word Embedding

​	*词向量*是用来表示词的向量，也可以被认为是词的特征向量或表示。将词映射到真实向量的技术称为词嵌入 Word Embedding

## 1. One-Hot Vector

如何表示词？

用 **one-hot** 编码；假设字典中不同单词的数量（字典大小）为N，并且每个单词对应于从0到N-1的不同整数（索引）。为了获得索引为i的任何单词的one-Hot向量表示，我们创建一个长度为N的向量，初始化所有位置为0并将位置为i的元素设置为1。这样，每个单词都表示为长度为N的向量，它可以被神经网络直接使用。

如宝马、奔驰和奥迪分别用100、010和001表示

缺陷：

- one-hot word向量不能准确地表达不同单词之间的相似度，因其任意两个不同单词的余弦相似度为 0

  ![image-20250307145201675](./Word%20Embedding.assets/image-20250307145201675.png)

## 2. 自监督模型

​	You can get a lot of value by representing a word by means of its neighbors. 如何了解一个词的意思，通过其上下文来了解。

![image-20250307150057333](./Word%20Embedding.assets/image-20250307150057333.png)



### 2.1 CBOW模型

​	![image-20250307150349461](./Word%20Embedding.assets/image-20250307150349461.png)

去掉apple，把其上下文送入神经网络，让其预测被去掉的词是apple

<img src="./Word%20Embedding.assets/image-20250307150506276.png" alt="image-20250307150506276" style="zoom: 50%;" />



### 2.2 SG模型

![image-20250307150443645](./Word%20Embedding.assets/image-20250307150443645.png)

将apple送入神经网络，让其预测apple的上下文

<img src="./Word%20Embedding.assets/image-20250307150737812.png" alt="image-20250307150737812" style="zoom:50%;" />

### 3.3 结合

将两个模型结合起来，CBOW的输出作为SG的输入：

![image-20250307150926861](./Word%20Embedding.assets/image-20250307150926861.png)

实例：

输入一个1x10维独热码，根据输入权重矩阵(10x5)得到真实词向量表达(1x5)，激活后再乘输出权重矩阵(10x5)，根据softmax分类确定最终结果：

![image-20250307152327323](./Word%20Embedding.assets/image-20250307152327323.png)

使联系紧密的词距离更近，联系不紧密的词距离更远：

![image-20250307153202206](./Word%20Embedding.assets/image-20250307153202206.png)

用欧式距离来度量，进行计算

![image-20250307153222364](./Word%20Embedding.assets/image-20250307153222364.png)

