# Deep Learning Basic

ML = Looking for a function 寻找一个映射

- Speed Recognition
- Image Recognition

![image-20250228134355651](./Deep%20Learning%20Basic.assets/image-20250228134355651.png)

​	Model f1 性能优于 model f2；通过收集的Training data输入model后输出的正确性来找到表现最好的model，这是一种监督学习(Supervised Learning)

![image-20250228135546259](./Deep%20Learning%20Basic.assets/image-20250228135546259.png)

- Playing Go 下围棋
- Dialogue System

3 steps for Deep Learning：

<img src="./Deep%20Learning%20Basic.assets/image-20250228135735952.png" alt="image-20250228135735952" />

## 1. define a set of function

目前使用的模型便是神经网络。

### 1.1 Neural Network

- Neuron

  ![image-20250228140702506](./Deep%20Learning%20Basic.assets/image-20250228140702506.png)

  常见激活函数：Sigmoid、tanh、ReLu

  ReLu最常用 -- 计算简单、不会太容易出现梯度消失

- Network

  ![image-20250228143035388](./Deep%20Learning%20Basic.assets/image-20250228143035388.png)

  ![image-20250228143057695](./Deep%20Learning%20Basic.assets/image-20250228143057695.png)

  采用全连接(fully connect feedforward net):

  - 计算简单：看作输入向量 X 和权重矩阵 W 相乘，也比较好求导
  - 结构易变化



- Deep or Wide？

![image-20250228143825431](./Deep%20Learning%20Basic.assets/image-20250228143825431.png)

​	越深的神经网络记忆能力越好，但对算力要求越高



## 2. goodness of function

- Training Data

  ![image-20250228144346257](./Deep%20Learning%20Basic.assets/image-20250228144346257.png)

  Softmax 用于分类回归：

  ![image-20250228144719888](./Deep%20Learning%20Basic.assets/image-20250228144719888.png)



- Loss function

  loss越少， 说明模型拟合效果越好. 利用MSE求loss，再求和。

  ![image-20250228145336159](./Deep%20Learning%20Basic.assets/image-20250228145336159.png)



## 3. pick the best function

优化过程：通过对损失函数求梯度，确定优化方向，从而一步步接近最优解

![image-20250228145459186](./Deep%20Learning%20Basic.assets/image-20250228145459186.png)

![image-20250228145615637](./Deep%20Learning%20Basic.assets/image-20250228145615637.png)

![image-20250228145808484](./Deep%20Learning%20Basic.assets/image-20250228145808484.png)

对学习率这个超参的设置很重要。

只能找到local Minima，无法确定是全局最优解。且有可能停在saddle point(梯度消失情况下)

![image-20250228150212258](./Deep%20Learning%20Basic.assets/image-20250228150212258.png)

