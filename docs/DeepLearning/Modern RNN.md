# 现代循环神经网络

## 1. 门控循环单元GRU

​	在关注一个序列时，不是每个观察值都是同等重要，为了记住重要的观察值，我们引入 **更新门** （可以关注重要信息）和 **重置门**（可以遗忘不重要的信息）。

### 1.1 门

$R_t$ 表示Reset gate，$Z_t$ 表示update gate ，激活函数为sigmoid，以保证两个门矩阵的元素值在(0,1)之间。

相较之前的RNN网络又多了一些权重参数。

![image-20250314234439192](./Modern%20RNN.assets/image-20250314234439192.png)

### 1.2 候选隐状态

$\bigodot$ 表示元素点乘， $R_t$ 中元素为1的话，则 $H_{t-1}$ 相应位置元素通过，恢复之前介绍的普通神经网络；为0则被遗忘，以减少以往状态 $H_{t-1}$ 的影响，更多依赖输入 $X_t$。激活函数为tanh，保证candidate hidden state中的值保持在区间(-1,1)之间。

由于 $R_t$ 的激活函数是sigmoid，其元素值介于(0,1)。

![image-20250315121339959](./Modern%20RNN.assets/image-20250315121339959.png)

### 1.3 隐状态

结合 $Z_t$ 的效果，看新的隐状态 $H_T \in \mathbb{R}^{n\times h}$ 在多大程度上来自旧的状态 $H_{t-1}$ 和新的候选状态。当更新门接近1时，模型就倾向于只保留旧状态， 从而有效地跳过了依赖链条中的时间步t；而当更新门接近0时，新的隐状态就会接近候选隐状态。

这些设计可以帮助我们处理循环神经网络中的梯度消失问题， 并更好地捕获时间步距离很长的序列的依赖关系。 例如，如果整个子序列的所有时间步的更新门都接近于1， 则无论序列的长度如何，在序列起始时间步的旧隐状态都将很容易保留并传递到序列结束。

![image-20250315121409009](./Modern%20RNN.assets/image-20250315121409009.png)

当 $R_t$ 和 $Z_t$ 是全0时，相当于回到之前的循环神经网络

总之，门控循环单元具有以下两个显著特征：

- 重置门有助于捕获序列中的短期依赖关系；
- 更新门有助于捕获序列中的长期依赖关系。

### 1.4 实现

![image-20250315125825461](./Modern%20RNN.assets/image-20250315125825461.png)

```python
import torch
from torch import nn
from d2l import torch as d2l

batch_size, num_steps = 32, 35
train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)
#参数初始化
def get_params(vocab_size, num_hiddens, device):
    num_inputs = num_outputs = vocab_size

    def normal(shape):
        return torch.randn(size=shape, device=device)*0.01

    def three():
        return (normal((num_inputs, num_hiddens)),
                normal((num_hiddens, num_hiddens)),
                torch.zeros(num_hiddens, device=device))

    W_xz, W_hz, b_z = three()  # 更新门参数
    W_xr, W_hr, b_r = three()  # 重置门参数
    W_xh, W_hh, b_h = three()  # 候选隐状态参数
    # 输出层参数
    W_hq = normal((num_hiddens, num_outputs))
    b_q = torch.zeros(num_outputs, device=device)
    # 附加梯度
    params = [W_xz, W_hz, b_z, W_xr, W_hr, b_r, W_xh, W_hh, b_h, W_hq, b_q]
    for param in params:
        param.requires_grad_(True)
    return params
#定义模型
def init_gru_state(batch_size, num_hiddens, device):
    #返回一个形状为（批量大小，隐藏单元个数）的张量，张量的值全部为零。
    return (torch.zeros((batch_size, num_hiddens), device=device), )

def gru(inputs, state, params):
    W_xz, W_hz, b_z, W_xr, W_hr, b_r, W_xh, W_hh, b_h, W_hq, b_q = params
    H, = state
    outputs = []
    for X in inputs:
        #更新门
        Z = torch.sigmoid((X @ W_xz) + (H @ W_hz) + b_z)
        #遗忘门
        R = torch.sigmoid((X @ W_xr) + (H @ W_hr) + b_r)
        #候选隐状态
        H_tilda = torch.tanh((X @ W_xh) + ((R * H) @ W_hh) + b_h)
        #新的隐状态
        H = Z * H + (1 - Z) * H_tilda
        #输出
        Y = H @ W_hq + b_q
        outputs.append(Y)
    return torch.cat(outputs, dim=0), (H,)

vocab_size, num_hiddens, device = len(vocab), 256, d2l.try_gpu()
num_epochs, lr = 500, 1
model = d2l.RNNModelScratch(len(vocab), num_hiddens, device, get_params, init_gru_state, gru)
```



## 2. 长短期记忆网络LSTM

比GRU更加复杂。

### 2.1 门控记忆元

#### 输入门、忘记门和输出门

- 输出门（output gate）：从单元中输出条目
- 输入门（input gate）：决定何时将数据读入单元
- 遗忘门（forget gate）：重置单元的内容

![image-20250315141703561](./Modern%20RNN.assets/image-20250315141703561.png)

时间步t的门定义如下：大小都是 $n\times h$ ，使用sigmoid激活函数

![image-20250315140426153](./Modern%20RNN.assets/image-20250315140426153.png)

#### 候选记忆元

![image-20250315142148259](./Modern%20RNN.assets/image-20250315142148259.png)

#### 记忆单元

用于控制控制输入和遗忘，不像GRU那样只能选一个，由于 $F_t,I_t$ 是独立的，所以两个都可以选。其中输入门控制采用多少新数据，遗忘门控制保留多少过去的记忆。

![image-20250315142310744](./Modern%20RNN.assets/image-20250315142310744.png)

缓解梯度消失问题， 并更好地捕获序列中的长距离依赖关系

#### 隐状态

只要输出门接近1，我们就能够有效地将所有记忆信息传递给预测部分， 而对于输出门接近0，我们只保留记忆元内的所有信息，而不需要更新隐状态。

![image-20250315142901887](./Modern%20RNN.assets/image-20250315142901887.png)



### 2.2 实现

![image-20250315143223382](./Modern%20RNN.assets/image-20250315143223382.png)

类似GRU，只是公式不同



## 3. 深度循环神经网络

“更深” 。中间隐藏层的值既往上又往右传递：

![image-20250315144011884](./Modern%20RNN.assets/image-20250315144011884.png)

实现：

```python
#设置隐藏层数为2
vocab_size, num_hiddens, num_layers = len(vocab), 256, 2
num_inputs = vocab_size
device = d2l.try_gpu()
lstm_layer = nn.LSTM(num_inputs, num_hiddens, num_layers)
model = d2l.RNNModel(lstm_layer, len(vocab))
model = model.to(device)
```

## 4. 双向循环神经网络

文本预测需要根据上下文来推测，无法利用下文传达的信息的模型显然是不够完善的。

### 双向RNN

![image-20250315145436330](./Modern%20RNN.assets/image-20250315145436330.png)

![image-20250315145617755](./Modern%20RNN.assets/image-20250315145617755.png)

一条path从前往后看，另一条从后往前看，合并输出。

但是双向RNN非常不适合做推理，因为推理过程中没有“后文”。所以一般用来做特征提取、翻译工作。

### 实现

```python
# 通过设置“bidirective=True”来定义双向LSTM模型
vocab_size, num_hiddens, num_layers = len(vocab), 256, 2
num_inputs = vocab_size
lstm_layer = nn.LSTM(num_inputs, num_hiddens, num_layers, bidirectional=True)
model = d2l.RNNModel(lstm_layer, len(vocab))
model = model.to(device)
```

## 5. 编码器与解码器

![image-20250315161012538](./Modern%20RNN.assets/image-20250315161012538.png)

CNN 中，卷积层用于做特征提取，全连接层用于做Softmax回归。可以将卷积层看作编码器 -- 将输入编程成中间表达形式，全连接层看作解码器 -- 将中间表示解码成输出。

RNN同样也可以抽象这一结构：编码器将文本表示成向量，解码器将向量解码成输出。

encoder-decoder架构：

- 编码器处理输入
- 解码器生成输出

代码定义接口：

```python
from torch import nn


#@save
class Encoder(nn.Module):
    """编码器-解码器架构的基本编码器接口"""
    def __init__(self, **kwargs):
        super(Encoder, self).__init__(**kwargs)

    def forward(self, X, *args):
        raise NotImplementedError
        
#@save
class Decoder(nn.Module):
    """编码器-解码器架构的基本解码器接口"""
    def __init__(self, **kwargs):
        super(Decoder, self).__init__(**kwargs)

    def init_state(self, enc_outputs, *args):
        raise NotImplementedError

    def forward(self, X, state):
        raise NotImplementedError

#@save
class EncoderDecoder(nn.Module):
    """编码器-解码器架构的基类"""
    def __init__(self, encoder, decoder, **kwargs):
        super(EncoderDecoder, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, enc_X, dec_X, *args):
        enc_outputs = self.encoder(enc_X, *args)
        dec_state = self.decoder.init_state(enc_outputs, *args)
        return self.decoder(dec_X, dec_state)
```



## 6. Seq2Seq

机器翻译就是一个seq2seq模型。

### 推理模型

![image-20250315161723197](./Modern%20RNN.assets/image-20250315161723197.png)

- Encoder是一个RNN，用于读取输入，可以用双向RNN
- Decoder使用另外一个RNN来输出

传参细节：

![image-20250315164650698](./Modern%20RNN.assets/image-20250315164650698.png)

- 将encoder最后时间步的隐状态用作decoder的初始隐状态，所以encoder不需要全连接层来输出。

### 训练模型

![image-20250315165033662](./Modern%20RNN.assets/image-20250315165033662.png)

训练时解码器输入不仅是Encoder的隐状态，同时还要使用目标句子作为输入



### 评判生成序列好坏 -- BLEU

![image-20250315165316279](./Modern%20RNN.assets/image-20250315165316279.png)

uni-gram是一个token，2-gram是两个连续token，n-gram就是n个连续token。$P_n$ 中，分子是预测序列中与标签序列匹配的n-gram数量，分母是预测序列中总共的n-gram数量。

BLEU 越大效果越好：

![image-20250315165326573](./Modern%20RNN.assets/image-20250315165326573.png)

### 实现

……



## 7.束搜索

在seq2seq中我们使用了贪心搜索来预测序列 -- 将当前时刻预测概率最大的词输出。

- 贪心解有可能不是最优解（虽然效率很高）。
- 穷举搜索计算量又太大。

所以， 利用 bin search ：

设置超参数 beam size k，在时间步1，我们选择具有最高条件概率的k个词元。 这k个词元将分别是k个候选输出序列的第一个词元。在随后的**每个**时间步，基于上一时间步的k个候选输出序列，从 $k|\Upsilon|$ 个可能的选择中 挑出具有最高条件概率的k个候选输出序列.

![image-20250315170801260](./Modern%20RNN.assets/image-20250315170801260.png)

![image-20250315174721238](./Modern%20RNN.assets/image-20250315174721238.png)

在最终候选输出序列集合中选择其中条件概率**乘积最高**的序列作为输出序列（通常 $\alpha = 0.75$）：

![image-20250315174840727](./Modern%20RNN.assets/image-20250315174840727.png)

该选择公式使得结果不偏向短句子。
