# Attention Mechanisms

在心理学上，动物需要在复杂环境下有效关注值得注意的点。

- 人类根据 **非自主性提示** 和 **自主性提示** 选择注意点
- **非自主性提示** 是基于环境中物体的突出性和易见性，引起人的注意
- **自主性提示** 则是受主观意愿推动去注意

卷积、全连接、池化层都只考虑“非自主性提示”，哪个显眼就注意哪个。

而注意力机制则考虑 “自主性提示”：

- 该提示被称为 **查询query** 
- 每个输入是一个 **值value** 和非自主性提示 **key** 的对，即key-value pair
- 通过注意力池化层来有偏向性的选择某些输入

![image-20250319125256346](./Attention%20Mechanisms.assets/image-20250319125256346.png)

## 1. 注意力汇聚

Average pooling 肯定是不行的，没有将key和value联系起来。

### 3.1 Nonparametric attention pooling

用Nadaraya-Watson核回归实现非参数的注意力汇聚：

![image-20250319125501985](./Attention%20Mechanisms.assets/image-20250319125501985.png)

- 给定数据 $(x_i,y_i)$ ，根据输入的位置对输出 $y_i$ 进行加权。 $K$ 是一个衡量距离的核函数。

受此启发，得到通用 attention pooling 公式：
$$
f(x) = \sum_{i=1}^{n}\alpha(x,x_i)y_i
$$

- attention pooling 是 $y_i$ 的加权平均
- 将查询 $x$ 与键 $x_i$ 之间的关系建模为 attention weight $\alpha(x,x_i)$ ，这个权重将被分配给每一个对应值 $y_i$
- 对于任何查询，模型在所有键值对注意力权重都是一个有效的概率分布： 它们是非负的，并且总和为1

如果使用高斯核：

![image-20250319165925497](./Attention%20Mechanisms.assets/image-20250319165925497.png)

一个键 $x_i$ 越是接近给定的查询 $x$ ， 那么分配给这个键对应值 $y_i$ 的注意力权重就会越大， 也就“获得了更多的注意力”。



### 1.2 Parametric attention pooling

将可学习的参数集成到注意力汇聚中，在查询 $x$ 和键 $x_i$ 之间的距离乘以可学习参数 $w$ ，再通过训练模型来学习参数。

与非参数的注意力汇聚模型相比， 带参数的模型加入可学习的参数后， 曲线在注意力权重较大的区域变得更不平滑。



## 2. 注意力分数

![image-20250319192030164](./Attention%20Mechanisms.assets/image-20250319192030164.png)

通过 Attention scoring function 实现对查询和键之间的关系建模(评估query和key的相似度)，将其结果输入到 softmax 中得到注意力权重. 最后，注意力汇聚的输出就是基于这些注意力权重的值的加权和。

![image-20250319193237838](./Attention%20Mechanisms.assets/image-20250319193237838.png)

$\alpha(q,k_i)$ :  将 $q$ 和 $k_i$ 两个向量输入注意力评分函数 $a$ , 将其映射成标量, 再经softmax运算得到注意力权重.



### 2.1 Additive Attention

当查询和键是 **不同长度的矢量** 时，可以使用加性注意力作为评分函数。 给定query $\bf{q}\in \mathbb{R}^q$, key $\bf{k} \in \mathbb{R}^k$ :

![image-20250319193926714](./Attention%20Mechanisms.assets/image-20250319193926714.png)

等价于将key和query合并起来后放到一个 **隐藏大小为 h**, **输出大小为1** 的单隐藏层MLP.



### 2.2 Scaled Dot-Product Attention

当查询和键是 **相同长度** 为d时, 可以使用缩放点积注意力. 

![image-20250319195021596](./Attention%20Mechanisms.assets/image-20250319195021596.png)

假设查询和键的所有元素都是独立的随机变量， 并且都满足零均值和单位方差， 那么两个向量的点积的均值为0，方差为d。 为确保无论向量长度如何， 点积的方差在不考虑向量长度的情况下仍然是1.

![image-20250319195121419](./Attention%20Mechanisms.assets/image-20250319195121419.png)



## 3. Bahdanau 注意力

![image-20250320111510510](./Attention%20Mechanisms.assets/image-20250320111510510.png)

在原始模型中, 上下文变量 $\bf{c}$ 用于表示 source sentence 的信息; 而在该架构中:

- $\bf{h_t}$ 表示编码器在时间步 t 的隐藏层信息, 且既表示key又表示value;
-  $s_{t'-1}$ 表示解码器在时间步 t'-1 的隐藏层信息, 用于表示 query;
- 在每个 decoding time step $t'$, $\bf{c}_{t'}$ 都会被更新. 假设input sequence 长度为 $T$:

$$
\bf{c}_{t'} = \sum_{t=1}^{T}\alpha(s_{t'-1},h_t)h_t
$$

- $\bf{c}_{t'}$ 将被输入解码器的循环层, 来生成解码器时间步 t' 的 state $s_{t'}$

#### 代码

```python
class Seq2SeqAttentionDecoder(AttentionDecoder):
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 dropout=0, **kwargs):
        super(Seq2SeqAttentionDecoder, self).__init__(**kwargs)
        #采用additive attention为评分函数
        self.attention = d2l.AdditiveAttention(
            num_hiddens, num_hiddens, num_hiddens, dropout)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(
            embed_size + num_hiddens, num_hiddens, num_layers,
            dropout=dropout)
        self.dense = nn.Linear(num_hiddens, vocab_size)

    def init_state(self, enc_outputs, enc_valid_lens, *args):
        # outputs的形状为(batch_size，num_steps，num_hiddens).
        # hidden_state的形状为(num_layers，batch_size，num_hiddens)
        outputs, hidden_state = enc_outputs
        return (outputs.permute(1, 0, 2), hidden_state, enc_valid_lens)

    def forward(self, X, state):
        # enc_outputs的形状为(batch_size,num_steps,num_hiddens).
        # hidden_state的形状为(num_layers,batch_size,
        # num_hiddens)
        enc_outputs, hidden_state, enc_valid_lens = state
        # 输出X的形状为(num_steps,batch_size,embed_size)
        X = self.embedding(X).permute(1, 0, 2)
        outputs, self._attention_weights = [], []
        for x in X:
            # query的形状为(batch_size,1,num_hiddens)
            query = torch.unsqueeze(hidden_state[-1], dim=1)
            # context的形状为(batch_size,1,num_hiddens)
            # enc_valid_lens是大小为batch_size的向量,每个值表示该样本句子有效长度,算有效长度内的注意力权重,大于长度的部分可以不要管
            context = self.attention(
                query, enc_outputs, enc_outputs, enc_valid_lens)
            # 在特征维度上连结
            x = torch.cat((context, torch.unsqueeze(x, dim=1)), dim=-1)
            # 将x变形为(1,batch_size,embed_size+num_hiddens)
            out, hidden_state = self.rnn(x.permute(1, 0, 2), hidden_state)
            outputs.append(out)
            self._attention_weights.append(self.attention.attention_weights)
        # 全连接层变换后，outputs的形状为
        # (num_steps,batch_size,vocab_size)
        outputs = self.dense(torch.cat(outputs, dim=0))
        return outputs.permute(1, 0, 2), [enc_outputs, hidden_state,
                                          enc_valid_lens]

    @property
    def attention_weights(self):
        return self._attention_weights
```

