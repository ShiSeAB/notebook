# Recurrent neural network

​	CNN用于解决空间性数据（图片），而RNN用于解决时间性数据（文本、视频）。

## 1. 序列模型

​	很多数据是有时序变化的，例如电影的评价随时间变化而变化。

​	在时间t观察到 $x_t$，那么得到 $T$ 个不独立的随机变量：
$$
(x_1,...x_T)\textasciitilde p(x)
$$
​	使用条件概率展开：上面是正序，下方为反序（但一般未来事件是基于前面发生事件推测，所以根据未来事件推前面事件有点不符合物理规律）

![image-20250311093147972](./Recurrent%20neural%20network.assets/image-20250311093147972.png)

### 1.1 自回归模型

​	对见过的数据建模，也称自回归模型(自己预测自己)，如下面公式中的$f$
$$
p(x_t|x_1,...x_{t-1}) = p(x_t|f(x_1,...x_{t-1}))
$$

#### A. 马尔科夫假设

假设当前数据只跟 $\tau$ 个过去数据点相关

![image-20250311094517554](./Recurrent%20neural%20network.assets/image-20250311094517554.png)

可将过去数据点看作一个 $\tau$ 长的向量，用其预测一个标量，可以用MLP做到。

#### B. 潜变量模型

引入潜变量 $h_t$ 来表示过去的信息 $h_t = f(x_1,...x_{t-1})$
$$
x_t = p(x_t|h_t)
$$
![image-20250311094843637](./Recurrent%20neural%20network.assets/image-20250311094843637.png)

潜变量也一直在更新。所以有两个模型，每个模型只与两个变量相关。

### 1.2 实现马尔科夫模型

```python
tau = 4
#T-4 个 4维的向量
features = torch.zeros((T - tau, tau))
for i in range(tau):
    features[:, i] = x[i: T - tau + i]
#标签 yi = xi
labels = x[tau:].reshape((-1, 1))

batch_size, n_train = 16, 600
# 只有前n_train个样本用于训练
train_iter = d2l.load_array((features[:n_train], labels[:n_train]),
                            batch_size, is_train=True)

# 初始化网络权重的函数
def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)

# 一个简单的多层感知机
def get_net():
    net = nn.Sequential(nn.Linear(4, 10),
                        nn.ReLU(),
                        nn.Linear(10, 1))
    net.apply(init_weights)
    return net

# 平方损失。注意：MSELoss计算平方误差时不带系数1/2
loss = nn.MSELoss(reduction='none')

def train(net, train_iter, loss, epochs, lr):
    trainer = torch.optim.Adam(net.parameters(), lr)
    for epoch in range(epochs):
        for X, y in train_iter:
            trainer.zero_grad()
            l = loss(net(X), y)
            l.sum().backward()
            trainer.step()
        print(f'epoch {epoch + 1}, '
              f'loss: {d2l.evaluate_loss(net, train_iter, loss):f}')

net = get_net()
train(net, train_iter, loss, 5, 0.01)

```

预测：

![image-20250311104959049](./Recurrent%20neural%20network.assets/image-20250311104959049.png)

one-step-ahead prediction 是给定 $\tau$ 个数据，预测下一个； k-step-ahead-prediction 是给定 $\tau$ 个数据，预测下k个数据，在预测过程中，我们会使用自己预测的数据作为输入，导致误差积累，从而不那么准确。

![image-20250311105418182](./Recurrent%20neural%20network.assets/image-20250311105418182.png)



## 2. 文本预处理

### 2.1 加载文本

​	将文本作为字符串加载到内存中，忽略标点符号和字母大写（下面代码较为简单、暴力）。

```python
#@save
d2l.DATA_HUB['time_machine'] = (d2l.DATA_URL + 'timemachine.txt',
                                '090b5e7e70c295757f55df93cb0a180b9691891a')

def read_time_machine():  #@save
    """将时间机器数据集加载到文本行的列表中"""
    with open(d2l.download('time_machine'), 'r') as f:
        lines = f.readlines()
    return [re.sub('[^A-Za-z]+', ' ', line).strip().lower() for line in lines]

lines = read_time_machine()
print(f'# 文本总行数: {len(lines)}')
print(lines[0])
print(lines[10])
```

### 2.2 词元化

下面的`tokenize`函数将文本行列表（`lines`）作为输入， 列表中的每个元素是一个文本序列（如一条文本行）。 每个文本序列又被拆分成一个**词元列表**，*词元*（token）是文本的基本单位。 最后，返回一个由**词元列表组成的列表**，其中的每个词元都是一个字符串（string）

不过，中文分词是一个大工程，比英文分词更难。

```python
def tokenize(lines, token='word'):  #@save
    """将文本行拆分为单词或字符词元"""
    if token == 'word':
        return [line.split() for line in lines]
    elif token == 'char':
        return [list(line) for line in lines]
    else:
        print('错误：未知词元类型：' + token)

tokens = tokenize(lines)
for i in range(11):
    print(tokens[i])
```

### 2.3 词表

构建一个字典，通常也叫做*词表*（vocabulary）， 用来将string类型的词元映射到从0开始的数字索引中。

我们先将训练集中的所有文档合并在一起，对它们的唯一词元进行统计， 得到的统计结果称之为*语料*（corpus）。 然后根据每个唯一词元的出现频率，为其分配一个数字索引。 很少出现的词元通常被移除，这可以降低复杂性。

语料库中不存在或已删除的任何词元都将映射到一个特定的未知词元“<unk>”。 我们可以选择增加一个列表 `reserved_tokens`，用于保存那些被保留的词元， 例如：填充词元（“<pad>”）； 序列开始词元（“<bos>”）； 序列结束词元（“<eos>”）。

```python
class Vocab:  #@save
    """文本词表"""
    def __init__(self, tokens=None, min_freq=0, reserved_tokens=None):
        if tokens is None:
            tokens = []
        if reserved_tokens is None:
            reserved_tokens = []
        # 按出现频率排序
        counter = count_corpus(tokens)
        self.token_freqs = sorted(counter.items(), key=lambda x: x[1],
                                   reverse=True)
        # 未知词元的索引为0
        self.unk,uniq_tokens = 0, ['<unk>'] + reserved_tokens
        uniq_tokens += [
            token for token, freq in self.token_freqs
            if freq >= min_freq and token not in uniq_tokens
        ]
        #索引与token建立联系
        self.idx_to_token = []
        self.token_to_idx = dict()
        for token in uniq_tokens:
            self.idx_to_token.append(token)
            self.token_to_idx[token] = len(self.idx_to_token)

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]

    def to_tokens(self, indices):
        if not isinstance(indices, (list, tuple)):
            return self.idx_to_token[indices]
        return [self.idx_to_token[index] for index in indices]

    @property
    def unk(self):  # 未知词元的索引为0
        return 0

    @property
    def token_freqs(self):
        return self._token_freqs

def count_corpus(tokens):  #@save
    """统计词元的频率"""
    # 这里的tokens是1D列表或2D列表
    if len(tokens) == 0 or isinstance(tokens[0], list):
        # 将词元列表展平成一个列表
        tokens = [token for line in tokens for token in line]
    return collections.Counter(tokens)
```

现在，我们可以将每一条文本行转换成一个数字索引列表。

```python
for i in [0, 10]:
    print('文本:', tokens[i])
    print('索引:', vocab[tokens[i]])
```

```
文本: ['the', 'time', 'machine', 'by', 'h', 'g', 'wells']
索引: [1, 19, 50, 40, 2183, 2184, 400]
文本: ['twinkled', 'and', 'his', 'usually', 'pale', 'face', 'was', 'flushed', 'and', 'animated', 'the']
索引: [2186, 3, 25, 1044, 362, 113, 7, 1421, 3, 1045, 1]
```

最后，将上述所有功能整合到一起。

```python
def load_corpus_time_machine(max_tokens=-1):  #@save
    """返回时光机器数据集的词元索引列表和词表"""
    lines = read_time_machine()
    tokens = tokenize(lines, 'char')
    vocab = Vocab(tokens)
    # 因为时光机器数据集中的每个文本行不一定是一个句子或一个段落，
    # 所以将所有文本行展平到一个列表中
    corpus = [vocab[token] for line in tokens for token in line]
    if max_tokens > 0:
        corpus = corpus[:max_tokens]
    return corpus, vocab

corpus, vocab = load_corpus_time_machine()
len(corpus), len(vocab)
#（170580，28）
```

## 3. 语言模型

