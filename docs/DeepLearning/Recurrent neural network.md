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

- 概念：给定文本序列 $x_1,...,x_T$，语言模型的目标是估计**联合概率** $p(x_1,..,x_T)$.
  - 预训练模型
  - 生成文本。给定前面的词，不断使用 $x_t \textasciitilde P(x_t|x_{t-1},...x_1)$，一个理想的语言模型就能够基于模型本身生成自然文本。
  - 判断多个语言序列中哪个更常见。例如“人咬狗”和“狗咬人”，显然是后者更常见

### 3.1 建模

![image-20250312195930974](./Recurrent%20neural%20network.assets/image-20250312195930974.png)

- n是总词数corpus(token的数量)，$n(x),n(x,x')$ 是单个单词和连续单词的出现次数

![image-20250312200234490](./Recurrent%20neural%20network.assets/image-20250312200234490.png)

#### N元语法

​	当文本序列很长时，如果文本量不够大，那么文本序列出现次数很可能小于1.为了解决这个问题。利用马尔可夫假设，其中N就是 $\tau$ ：

![image-20250312200640649](./Recurrent%20neural%20network.assets/image-20250312200640649.png)

N越大，需要存的信息就越多：每个长为N的序列的概率都要被存下来。但一般一元语法不可行，因为会大大高估 *停用词* 的频率。

```python
import random
import torch
from d2l import torch as d2l

tokens = d2l.tokenize(d2l.read_time_machine())
# 因为每个文本行不一定是一个句子或一个段落，因此我们把所有文本行拼接到一起
corpus = [token for line in tokens for token in line]
vocab = d2l.Vocab(corpus)
vocab.token_freqs[:10]
#freq是一元语法中的词元频率，bigram_freq二元，trigram_freqs三元
bigram_freqs = [freq for token, freq in bigram_vocab.token_freqs]
trigram_freqs = [freq for token, freq in trigram_vocab.token_freqs]
d2l.plot([freqs, bigram_freqs, trigram_freqs], xlabel='token: x',
         ylabel='frequency: n(x)', xscale='log', yscale='log',
         legend=['unigram', 'bigram', 'trigram'])
```

发现单词序列遵循齐普夫定律，指数大小受序列长度影响：

![image-20250312202448950](./Recurrent%20neural%20network.assets/image-20250312202448950.png)

### 3.2 读取长序列数据

​	给定一个长序列，随机抽取其中一个长为 $\tau$ 的文本序列作为输入X（称为**随机采样**），预测下一个词元，因此输出标签Y是移位了一个词元的原始序列。为避免一个文本数据被使用多次，将长序列划分为相同长度的子序列，作为小批量被输入到模型中。设置随机偏移量k，从长序列的词元k处开始划分。

```python
#num_steps相当于tau
def seq_data_iter_random(corpus, batch_size, num_steps):  #@save
    """使用随机抽样生成一个小批量子序列"""
    # 从随机偏移量randint(0, num_steps - 1)开始对序列进行分区，随机范围包括num_steps-1
    corpus = corpus[random.randint(0, num_steps - 1):]
    # 减去1，是因为我们需要考虑标签，子序列数量
    num_subseqs = (len(corpus) - 1) // num_steps
    # 长度为num_steps的子序列的起始索引，即在长序列中的位置，每次跳num_steps个token
    initial_indices = list(range(0, num_subseqs * num_steps, num_steps))
    # 打乱子序列顺序
    # 在随机抽样的迭代过程中，
    # 来自两个相邻的、随机的、小批量中的子序列不一定在原始序列上相邻
    random.shuffle(initial_indices)

    def data(pos):
        # 返回从pos位置开始的长度为num_steps的序列
        return corpus[pos: pos + num_steps]

    num_batches = num_subseqs // batch_size
    for i in range(0, batch_size * num_batches, batch_size):
        # 在这里，initial_indices包含子序列的随机起始索引
        initial_indices_per_batch = initial_indices[i: i + batch_size]
        X = [data(j) for j in initial_indices_per_batch]
        Y = [data(j + 1) for j in initial_indices_per_batch]
        yield torch.tensor(X), torch.tensor(Y)
```

​	在迭代过程中，除了对原始序列可以随机抽样外， 我们还可以保证两个相邻的小批量中的子序列在原始序列上也是相邻的。称之为 **顺序分区** 。效果如下：

![image-20250312214124653](./Recurrent%20neural%20network.assets/image-20250312214124653.png)

​	我们可以看到每个批量中的两个子序列分别与其它批量中对应子序列相邻。



## 4. RNN

​	利用隐变量，输入“你”，隐变量应预测到“好”，接着输入“好”，隐变量预测到“，”……

![image-20250312220344986](./Recurrent%20neural%20network.assets/image-20250312220344986.png) 

RNN模型为：

![image-20250312220649557](./Recurrent%20neural%20network.assets/image-20250312220649557.png)

$W_{hh}是h_{t-1}$ 的权重，拥有一定时序的预测目的；$W_{hx}$ 是 $x_{t-1}$ 的权重，$b_h$ 是bias。

### 4.1 Perplexity 困惑度

我们可以通过一个序列中所有的n个词元的交叉熵损失的平均值来衡量模型的质量：

![image-20250312222450418](./Recurrent%20neural%20network.assets/image-20250312222450418-1741789491849-1.png)

其中 P 由语言模型给出，$x_t$ 是时间步 t 从该序列中观察到的实际词元。

不过困惑度多取了一个指数：

![image-20250312223903117](./Recurrent%20neural%20network.assets/image-20250312223903117.png)

困惑度的最好的理解是“下一个词元的实际选择数的调和平均数”，最好情况下为1，完美预测；最坏情况下无穷大。

### 4.2 梯度裁剪

​	$g$ 表示一个存放所有层梯度的向量，如果 $g$ 的模超过 $\theta$，那么就将其“拉回来”：

![image-20250312225532105](./Recurrent%20neural%20network.assets/image-20250312225532105.png)

​	这一策略用于解决梯度过大的问题。



