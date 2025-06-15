# Lexical Analysis

## 1. 词法分析概述

程序以 **字符串** 的形式传递给编译器，所以词法分析的任务是将输入字符串识别为 **有意义的子串** 

- partition input strings into substrings(==Lexeme==)
- classify them according to their roles(==Tokens==)

 **Token** 是关键字、操作符、标识符、字符串等，lexeme是token的实例：

![image-20250220151437295](./Lexical%20Analysis.assets/image-20250220151437295.png)



!!! note  "词法分析例：字符流 -> Token流![image-20250303145035164](./Lexical%20Analysis.assets/image-20250303145035164.png)"



## 2. 正则表达式 -- 形式化描述词法

### 2.1 形式语言

- **字母表**(alphabet)：符号的有限集合
- **串** (String, word) : 字母表中符号的 **有穷序列**
  - 串s的长度，通常记作|s|，是指s中符号的个数
  - 空串是长度为0的串，用 ε（epsilon）表示



!!! warning "$\varepsilon$ 表示空串，{$\varepsilon$}是一个非空集合，里面的元素是一个空串，$\emptyset$是一个空集



串上的连接和运算：

- **连接**(concatenation): y附加到x后形成的串记作xy
  -  例如，如果 x=dog且 y=house，那么xy=doghouse
  -  空串是连接运算的单位元, 即对于任何串s都有εs = sε = s
- **幂运算**

!!! note  "公式如下![image-20250303103325839](./Lexical%20Analysis.assets/image-20250303103325839.png)"

 **语言** 是字母表上的一个串集，有如下运算，其中优先级 **幂>连接>并** ：

![image-20250303103535621](./Lexical%20Analysis.assets/image-20250303103535621.png)

### 2.2 正则表达式RE

![image-20250303104221253](./Lexical%20Analysis.assets/image-20250303104221253.png)

优先级：* >连接>选择|

RE的一些代数定律：

![image-20250303104411898](./Lexical%20Analysis.assets/image-20250303104411898.png)



!!! note "example![image-20250303104518839](./Lexical%20Analysis.assets/image-20250303104518839.png)"



#### 正则定义

对于比较复杂的语言，为了构造简洁的正则式，可先构造简单的正则式，再将这些正则式组合起来，形成一个与该语言匹配的正则序列

![image-20250303105801879](./Lexical%20Analysis.assets/image-20250303105801879.png)

例：

![image-20250303105904079](./Lexical%20Analysis.assets/image-20250303105904079.png)

#### 正则规则的“二义性”

根据不同原则判断：

- 最长匹配 Longest match：'if8'会被识别成identifier
- 规则优先 Rule priority：‘if8’ 中的if match IF



## 3. 有穷自动机 FA

其实RE和FA都是计算理论的内容，这里简单介绍

### 3.1 Finite Automata

![image-20250303110537520](./Lexical%20Analysis.assets/image-20250303110537520.png)

![image-20250303110937645](./Lexical%20Analysis.assets/image-20250303110937645.png)

- 给定输入串$x$，如果存在**一个**对应于串x的从初始状态到某个终止状态的转换序列，则称串$x$被该$FA$接收；

- 由一个有穷自动机$M$接收的所有串构成的集合，称为**该$FA$接收（或定义）的语言**，记为$L(M)$

  ![image-20250303111214921](./Lexical%20Analysis.assets/image-20250303111214921.png)

### 3.2 FA的分类

#### 3.2.1 NFA 非确定有穷自动机

![image-20250303111421630](./Lexical%20Analysis.assets/image-20250303111421630.png)

#### 3.2.2 DFA 确定性有穷自动机

![image-20250303111451377](./Lexical%20Analysis.assets/image-20250303111451377.png)

#### 3.2.3 两者关系

DFA与NFA可以互相转化（定义同一语言）。



### 3.3 FA与词法分析

词法分析: 如何**自动化**构造FA, 来识别用RE刻画的Token，

因为NFA需对多种路径试探+失败回退，效率很低，所以一般是构造DFA：

![image-20250303112347975](./Lexical%20Analysis.assets/image-20250303112347975-1740972228944-2.png)

## 4.词法分析器的自动生成

给定RE，如何自动构造其DFA？以下是一个算法用于自动生成，但比较复杂，可以不完全按照该流程。

![image-20250303112842234](./Lexical%20Analysis.assets/image-20250303112842234.png)

### 4.1 RE -> NFA

采用Thompson算法，基于对RE的结构做归纳。

#### 直接构造

![image-20250303113838228](./Lexical%20Analysis.assets/image-20250303113838228.png)

#### 递归构造

![image-20250303113920610](./Lexical%20Analysis.assets/image-20250303113920610.png)

![image-20250303113931019](./Lexical%20Analysis.assets/image-20250303113931019.png)

![image-20250303113946145](./Lexical%20Analysis.assets/image-20250303113946145.png)

![image-20250303114002952](./Lexical%20Analysis.assets/image-20250303114002952.png)

但如果人工构造的话会非常简单：

![image-20250303123141336](./Lexical%20Analysis.assets/image-20250303123141336.png)

### 4.2 NFA -> DFA

采用 **子集构造法** (subset construction)

- DFA的每个状态是NFA的状态集合的一个**子集**
- 读了输入 $a_i$ 后NFA能到达的所有状态：$s_1, s_2, …,s_k$，
  则DFA到达一个状态，对应于NFA的{$s_1, s_2, …,s_k$}.

![image-20250303133047019](./Lexical%20Analysis.assets/image-20250303133047019.png)

过程：

![image-20250303133306211](./Lexical%20Analysis.assets/image-20250303133306211.png)

NFA -> DFA示例：

![image-20250303133434106](./Lexical%20Analysis.assets/image-20250303133434106.png)首先计算 状态0 的$\varepsilon$-closure，记为$A$

![image-20250303133734883](./Lexical%20Analysis.assets/image-20250303133734883.png)

接着计算 $\varepsilon - closure(move(A,a))$，记为$B$. $move(A,a)=\{3,8\}$ 

![image-20250303133938578](./Lexical%20Analysis.assets/image-20250303133938578.png)

再计算 $\varepsilon - closure(move(A,b))$，记为$C$

![image-20250303134141001](./Lexical%20Analysis.assets/image-20250303134141001.png)

接着计算 $\varepsilon - closure(move(B,a))$，发现该集合等于 $B$

![image-20250303134604980](./Lexical%20Analysis.assets/image-20250303134604980.png)

$\varepsilon - closure(move(B,b)) = \varepsilon-closure(5,9)$ 记为D

![image-20250303134646718](./Lexical%20Analysis.assets/image-20250303134646718.png)

计算完A、B的状态转换，计算C、D的，D是该DFA的final state：

![image-20250303135004478](./Lexical%20Analysis.assets/image-20250303135004478.png)

![image-20250303135118183](./Lexical%20Analysis.assets/image-20250303135118183.png)



### 4.3 DFA简化

一个正则语言可对应于多个识别此语言的DFA，我们需要找到最简单的。

#### Distinguishable States

如果存在串 $x$，使得从 $s、t$ 出发，一个到达**接受状态**，一个到达**非接受状态**，那么string $x$ 就区分了状态 $s$ 和 $t$.

![image-20250303135829194](./Lexical%20Analysis.assets/image-20250303135829194.png)

#### DFA简化算法(Hopcroft 算法)

- 划分部分

  - 初始划分: 非接受状态组和接受状态组 $P = \{S-F,F\}$

  - 反复分裂划分

    ![image-20250303140614330](./Lexical%20Analysis.assets/image-20250303140614330.png)

  - 直到不可再分裂，

    ```
    if P_new == P
    	P_final = P;
    else
    	P = P_new;
    	go to step 2;
    ```

    

  例：

  ![image-20250303140640751](./Lexical%20Analysis.assets/image-20250303140640751.png)

  

- 构造部分
在$P_{final}$的每个组中选择一个状态作代表，作为最小化DFA中的状态
  
- 其中包含原开始状态的组选出的状态为新开始状态
  - 原接受状态组中必须只包含接受状态，选择出的状态为新接受状态
  
例：
  
![image-20250303141835263](./Lexical%20Analysis.assets/image-20250303141835263.png)
  
