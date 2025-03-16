# Parsing - 2

该部分主要介绍Bottom-Up方法，不断凑出产生式的RHS。

## 1. Shift-Reduce

LR 分析的一般模式：shift-reduce

![image-20250306151428433](./Parsing%20-%202.assets/image-20250306151428433.png)

构建这棵树的过程是最右推导的逆过程(Rightmost derivation in reverse)，最右推导为：

$E \Rightarrow E+(E) \Rightarrow E+(int) \Rightarrow E+(E)+(int) \Rightarrow E+(int)+(int) \Rightarrow int+(int)+(int)$

Shift-Reduce是基于栈的：

![image-20250313134035295](./Parsing%20-%202.assets/image-20250313134035295.png)

- Symbol Stack：存 left-substring $\alpha$ (terminal or nonterminal)
- Input Stream：存剩余输入 $\beta$
- **Shift** 意为 push next input(terminal) on to top of stack
- **Reduce** 规则如下：
  - 栈顶应match RHS of rule ($X -> AB$，栈顶字符应match AB)
  - 将栈顶match的RHS pop出来($pop~BA$)
  - 接着push the LHS onto the Stack($push~X$)

## 2. LR(0)分析

假设下次用到的 rule 为 $X\rightarrow \alpha \beta$，使用它进行规约前，栈顶可能什么都没有，也可能只有一个 $\alpha$，又可能match了RHS，这是不确定的，对应操作也应不同。

所以维护一个状态，用于记录当前识别的进度，以便知道栈顶内容是否可以进行规约了。

**项** 类似于Automata中的状态：

![image-20250313135552543](./Parsing%20-%202.assets/image-20250313135552543.png)

![image-20250313135908199](./Parsing%20-%202.assets/image-20250313135908199.png)

状态跳转：

![image-20250313135719256](./Parsing%20-%202.assets/image-20250313135719256.png)

状态+跳转 -> 自动机，这种相应的有穷自动机就是LR(0)自动机。

### 2.1 LR(0) Parsing NFA



### 2.2 LR(0) Parsing DFA





## 3. SLR分析





