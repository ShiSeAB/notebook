# Buttom-Up Parsing

该部分主要介绍Bottom-Up方法：

![image-20250317135852620](./Parsing%20-%202.assets/image-20250317135852620.png)

文法：$LR(0),SLR(1),LR(1),LALR(1)$

表达力排序：

![image-20250317141921774](./Parsing%20-%202.assets/image-20250317141921774.png)



## 1. Shift-Reduce

Bottom-Up Parsing: 从串w规约为文法开始符号S

- 规约: 与某产生式 **右部** RHS 相匹配的特定子串, 被替换为该产生式 **头部** LHS 的非终结符
- 但问题是：什么时候规约（读入多少token后规约）？规约到哪个非终结符？

LR 分析的一般模式：shift-reduce

例：我们发现光标移动Shift过程中什么时候规约Reduce是很重要的（做更重要的决策），且树是从下面往上建的

![image-20250306151428433](./Parsing%20-%202.assets/image-20250306151428433.png)

构建这棵树的过程是最右推导的逆过程(Rightmost derivation in reverse)，**最右推导** 过程为：

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

状态+跳转 -> 自动机，这种相应的有穷自动机就是LR(0)自动机：

- 文法产生式数量是有限的
- 每个产生式右部的长度也是有限的
- 故称为有穷

### 2.1 LR(0) Parsing NFA

​	该 NFA 不是指直接用来识别 LR(0) 语言的自动机（ NFA 只能识别正则语言，然而 LR(0) 是上下文无关语言）。该NFA是用来“记录当前识别进度”的（帮助判断栈顶内容是否可归约了）。

**起始&终结状态**

增加新开始符号 $S'$ 并加入产生式 $S'\rightarrow S\$$ 

![image-20250317144135653](./Parsing%20-%202.assets/image-20250317144135653.png)

**状态迁移关系**

![image-20250317144245797](./Parsing%20-%202.assets/image-20250317144245797.png)

![image-20250317144300750](./Parsing%20-%202.assets/image-20250317144300750.png)

**例**

![image-20250317144616309](./Parsing%20-%202.assets/image-20250317144616309.png)



### 2.2 LR(0) Parsing DFA

#### 2.2.1 NFA ->> DFA

利用子集构造法：

![image-20250317144955067](./Parsing%20-%202.assets/image-20250317144955067.png)



#### 2.2.2 直接构造

首先是项集构造，DFA起始状态为 $S'\rightarrow ·S\$$ ，根据下述算法扩充项集，即将以 S 为 LHS  的项加进去：

![image-20250317145543090](./Parsing%20-%202.assets/image-20250317145543090.png)

接着用GOTO算法进行状态转移，其中 $Goto(I,X)$ 定义为 $I$ 中所有形如 $A \rightarrow \alpha·X\beta$ 的项所对应的项 $A\rightarrow \alpha X·\beta$ 的集合的闭包，$X$ 为input：

![image-20250317154519081](./Parsing%20-%202.assets/image-20250317154519081.png)

用Closure补全项集：

![image-20250317161903463](./Parsing%20-%202.assets/image-20250317161903463.png)

收敛条件：

![image-20250317161929831](./Parsing%20-%202.assets/image-20250317161929831.png)



不过考试一般不考LR(0)，因为比较简单。



### 2.3 LR(0) Parsing 分析表

从LR(0) Parsing DFA到语法分析表：

![image-20250317234129112](./Parsing%20-%202.assets/image-20250317234129112.png)

- $s2$ 代表 **shift** 并go to state 2，$T[1,x] = s2$ 代表在状态1读入x，就shift并跳转到状态2，此时一般dot不在end.
-  dot at the end时，就要 **reduce** 了，例如 $T[3,x] = r2$ ，代表用生成式2 $S\rightarrow y$ 进行规约。
- **accept** 说明接受输入串，一般是在final state。
- 在 **Goto** 表中，输入是 non-terminal，$T[1,S]=g4$ 代表输入S，goto 状态4，而在Action表中输入是terminal



### 2.4 LR(0) Parsing 过程

![image-20250318110818445](./Parsing%20-%202.assets/image-20250318110818445.png)

![image-20250318110829054](./Parsing%20-%202.assets/image-20250318110829054.png)

算法：

![image-20250318111227476](./Parsing%20-%202.assets/image-20250318111227476.png)

例：

起始在状态1；输入x后到状态2；再输入x还是状态2；接着输入y跳转到状态3；由于下一个输入就是 $ ，所以reduce，|y|=1，pop状态3，将Goto[2,S]=5入栈；在状态5下reduce，|xS| = 2, pop状态5、2，将Goto[2,S] = 5入栈；

![image-20250318111453893](./Parsing%20-%202.assets/image-20250318111453893.png)

考点：Grammar的DFA长什么样、预测表长什么样，给一个input string，状态栈的变化。

由于LR(0)分析器不会查看下一个输入符号，所以没有足够的上下文来做出更正确(或者说更”聪明”)的决定。





## 3. SLR分析

SLR(1) 中的 S 表示 Simple。SLR(1) Parsing 在 LR(0) 的基础上通过简单的判断尝试解决冲突。

### 3.1 SLR(1) Parsing思路

SLR(1) 在生成 LR(0) DFA 后，计算每一个 non-terminal 的 Follow Set，并根据这两者创建 SLR(1) 分析表。

- 每步规约都应满足 $t\in Follow(E)$，只有当下一个输入 token $t\in Follow(E) = \{\$\}$ ，才能进行规约，所以在状态3时输入 $x,+$ 无法规约，同理状态5，输入是 $\{\$,+\}$ 时可规约 。


![image-20250318220842750](./Parsing%20-%202.assets/image-20250318220842750.png)

只有当输入为 $ 时，状态3才能进行规约；同理状态6. 状态5输入可以是{+，$}.

如果这样构造出的 SLR(1) Parsing Table 没有含冲突的表项，那么称这个文法为 SLR(1) Grammar，否则不是。



### 3.2 问题与局限

SLR(1) 解决了LR(0) 的shift or reduce 和用哪个production 规约的问题。但Follow集仍是 “合理但可能不够精确的近似”。

![image-20250318222513256](./Parsing%20-%202.assets/image-20250318222513256.png)

什么是最右句型？-- 最右推导中出现的句型



## 4. LR(1) 分析

包含更多信息，精确指明何时应该规约。

LR(1)项的形式： $A\rightarrow \alpha ·\beta,a$

- $a$ 称为向前看符号(lookahead symbol)，可以是terminal或$

闭包计算：

![image-20250319101629437](./Parsing%20-%202.assets/image-20250319101629437.png)

![image-20250319102237226](./Parsing%20-%202.assets/image-20250319102237226.png)

- $w$ 是terminal

例-计算该Grammar的DFA状态1：

![image-20250319102520857](./Parsing%20-%202.assets/image-20250319102520857.png)

![image-20250319102533040](./Parsing%20-%202.assets/image-20250319102533040.png)

![image-20250319102544649](./Parsing%20-%202.assets/image-20250319102544649.png)

![image-20250319102639185](./Parsing%20-%202.assets/image-20250319102639185.png)

![image-20250319102707310](./Parsing%20-%202.assets/image-20250319102707310.png)





Goto计算：比较简单

![image-20250319102737070](./Parsing%20-%202.assets/image-20250319102737070.png)

Reduce Action：

![image-20250319102822080](./Parsing%20-%202.assets/image-20250319102822080.png)

局限: LR(1) 的parsing table 会非常大,状态很多.



## 5. LALR(1)

在LR(1) 中,有的状态只有 look ahead symbol不同,由此可以考虑合并,从而减少状态数:

![image-20250320134453773](./Parsing%20-%202.assets/image-20250320134453773.png)

- 定义 **core** : core of a set of LR items is the set of first components(不带look ahead symbol)

- core 相同的状态可以合并. 合并后的状态称为LALR(1) states

- LALR(1) Parsing DFA 就是由LR(1) DFA合并直到所有states的core都不同

- 合并过程中,边也要相应改变

  ![image-20250320135229104](./Parsing%20-%202.assets/image-20250320135229104.png)

例:

![image-20250320134923816](./Parsing%20-%202.assets/image-20250320134923816.png)

### Parsing Generator

Yacc 是基于 LALR(1), 用BNF形式书写的语法分析器的生成器, 与Lex的联系为:

![image-20250320135900682](./Parsing%20-%202.assets/image-20250320135900682.png)

- 消除二义性
- 冲突解决



## 6. 小结

![image-20250320135959261](./Parsing%20-%202.assets/image-20250320135959261.png)

![image-20250320140236432](./Parsing%20-%202.assets/image-20250320140236432.png)

- LR(0) SLR 都不会显式地考虑look ahead symbol,项集比较简单

![image-20250415115943241](./Parsing%20-%202.assets/image-20250415115943241.png)
