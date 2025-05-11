# Translating into Intermediate Representation

重点是 translation

## 1. 中间表示概述

![image-20250427101150748](./ch7-IR.assets/image-20250427101150748.png)

- 前端：从源码到IR生成
- 中端：基于IR的分析与变换（可能生成新IR）
- 后端：(机器相关优化)；从IR 到目标代码

### 1.1 为什么需要中间表示(IR)

yps：这个原因肯定会考

![image-20250427101302948](./ch7-IR.assets/image-20250427101302948.png)

直接翻译成机器码的危害：

- hinders **modularity**
- hinders **portability**

### 1.2 IR 分类（不要求掌握）

根据抽象层次( 实际编译器可能采用多层IR )：

![image-20250427101654291](./ch7-IR.assets/image-20250427101654291.png)

- 高层中间表示 High-level IR: 贴近输入语言，方便由前端生成

  ![image-20250427101804720](./ch7-IR.assets/image-20250427101804720.png)

- 低层中间表示 Low-level IR: 贴近目标语言，方面目标代码生成

- 中层中间表示 Middle-Level IR

根据结构特征：

- 结构化表示 Structural

  - Graphically oriented (e.g., tree, DAG,...)
  - Heavily used in source-to-source translators

  ![image-20250427102033892](./ch7-IR.assets/image-20250427102033892.png)

- 线性表示 Linear：存储布局是线性的

  ![image-20250427102251442](./ch7-IR.assets/image-20250427102251442.png)

- 混合表示 Hybrid：Combination of graphs and linear code

  ![image-20250427102302905](./ch7-IR.assets/image-20250427102302905.png)

### 1.3 三地址码(Three-Address Code)

一般形式：$x = y ~op ~z$

- 每个指令最多1个算符，最多3个操作数(三地址)

- 例：

  ![image-20250427102546311](./ch7-IR.assets/image-20250427102546311.png)

![image-20250427102824906](./ch7-IR.assets/image-20250427102824906.png)

实现：

- The entire sequence of three-address instructions is implemented as **an array of linked list**

- implement three-address code as **quadruples**（四元组）

  - one field for the operation
  - three fields for the addresses

- 对于 fewer than three addresses 的指令，一个或多个地址字段被赋予 null 或“empty”值。

  ![image-20250427103234427](./ch7-IR.assets/image-20250427103234427.png)

- Other implementation: **triples, indirect triples**

Static Single Assignment (SSA) 不考

- 特殊的三地址代码，其所有变量在代码中只被赋值一次

  ![image-20250427103544656](./ch7-IR.assets/image-20250427103544656.png)

- 方便了编译器中的很多分析和优化

  - 查询def-use （定义变量使用情况）信息(某些分析的子过程)
  - 加速现有算法(基于SSA的稀疏分析)
  - 严格依赖SSA的算法(ssapre, new gvn,…)
  - “免费”提升精度（如流敏感指针分析)




## 2. IR Tree 

虎书只用一层 IR，即 **IR Tree**，介于 AST 和 assembly 之间

![image-20250427104705556](./ch7-IR.assets/image-20250427104705556.png)

- $e$ 表示 expressio
- $s$ 表示 statement
- $o$ 表示 operator
- $i$ 表示 integer
- $f$ 表示 function
- $t$ 表示 temporary

### The Expressions

![image-20250427104733996](./ch7-IR.assets/image-20250427104733996.png)

![image-20250427104816888](./ch7-IR.assets/image-20250427104816888.png)

ESEQ(s, e)：The statement s is evaluated for side effects, then e is evaluated for a result.

- 假设s是statement a=5, e是expression a+5
- Statement (如a=5)不返回值,但是有副作用
- 表达式ESEQ(a=5, a + 5)最终的结果是10

关于副作用(Side effects)

- 副作用意味着更新存储单元 memory cell 或临时寄存器 temporary register 的内容

### The Statements

![image-20250427104906988](./ch7-IR.assets/image-20250427104906988.png)

![image-20250427104920619](./ch7-IR.assets/image-20250427104920619.png)



## 3. IR Tree 的生成

![image-20250427105303407](./ch7-IR.assets/image-20250427105303407.png)

### 3.1 Translation of expressions

Mapping AST Expressions to IR Tree

- Ex: AST Expressions with return values (e.g., a + b)
- Nx: AST Expressions that return no value (e.g., print(x))
- Cx: AST Expressions with Boolean values (conditional jump)

![image-20250427105613785](./ch7-IR.assets/image-20250427105613785.png)

![image-20250427110005361](./ch7-IR.assets/image-20250427110005361.png)

![image-20250427110014965](./ch7-IR.assets/image-20250427110014965.png)

问题：语言结构通常需要在期望不同形式的上下文中使用，所以需要转换形式：

![image-20250427110144939](./ch7-IR.assets/image-20250427110144939.png)

#### 3.1.1 Simple Variables

对于在当前过程的堆栈帧中声明的变量 v ，frame point 为 fp，获取在frame中偏移量为 k 的 v：
$$
MEM(BINOP(PLUS,TEMP~fp,CONST~k))
$$

- 在 tiger 中，所有变量size都相同 -- word size

![image-20250504203616146](./ch7-IR.assets/image-20250504203616146.png)

比较复杂的是 nest 时访问上层的 variable **v** through static links：

![image-20250504223701490](./ch7-IR.assets/image-20250504223701490.png)



#### 3.1.2 Array Variables

Tiger 数组赋值：不 copy 内容，而是指针

![image-20250504204019126](./ch7-IR.assets/image-20250504204019126.png)

- 不过 Pascal 语言是 copy内容
- C 语言中上述赋值不合法



#### 3.1.3 Structured L-values

R-value:

- does not denote an assignable location
- 可计算但不可被赋值

L-value：

- Denotes a **location** that can be assigned to
- 可出现在赋值语句右侧（表示 location 中的内容）

左值又分 scalar 和 structured：

- Scalar L-value: An integer or pointer value（Tiger中所有变量和左值都是 scalar 的，其array和record都是pointer）

- Structured L-value: C or Pascal has structured L-values（就是结构体）：

  ![image-20250504205437935](./ch7-IR.assets/image-20250504205437935.png)

两者在 memory 中的表示：

![image-20250504205540378](./ch7-IR.assets/image-20250504205540378.png)

所以取 Structured L-value 时还需加上 object 在结构体内的偏移量 **S**：

![image-20250504205706536](./ch7-IR.assets/image-20250504205706536.png)

#### 3.1.4 Subscripting and Field Selection

![image-20250504205800927](./ch7-IR.assets/image-20250504205800927.png)

- MEM(e) 表示数组基地址 ‘a’

![image-20250504210328587](./ch7-IR.assets/image-20250504210328587.png)

#### 3.1.5 Arithmetic

- Straightforward
- 无 unary operator
- 无 浮点数

#### 3.1.6 Conditionals

即Cx

![image-20250504210503459](./ch7-IR.assets/image-20250504210503459.png)

其中 And 和 Or 操作需要实现短路求值：

![image-20250504210544405](./ch7-IR.assets/image-20250504210544405.png)

![image-20250504210557407](./ch7-IR.assets/image-20250504210557407.png)

#### 3.1.7 While Loops

![image-20250504210634469](./ch7-IR.assets/image-20250504210634469.png)

![image-20250504210649418](./ch7-IR.assets/image-20250504210649418.png)

#### For Loops

![image-20250504224237708](./ch7-IR.assets/image-20250504224237708.png)

#### Function Calls

$call~f(a_1,...a_n)$​ 翻译很简单，但是 static link 需要作为 implicit extra argument

![image-20250504224521168](./ch7-IR.assets/image-20250504224521168.png)

### Translation of declarations

**变量声明**，在栈帧中的偏移量一定要清楚。

![image-20250504224751500](./ch7-IR.assets/image-20250504224751500.png)

类型声明无需生成 IR code

**函数声明** ，中间代码包括三个部分：

- Prologue：
  - 声明函数开始的指令
  - 函数名 label 定义
  - 分配栈帧的指令
  - store 指令用于保存 callee-save registers and return address
  - store 指令用于保存 arguments and static links
- Body
- Epilogue
  - move return result to a special register 的指令
  - reset the stack pointer 的指令
  - return 指令
  - 声明函数结束的指令
