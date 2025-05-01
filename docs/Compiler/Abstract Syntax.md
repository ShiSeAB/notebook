# ch4 -Abstract Syntax

编程语言 = 语法 + 语义

- 语法: What sequences of characters are valid programs? 
- 语义: What is the behavior of a valid programs?

 Abstract Syntax 用于衔接语法和语义部分(抽象语法树)

## 1. Attribute Grammar

该部分内容 **不要求掌握**,为前置内容.

属性文法: 上下文无关语法+属性+属性计算规则.

- 属性: 描述文法符号的语义特征，如变量类型、值. 例: 非终结符E的属性E.val（表达式的值）
- 属性计算规则(语义规则): 与产生式相关联、反映文法符号属性间关系的规则, 比如”如何计算E.val”

潜在应用:

- 推导类--程序分析

  ![image-20250412161719889](./Abstract%20Syntax.assets/image-20250412161719889.png)

- 生成类--程序合成

  ![image-20250412161733052](./Abstract%20Syntax.assets/image-20250412161733052.png)



## 2. Semantic Action

语义动作可应用于构造AST.

每一个 terminal 和 nonterminal 都有 its own type of semantic value，当存在 rule ：$A \rightarrow BCD$ 时，semantic action 必须返回与 nonterminal A 的 type 一致的一个 value，该value可从 B，C，D 的 values 得到。

例：

![image-20250412162942470](./Abstract%20Syntax.assets/image-20250412162942470.png)

所以，semantic action 定义如下：

- the **values** returned by parsing functions,
- or the **side effects** of those functions
- or both

Semantic Actions in Yacc-Generated Parsers：

![image-20250412163222397](./Abstract%20Syntax.assets/image-20250412163222397.png)

### 2.1 Abstract syntax tree

#### 2.1.1 定义

AST是 Parse Tree 的一个保留核心细节的化简版本,  被输入进语义分析部分. 所以在Parsing过程中还需要生成AST.

![image-20250412163732448](./Abstract%20Syntax.assets/image-20250412163732448.png)

其中，Parse trees encodes the grammatical structure of the source，但携带了很多不必要的信息（Redundant and useless tokens for later phases、太依赖语法），编译器只需要知道 expression 的operators 和 operands 就可以了。

![image-20250412164128238](./Abstract%20Syntax.assets/image-20250412164128238.png)

应用：

- Applications inside compilers
  - Semantic analysis , e.g., type checking
  -  Translation to in termediate representations
  -  Some high-level optimizations, e.g., constant fold
-  Applications outside compilers
  - Pretty print, code editing, code diff, etc.
  - “Advanced” analysis tools (e.g., Clang Static Analyzer)

#### 2.1.2 表示

![image-20250412165316561](./Abstract%20Syntax.assets/image-20250412165316561.png)

![image-20250412165517582](./Abstract%20Syntax.assets/image-20250412165517582.png)

![image-20250412165542078](./Abstract%20Syntax.assets/image-20250412165542078.png)

![image-20250412165555020](./Abstract%20Syntax.assets/image-20250412165555020.png)

构造: 通过semantic actions 

*C语言实现细节不会考*()

#### 2.1.3 生成

**Top-down** -- C和Java实现,应该不会考

递归下降，手写

![image-20250412165819684](./Abstract%20Syntax.assets/image-20250412165819684.png)

C语言：

![image-20250412170349343](./Abstract%20Syntax.assets/image-20250412170349343.png)

![image-20250412170424778](./Abstract%20Syntax.assets/image-20250412170424778.png)

![image-20250412170453121](./Abstract%20Syntax.assets/image-20250412170453121.png)

Java可以边构造边打印。

**Bottom-Up** -- Yacc 会考含义

![image-20250412171131786](./Abstract%20Syntax.assets/image-20250412171131786.png)

