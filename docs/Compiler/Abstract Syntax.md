# Abstract Syntax

## 1. Attribute Grammar

该部分内容不要求掌握,为前置内容.

属性文法: 上下文无关语法+属性+属性计算规则.

潜在应用:推导类--程序分析, 生成类--程序合成



## 2. Semantic Action

### 2.1 Abstract syntax tree

AST是Parse Tree的一个保留核心细节的化简版本,  被输入进语义分析部分. 所以在Parsing过程中还需要生成AST.



表示

构造: 通过semantic actions 

*C语言实现细节不会考*

#### 生成

- Top-down -- C和Java实现,应该不会考
- Bottom-Up -- Yacc 会考含义