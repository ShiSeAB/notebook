# chapter 3 - addressing mode

- operation mode

  address size看寻址寄存器大小，因为是用该寄存器中存储的地址寻址；同理operand-size也是看存储操作数的寄存器大小。如右图，EAX是32位，RBX是64位，正好符合64-bit模式下的default size。

<table frame = void>
    <tr>
    <td><center><img src="D:\Softwares\Typora\photos\image-20241222112130530.png" alt="image-20241222112130530" 
                     style="zoom:40%;" /></center></td>
    <td><center><img src="D:\Softwares\Typora\photos\image-20241222112102977.png" alt="image-20241222112102977"
                     style="zoom:40%;" /></center></td>
    </tr>
</table>



- addressing mode

  <img src="D:\Softwares\Typora\photos\image-20241222112638240.png" alt="image-20241222112638240" style="zoom:50%;" />

## 3-1 data-addressing modes

<img src="D:\Softwares\Typora\photos\image-20241222113253613.png" alt="image-20241222113253613" style="zoom:50%;" />

​	三种operand：

- immediate   只能做源操作数
- register
- memory

<img src="D:\Softwares\Typora\photos\image-20241222113440177.png" alt="image-20241222113440177" style="zoom:50%;" />

### register

​	The most common form of data addressing.

- 两个寄存器长度要一致。

- example： MOV BX, CX，注意EBX高16位不变，CX中存储的1234存到BX中。

  <img src="D:\Softwares\Typora\photos\image-20241222120258281.png" alt="image-20241222120258281" style="zoom:50%;" />

- 只有cmp和Test操作的目的寄存器不变，变的是flag寄存器

### immediate

example

<img src="D:\Softwares\Typora\photos\image-20241222120530332.png" alt="image-20241222120530332" style="zoom:50%;" />

- 当立即数用16进制表示时，必须用0-9打头，才能不被识别成 label

  <img src="D:\Softwares\Typora\photos\image-20241222120619245.png" alt="image-20241222120619245" style="zoom:50%;" />

- ASCII-data要用 '' 圈起来

<img src="D:\Softwares\Typora\photos\image-20241222120723822.png" alt="image-20241222120723822" style="zoom:50%;" />



### memory

关键是算 effective address

<img src="D:\Softwares\Typora\photos\image-20241222120846166.png" alt="image-20241222120846166" style="zoom:50%;" />

#### Direct Data Addressing 

​	Address is formed by adding the displacement to the default data segment address or an alternate segment address.

​	**EA = Disp**

- Direct Addressing

  目的寄存器为A类寄存器，inst长度为3 Bytes

  <img src="D:\Softwares\Typora\photos\image-20241222121600414.png" alt="image-20241222121600414" style="zoom:50%;" />

  <img src="D:\Softwares\Typora\photos\image-20241222121927687.png" alt="image-20241222121927687" style="zoom:50%;" />

  上述例子中用到很多**数据标签**。

- Displacement Addressing

  目的寄存器一般不为A类寄存器，inst长度为4 Bytes。

  <img src="D:\Softwares\Typora\photos\image-20241222122025198.png" alt="image-20241222122025198" style="zoom:50%;" />

如图为compiler利用direct addressing编译C代码，var是一个数据标签

<img src="D:\Softwares\Typora\photos\image-20241222122428750.png" alt="image-20241222122428750" style="zoom:50%;" />

#### Register Indirect Addressing

​	**EA = Base**

​	8086&80286:  only **BX, BP, SI and DI** registers 可以做Base寄存器

​	80386： allow any extended register。

​	BX, SI and DI 的段基地址都是**Data segment**，BP的seg为**Stack Segment**

- size directive：用于指明要取数据的长度，避免ambiguous

  

#### Base-Plus-Index Addressing

​	**EA = Base + Index**

​	8086&80286:  only BX, BP可以做Base寄存器，DI、SI做index寄存器

​	80386：除了ESP不能做index寄存器，没有其它限制。

#### Register Relative Addressing

​	**EA = Base or Index + Disp**

​	8086&80286:  only BX, BP可以做Base寄存器，DI、SI做index寄存器

​	80386：除了ESP不能做index寄存器，没有其它限制。

#### Base Relative-Plus-Index Addressing 

​	**EA = Base + index + Disp**

​	8086&80286:  only BX, BP可以做Base寄存器，DI、SI做index寄存器

​	80386：除了ESP不能做index寄存器，没有其它限制。

#### Scaled-Index Addressing

​	**EA = Base + scale * index + Disp**

## 3–2 PROGRAM MEMORY-ADDRESSING MODES 

两类指令：JMP (jump) and CALL instructions.

<img src="D:\Softwares\Typora\photos\image-20241222142411654.png" alt="image-20241222142411654" style="zoom:50%;" />

## 3–3 STACK MEMORY-ADDRESSING MODES

PUSH&POP