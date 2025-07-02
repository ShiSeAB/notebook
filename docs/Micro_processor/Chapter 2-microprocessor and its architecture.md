## Chapter 2-microprocessor and its architecture

目标：

- 了解每个program-visible寄存器的功能与目的
- 重点了解flag register，说出每个bit代表什么
- real mode、protected mode、64-bit flat memory model下各是如何寻址的
- memory-paging mechanism是如何工作的

### 2-1 INTERNAL MICROPROCESSOR ARCHITECTURE

####  program visible

​	如下分别为32位和64位架构下的寄存器，其中MMX寄存器复用了在x87中的80位拓展的浮点寄存器，在IA中用于向量化操作（取低64位）。当MMX从浮点操作转化为向量化操作时，不用额外操作，直接就可以用（向量化操作兼容浮点操作），但是从向量操作变到浮点操作时，需要做一个EMMS（empty MM State）的操作，否则FPU会崩溃。

<table frame = void>
    <tr>
    <td><center><img src="D:\Softwares\Typora\photos\image-20241221120624509.png" alt="image-20241221120624509" 
                     style="zoom:40%;" /></center></td>
    <td><center><img src="D:\Softwares\Typora\photos\image-20241221120645587.png" alt="image-20241221120645587"
                     style="zoom:40%;" /></center></td>
    </tr>
</table>

<img src="D:\Softwares\Typora\photos\image-20241221123517229.png" alt="image-20241221123517229" style="zoom:67%;" />

​	64位与32位的区别是寄存器数量增多、长度扩展。

##### 8个通用寄存器

![image-20241221124632455](D:\Softwares\Typora\photos\image-20241221124632455.png)

- **Accumulator**

  The accumulator is used for instructions such as multiplication, division, and some of the adjustment instructions. Accumulator有特殊编码，所以在算术操作时不需要额外位来指示放在哪个寄存器，但是用其它寄存器就需要额外位来指示放在哪个寄存器。如图，立即数被可被视为字符串，按照**小端模式**存储在instuction中（inst为050407ffff，先存 04 然后 07 然后 ff）

<img src="D:\Softwares\Typora\photos\image-20241221125431249.png" alt="image-20241221125431249" style="zoom:50%;float:left" />

- **Base index**

  ![image-20241221130640706](D:\Softwares\Typora\photos\image-20241221130640706.png)

- **Counter**

  一般用于循环计数

  <img src="D:\Softwares\Typora\photos\image-20241221130801436.png" alt="image-20241221130801436" style="zoom:67%;" />

- **Data**

  如图，16位乘法产生的32位积的低位部分放在AX中，高位部分放在DX中。

  <img src="D:\Softwares\Typora\photos\image-20241221130828313.png" alt="image-20241221130828313" style="zoom:67%;" />

- **R8-R15** 64bits

  B代表Byte、W代表word（即2个byte），D代表double word，注意ABCD寄存器还有1byte的高位寄存器

  <img src="D:\Softwares\Typora\photos\image-20241221131315881.png" alt="image-20241221131315881" style="zoom:67%;" />

partial register 现象（目前只有x86为了兼容性还有该功能）：

​	可以看到，给EAX赋值时，RAX高32位自动清零，而给AX赋值时，EAX高16位不变，给AL赋值时，AX高8位不变：

<table frame = void>
    <tr>
    <td><center><img src="D:\Softwares\Typora\photos\image-20241221124947909.png" alt="image-20241221124947909"
                     style="zoom:40%;" /></center></td>
    <td><center><img src="D:\Softwares\Typora\photos\image-20241221125133214.png" alt="image-20241221125133214"
                     style="zoom:40%;" /></center></td>
    </tr>
</table>

​	可以将汇编代码分为**两个独立**的部分。第一部分是乘法，首先将内存地址 `mem1` 处的值加载到寄存器 `EAX` 中，接着做乘法（耗时长），做完以后将结果放到地址 `mem2` 中；第二部分是一个加法。编译器识别到以后可以通过重命名的方法来乱序执行。但是如果把EAX换成AX，就会造成假依赖关系：AX需要等待EAX乘法计算结果（因为给AX赋值时，EAX高16位要不变），从而浪费时间。

<img src="D:\Softwares\Typora\photos\image-20241221131847763.png" alt="image-20241221131847763" style="zoom:50%;" />

##### special-purpose reg

​	Include RIP, RSP（不作为通用寄存器）, and RFLAGS 

- **RIP**：addresses the next instruction in a section of memory.

- **RSP**：addresses an area of memory called the stack. 

- **RFLAGS**：carrry flag为1表示加进位或减借位，zero flag为1表示运算结果为0，sign flag存储结果符号，overflow 为1表示溢出：

  <table frame = void>
      <tr>
      <td><center><img src="D:\Softwares\Typora\photos\image-20241221140458635.png" alt="image-20241221140458635"
                       style="zoom:40%;" /></center></td>
      <td><center><img src="D:\Softwares\Typora\photos\image-20241221141058631.png" alt="image-20241221141058631"
                       style="zoom:40%;" /></center></td>
      </tr>
  </table>

  parity flag置1时运算结果中1的个数为奇数，偶数则clear；

  Auxiliary flag用于支持BCD码运算，表示其进位.**如右图**，8+9 > 9,故Auxiliary位置1，使得8+9的结果要+6，2+6还要加进位1，得正确结果97：

  <table frame = void>
      <tr>
      <td><center><img src="D:\Softwares\Typora\photos\image-20241221142404919.png" alt="image-20241221142404919"
                       style="zoom:40%;" /></center></td>
      <td><center><img src="D:\Softwares\Typora\photos\image-20241221142651626.png" alt="image-20241221142651626"
                       style="zoom:50%;" /></center></td>
      </tr>
  </table>

##### segment reg

​	包括CS、DS、ES、SS、FS和GS六个reg：

<table frame = void>
    <tr>
    <td><center><img src="D:\Softwares\Typora\photos\image-20241221144021435.png" alt="image-20241221144021435"
                     style="zoom:40%;" /></center></td>
    <td><center><img src="D:\Softwares\Typora\photos\image-20241221144100596.png" alt="image-20241221144100596" 
                     style="zoom:50%;" /></center></td>
    </tr>
</table>

​	64-bit模式下，只有CS、FS和GS三个寄存器可用，其它全部置0，且FS和GS只有base可用：

<img src="D:\Softwares\Typora\photos\image-20241221144128418.png" alt="image-20241221144128418" style="zoom:50%;float:left" />

#### system register

<img src="D:\Softwares\Typora\photos\image-20241221144551359.png" alt="image-20241221144551359" style="zoom:67%;" />



### 2-2 Modes of Operation

<img src="D:\Softwares\Typora\photos\image-20241221145012163.png" alt="image-20241221145012163" style="zoom:67%;" />

#### Long mode 长模式

​	**Long mode**, which Intel calls IA-32e ("e" for "extensions"), is an extension of legacy protected mode. 但是不支持legacy mode下的virtual和real模式。

##### 64-bit mode

​	 supports **all of the features and register extensions** of the 64 architecture.

##### compatibility mode

- allowing 64-bit operating systems to run **existing 16-bit and 32-bit x86 applications.**

- access the first **4GB** of virtual-address space

#### Legacy mode 历史遗留模式

​	有如上三种模式。

#### SMM（**System management mode**）

smm -> reset-> real mode -> protected mode -> compatibility mode -> 64-bit mode

每次reset以后都会回到实模式，每次SMI后回到SMM模式

<img src="D:\Softwares\Typora\photos\image-20241221150842415.png" alt="image-20241221150842415" style="zoom:50%;" />

#### Memory management requirement

需要满足三个条件：relocation可重定位（虚拟地址转化为物理地址）、protection（不同进程间内存保护与隔离）、sharing（不同进程间可用共享部分信息）。

##### Segmentation 分段式内存管理

​	段大小可以不一致，需要多少就分多少，导致外部碎片。

<table frame = void>
    <tr>
    <td><center><img src="D:\Softwares\Typora\photos\image-20241221152923105.png" alt="image-20241221152923105"
                     style="zoom:40%;" /></center></td>
    <td><center><img src="D:\Softwares\Typora\photos\image-20241221153237505.png" alt="image-20241221153237505" 
                     style="zoom:50%;" /></center></td>
    </tr>
</table>



##### Paging 分页式管理

比分段更加精细，页大小相同，导致内部碎片，且地址转换比分段慢（多级页表、MMU）

<table frame = void>
    <tr>
    <td><center><img src="D:\Softwares\Typora\photos\image-20241221153434202.png" alt="image-20241221153434202"
                     style="zoom:40%;" /></center></td>
    <td><center><img src="D:\Softwares\Typora\photos\image-20241221153512977.png" alt="image-20241221153512977"
                     style="zoom:50%;" /></center></td>
    </tr>
</table>

### 2-3 real mode memory addressing

​	由于实模式的地址空间为1MB，所以逻辑地址长度为20 bits（没有分页，线性地址就是物理地址，否则是虚拟地址）。段大小为64KB，所以offset为16位。

​	实模式下**没有段表**，段寄存器（有6个）直接存储段首地址的高16位，寄存器长度为16bits。

<table frame = void>
    <tr>
    <td><center><img src="D:\Softwares\Typora\photos\image-20241221154550210.png" alt="image-20241221154550210"
                     style="zoom:40%;" /></center></td>
    <td><center><img src="D:\Softwares\Typora\photos\image-20241221153828973.png" alt="image-20241221153828973"
                     style="zoom:40%;" /></center></td>
    <td><center><img src="D:\Softwares\Typora\photos\image-20241221162051051.png" alt="image-20241221162051051"
                     style="zoom:40%;" /></center></td>
    </tr>
</table>

- **address wrapping problem** 回滚现象

  下面两种情况为溢出错误，如左边，根据逻辑地址知段首为FFFF0H，偏移量为FFFFH，得到线性地址溢出（超过1MB），导致物理地址回滚（兼容性问题，保护模式下32位4GB内存是否回滚呢？需要设置）

  <img src="D:\Softwares\Typora\photos\image-20241221162757720.png" alt="image-20241221162757720" style="zoom:50%;float:left" />

### 2-4 protected mode memory addressing

selector 存储寻找相应descriptor（操作系统设置）的索引/指针，descriptor存储段的地址、长度、权限等信息。CPU需要检查权限后再决定是否可以相加。

<img src="D:\Softwares\Typora\photos\image-20241221163418106.png" alt="image-20241221163418106" style="zoom:50%;" />

- 描述符表类别：

<img src="D:\Softwares\Typora\photos\image-20241221163838638.png" alt="image-20241221163838638" style="zoom:50%;" />

- 初始化描述符表时，将其赋为**null descriptor**（全0），避免选择子寻址时指向错误的地址（指向全0引发中断，避免错误）。

- 一个descriptor长度为64bits（8B），GDT和LDT最大为64KB，所以最多能有8K=8192个entry

  如下题，每个进程需要一个代码段和一个数据段，共占GDT的2个entry，所以最多跑4096个Proc。

  <img src="D:\Softwares\Typora\photos\image-20241221165444849.png" alt="image-20241221165444849" style="zoom:50%;" />

- （**必考！！！**）左图为不同版本下descriptor的格式，右图为力度位G的作用。在80386中，limit只有20位，而地址有32位，如果G为1，limit以4KB为单位，相当于左移12位，就有32位了，limit的范围也相当于变成[ffffffff,fff]，即4KB~4GB。由于limit表示段内最大偏移量，所以计算段大小时limit+1（段内最后一个地址）。

  <table frame = void>
      <tr>
      <td><center><img src="D:\Softwares\Typora\photos\image-20241221170515890.png" alt="image-20241221170515890"
                       style="zoom:50%;" /></center></td>
      <td><center><img src="D:\Softwares\Typora\photos\image-20241221170625406.png" alt="image-20241221170625406"
                       style="zoom:50%;" /></center></td>
      </tr>
  </table>

  example：problem1很简单，直接加就好了；对于problem2，算结束地址即为开始地址+段大小-1，段大小通过limit来算（注意这里的简便方法，limit*4K相当于左移12位，4K-1=FFFH，故size-1 = 001FFFFFH）

  <table frame = void>
      <tr>
      <td><center><img src="D:\Softwares\Typora\photos\image-20241221172213832.png" alt="image-20241221172213832"
                       style="zoom:50%;" /></center></td>
      <td><center><img src="D:\Softwares\Typora\photos\image-20241221172258205.png" alt="image-20241221172258205"
                       style="zoom:50%;" /></center></td>
      </tr>
  </table>

  

- Access right

  注意S和type的配合。以及DPL两位分别表示4个level

  <img src="D:\Softwares\Typora\photos\image-20241221173040873.png" alt="image-20241221173040873" style="zoom:50%;" />

<img src="D:\Softwares\Typora\photos\image-20241221173232829.png" alt="image-20241221173232829" style="zoom:50%;" />

- 选择子格式

  由于描述符为8192个，即8K，所以索引只需要13位；选择子是有权限的。

  <img src="D:\Softwares\Typora\photos\image-20241221173436210.png" alt="image-20241221173436210" style="zoom:50%;" />

  为什么要设置DPL、RPL、CPL三个权限，因为要确保syscall访问kernel里的库函数成功

  <table frame = void>
      <tr>
      <td><center><img src="D:\Softwares\Typora\photos\image-20241221174344642.png" alt="image-20241221174344642"
                       style="zoom:50%;" /></center></td>
      <td><center><img src="D:\Softwares\Typora\photos\image-20241221174451410.png" alt="image-20241221174451410"
                       style="zoom:50%;" /></center></td>
      </tr>
  </table>

  

### 2-6 memory paging

​	os的内容+扩展内容，看ppt。

​	汇编上是logical addr -> linear(virtual) addr -> physical addr	

​	total meltdown是什么意思？