# Chapter 4: Data Movement instruction

## 4–1 MOV Revisited



<img src="D:\Softwares\Typora\photos\image-20241222174115725.png" alt="image-20241222174115725" style="zoom:50%;" />

–L=0 and D/B =0 for 16-bit instruction mode 

–L=0 and D/B =1 for 32-bit instruction mode 

–L=1 for 64-bit instruction mode

![image-20241222174202661](D:\Softwares\Typora\photos\image-20241222174202661.png)

关于prefix： 非default operand size，要加66H，非default address size，要加67H

#### opcode部分：

<table frame = void>
    <tr>
    <td><center><img src="D:\Softwares\Typora\photos\image-20241222174554803.png" alt="image-20241222174554803"
                     style="zoom:100%;" /></center></td>
    <td><center><img src="D:\Softwares\Typora\photos\image-20241222174621799.png" alt="image-20241222174621799"
                     style="zoom:100%;" /></center></td>
    <td><center><img src = "D:\Softwares\Typora\photos\image-20241222174457763.png" alt="image-20241222174457763"
                     style="zoom:80%"/></center></td>
    </tr>
</table>



#### MOD-REG-R/M 部分

- MOD

<img src="D:\Softwares\Typora\photos\image-20241222174956107.png" alt="image-20241222174956107" style="zoom:50%;" />

REG/Opcode 根据情况填寄存器编码或opcode，上面的指令填的reg，下面的填的opcode

<img src="D:\Softwares\Typora\photos\image-20241222175118586.png" alt="image-20241222175118586" style="zoom:30%;" />

- R/M   即reg/mem

  当为内存寻址时，有一个编码表

  <img src="D:\Softwares\Typora\photos\image-20241222175533131.png" alt="image-20241222175533131" style="zoom:50%;" />

  example



## 4–2 LOAD EFFECTIVE ADDRESS



## 4–3 STRING DATA TRANSFERS 

- CMPS //控制部分再介绍，比较复杂

下述传输指令不改变控制状态：

- LODS  只有该指令不能加rep前缀
- STOS
- MOVS
- INS
- OUTS

## 4–4 MISCELLANEOUS DATA TRANSFER INSTRUCTIONS 

- XCHG
- LAHF and SAHF 
- XLAT  隐含操作数为AL, BX
- IN and OUT 是端口用于读写数据的指令；而INS和OUTS是CPU用于读写数据的指令；IN是写入reg，OUT是从reg读，reg为A系列
- MOVSX and MOVZX
- BSWAP
- CMOV 

## 4–5 SEGMENT OVERRIDE PREFIX

改写默认段：

<img src="D:\Softwares\Typora\photos\image-20241224214201392.png" alt="image-20241224214201392" style="zoom:50%;" />

## 4–6 ASSEMBLER DETAIL  

编译器给编程人员提供的支持：

**伪指令**：告诉编译器该怎么做

- Storing Data in a Memory Segment
  -  **DB、DW、DD、DQ、DT**
  - **DUP**
  - **ALIGN**

<img src="D:\Softwares\Typora\photos\image-20241224222138409.png" alt="image-20241224222138409" style="zoom:67%;" />

- **ASSUME, EQU, and ORG**
- **PROC and ENDP**
- **MACRO and ENDM**

- **INCLUDE**
- **Memory Organization**