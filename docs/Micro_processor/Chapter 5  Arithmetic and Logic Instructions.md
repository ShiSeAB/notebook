# Chapter 5:  Arithmetic and Logic Instructions

## 5-1 ADDITION, SUBTRACTION AND COMPARISON

### 5-1-1 ADDITION

register addition & immediate addition & mem-to-reg addition & Array addition

- **ADD**
- **ADC**：会加上CF的值；有两个变种(增加并行性)
  - **ADCX**： uses the Carry flag and leaves the other flags untouched.
  - **ADOX**： uses the Overflow flag and leaves the other flag untouched.
- **INC**: 不会破坏CF(其它flag值会变)、指令编码比ADD短
- **XADD**  可用于构建乐观锁

### 5-1-2 **Subtraction** 

对照addition操作

- SUB
- SBB(subtract-with-borrow instruction)
- DEC

### 5-1-3 Comparison

- CMP
- CMPXCHG   可以用于做锁；ex. Lock-free Stack
- CMPXCHG8B/CMPXCHG16B 可以做更大字节的操作

## 5-2 MULTIPLICATION AND DIVISION 

### 5-2-1 Multiplication 

分为无符号MUL和有符号IMUL 

MUL：不允许立即数，只有一个操作数

- 8-Bit Multiplication：隐含操作数为A系列寄存器，AL存乘数，AX存积
- 16-Bit Multiplication： AX存乘数，积低16位在AX，高16位在DX
- 32-Bit Multiplication： EAX存乘数，积低32位在EAX，高32位在EDX
- 64-Bit Multiplication： RAX存乘数，积低64位在RAX，高64位在RDX

IMUL：

- one-operand form ： 和MUL一致（8、16、32、64 bit的寄存器配置），不允许立即数
- two-operand form：没有隐含操作数
- three-operand form

FLAGS Affected by MUL

FLAGS Affected by IMUL

### 5-2-2 Division 

unsigned (DIV) or signed (IDIV) integers

- 8-Bit Division：隐含操作数为A系列寄存器，AX存被除数，AL存商，AH存余数
- 16-Bit Division：被除数高16位放在DX，低16位放在AX；商放在AX，余数放在DX
- 32-Bit Division：被除数高32位放在EDX，低32位放在EAX；商放在EAX，余数放在EDX
- 64-Bit Division：被除数高64位放在RDX，低64位放在RAX；商放在RAX，余数放在RDX

符号填充操作：CBW/CWDE/CDQE    CWD/CDQ/CQO

余数处理

## 5-3 BCD and ASCII Arithmetic 

BCD Arithmetic: 先做16进制加减法，再做调整为BCD码

- DAA(decimal adjust after addition)  AL是隐含源和dest操作数
- DAS(decimal adjust after subtraction) 和DAA同理

ASCII Arithmetic ：先用16进制加减法再调整

- AAA (ASCII adjust after addition) ： AX为src和dest操作数
- AAS (ASCII adjust after subtraction)
- AAM (ASCII adjust after multiplication)
- AAD (ASCII adjust before division)

## 5-4 BASIC LOGIC INSTRUCTIONS 



- AND 
- OR
- Exclusive-OR
- TEST ： 通常用于判断有符号数是否小于/等于0
- Bit Test Instructions ：BT、BTC、BTR、BTS
- NOT：不会影响flag
- ==NEG==
- SHIFT移位操作
  - logical：SHL/SHR
  - arithmetic：SAL/SAR
  - Double-Precision Shifts： SHLD/SHRD
- Rotate旋转操作
  - 带CF旋转：RCL/RCR
  - 不带CF旋转：ROL/ROR
- Bit Scan Instructions
  - BSF(bit scan forward) scans the source number from the least significant bit toward the left. 从最低位开始向左扫描
  - BSR(bit scan reverse) scans the source number from the most significant bit toward the right. 从最高位向右扫描
  - TZCNT(trailing zero count) counts the number of trailing zero bits
  - LZCNT(leading zero count) returns the number of leading zero bits

## 5-6 STRING COMPARISONS 

- SCAS (string scan)
- CMPS (string compare)