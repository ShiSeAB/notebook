---
counter: True   
---



# Dynamic Early Exit in Reasoning Models

Large reasoning language models(LRLMs) 目前依赖 **test-time scaling(利用测试时的资源进行对模型能力的扩展)** ，延长 long CoT 生成来解决复杂问题是一种测试时缩放的方法。然而，long CoT中的 overthinking 问题会降低问题解决的效率，而且由于推理步骤极其详细或冗余，还存在导致准确性下降的风险。

本文提出 “allows LLMs to self-truncate CoT sequences by early exit during generation” 的方法，The proposed method monitors model behavior at potential reasoning transition points (e.g.,“Wait” tokens)潜在推理转换点， and dynamically terminates the next reasoning chain’s generation when the model exhibits high confidence in a trial answer. 无需 additional training，并可无缝集成到现有类似o1推理LLMs.

平均将思维链序列的长度缩短了 31% 至 43%，同时将准确率提高了 1.7% 至 5.7%。

## 介绍

思维链的冗余可归因于 supervised fine-tuning 或强化学习，在这些阶段中，模型在生成过程中动态调整其推理长度的能力被忽视了。

识别"Pearl Reasoning"(the critical point where the reasoning information becomes just sufficient)，并迫使模型在这一点上停止进一步思考，直接输出结论，就能够兼顾准确性和效率。验证长思维链确有 Pearl 以后，问题来到如何找到“pearl reasoning”。

论文方法为 **DEER** （Dynamic Early Exit in Reasonin），It regards the key moments when the model switches thought chains in reasoning as chances of early exit,  and prompting LRLMs to stop thinking and generate trial answers at these moments.

- Reasoning Transition Monitoring: "wait"为推理转换关键点
- Trial Answer Inducing：
- Confidence Evaluating

##  Preliminaries

### LRLM 生成模式

- use delimiters分隔符 to divide the output into two processes: slow thinking and conclusion，在慢思考过程中进行系统且全面的推理，最终总结思维过程并在结论部分给出最终答案。
- During the slow thinking process, LRLMs engage in complex thinking actions, such as Problem Restatement & Comprehension, Approach Exploration, and Result Verification

将每个 thinking action 称为一个 thinking chunk，chunks 之间的转换通常由 **action transition points** 标记（including ”Wait”, ”Alternatively”, and ”Hmm”）

![image-20250430220651213](./Dynamic%20Early%20Exit%20in%20Reasoning%20Models.assets/image-20250430220651213.png)

### Budget Forcing

该方法利用 LRLMs 中独特的生成模式来控制 test-time computation. 一种简单的解码时干预措施，即在测试时强制设定慢思考标记的最大数量和最小数量。具体而言，当达到标记数量上限时，他们会附加思考结束标记分隔符以及 “最终答案：”，以便提前退出思考阶段。为了确保达到标记数量下限，抑制思考结束标记分隔符的生成，并在大型推理语言模型正在进行的思维过程中添加动作转换点，从而促使模型再次检查其答案或尝试新的推理方法。然而，他们提出的预算强制方法是 static 的，仍有很大的改进空间。



## Motivation and Observations

分析LRLMs 中的 overthinking 问题，并探究 static early exit 的影响。

首先用 DeepSeek-R1-Distill-Qwen-14B 在 AIME2024、GPQA-Diamond、MATH-500上进行实验：先进行完整推理，然后保留慢思考过程，根据 action transition points 划分思维块。若样本超过5个思维块，保留样本，并对样本思维块进行截断（20%-90%）。在每个被截断的推理序列后附加一个思考结束标记分隔符，以强制终止慢思考过程。然后，模型基于部分推理内容生成最终结论。AIME2024每个样本结果如图1：

![image-20250430222542104](./Dynamic%20Early%20Exit%20in%20Reasoning%20Models.assets/image-20250430222542104.png)

图1证明了 “Pearl Reasoning” 的存在，约 75% 的样本包含这样的 “珍珠”（提前退出能得出正确答案），并且 36.7% 的样本在推理路径的前半段就出现了 “珍珠推理” 情况。此外，我们观察到，存在一些样本，只有通过提前退出才能得出正确答案（例如，图 1（a）中的问题 11、19 和 26）。

图2说明了静态退出的不足：

![image-20250430222213649](./Dynamic%20Early%20Exit%20in%20Reasoning%20Models.assets/image-20250430222213649.png)



## Method

core idea: 对 trial answer 的 confidence 表明 LRLMs 生成最终答案所需的思维信息是否充足. 

观察到当模型推理过程不完整或存在缺陷时，trial answer的置信度往往会显著降低。相反，当推理全面且逻辑合理时，模型生成的答案置信度会更高。这表明模型的 parameter space 本质上对 Pearl Reasoning 有一种隐含认知，但由于训练过程中忽略了 dynamically varying lengths of reasoning chains，模型无法明确提前终止推理。DEER通过激活并利用这种隐含认知来实现动态提前退出

![image-20250430231933303](./Dynamic%20Early%20Exit%20in%20Reasoning%20Models.assets/image-20250430231933303.png)

三个模块：

- reasoning transition monitor: 识别潜在退出点

- trial answer inducer：模型在潜在退出点暂停时，该模块 prompts($I$) 模型基于 so far 的推理内容生成中间答案。在 prompts 中加入答案分隔符 `\boxed{}` 来更精确识别 trial answer.
  $$
  A = LRLM(P,T,I)
  $$
  P=prompt，T=generated thoughts，I = answer inducer prompt，A=trial answer($A=[a_0,a_1,...,a_n]$)

- confidence evaluator：take the maximum predicted probability of each token as its confidence. For multi-token trial answers, the overall confidence is computed as the mean confidence across all constituent tokens

  ![image-20250501132212220](C:/Users/shise/AppData/Roaming/Typora/typora-user-images/image-20250501132212220.png)

最后，将得到的置信度与经验阈值*λ*进行比较，以此来决定是否提前退出。if $C>\lambda$ , 我们就认为 LRLM 当前生成的推理信息已足够，这表明该模型已经达到了 “Pearl Reasoning” 状态。此时，DEER 会停止进一步的推理操作，并着手给出结论。否则，模型会回到之前的转换点，以生成下一个思维块。



由于 Answer Inducer 和 Confidence Evaluator 引入额外延迟，所以将 DEER 和 a branch-parallel acceleration strategy 结合，以克服效率限制：

![image-20250501132847374](./Dynamic%20Early%20Exit%20in%20Reasoning%20Models.assets/image-20250501132847374.png)

- Multiple branches are linearized into a single sequence and generated in parallel using a special causal attention mask; ？
- 通过 confidence-based pruning 进行动态键值 cache 管理。这一策略使得 trial answer evaluation 和正在进行的推理链生成在时间上能够重叠，从而优化了整体的推理效率。

![image-20250501133501058](./Dynamic%20Early%20Exit%20in%20Reasoning%20Models.assets/image-20250501133501058.png)



## Experiment

### Implementation

![image-20250501133736123](./Dynamic%20Early%20Exit%20in%20Reasoning%20Models.assets/image-20250501133736123.png)

Metrics：

- ACC

  ![image-20250501134105032](./Dynamic%20Early%20Exit%20in%20Reasoning%20Models.assets/image-20250501134105032.png)

  $\mathbb{I}$ 为指示函数，判断括号内给定条件是否成立；$x_i$ 为问题，$y_i$ 为数据集中的真实答案

- LEN

  生成的文本越长，大型推理语言模型的推理成本就越高。因此，我们通过计算每个样本的平均生成长度来评估成本：

  ![image-20250501134459267](./Dynamic%20Early%20Exit%20in%20Reasoning%20Models.assets/image-20250501134459267.png)

实现细节：zero-shot CoT("think step by step, and put your final answer within \boxed{}.") 采用 greedy decoding with a single sample for the correctness evaluation. Apply rule-based evaluations directly to verify mathematical equivalence. Max generation len = 16384

###  Result

![image-20250501134840218](./Dynamic%20Early%20Exit%20in%20Reasoning%20Models.assets/image-20250501134840218.png)

对于左边两个较简单的数据集，模型参数越少，过度推理情况越严重。

LRLMs 在处理具有挑战性的问题时也会出现 overthinking 的现象，而且当模型的推理能力与基准测试的难度相匹配时，这种现象会更加明显(Diamind 和 AIME2024 中，随模型参数增多，deer对而vanilla CoT错的sample增加)。

###  Discussion

阈值 $\lambda$ 的影响：分别设为 0.9（过早退出）、0.95 和 1.0（过晚退出）时的实验结果。结果表明，当阈值设置得过低时，推理长度相对0.95只会轻微缩短，但准确率显著下降，是一种“对overthinking的过度纠正”；相反，阈值设置过高时推理长度显著延长且准确率下降。

![image-20250501135822543](./Dynamic%20Early%20Exit%20in%20Reasoning%20Models.assets/image-20250501135822543.png)

thought switch signals 的影响：更好的推理块分割方式能够进一步提升推理中的提前退出效果

![image-20250501150323675](./Dynamic%20Early%20Exit%20in%20Reasoning%20Models.assets/image-20250501150323675.png)



























