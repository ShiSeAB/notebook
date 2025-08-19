# Reward model

奖励模型适用于奖励不容易手工定义，需要学习人类偏好的任务；奖励函数则直接定义数学公式，更加简单。

## 背景

- 提出：论文 [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155) 构建一个能够对大模型Response进行打分的模型，从而在强化学习中使模型向人类偏好的方向学习。通常，奖励模型的输入包含一个问题Query和一个模型输出Response。其中偏好数据需要人类标注对多个 response 进行排序。训练奖励模型的损失函数如下示，yl 代表排名低于 yw 的所有 response：

  ![image-20250818131823012](./Reward%20model.assets/image-20250818131823012.png)

- 改进1：[Let's Verify Step by Step](https://arxiv.org/abs/2305.20050) 对中间状态进行监督，提出 Process Reward Model，即PRM，该模型能够对模型生成的中间状态进行打分，从而做到更加精准的credit assignment

- 改进2：[Math-Shepherd: Verify and Reinforce LLMs Step-by-step without Human Annotations](https://arxiv.org/abs/2312.08935) 使用自动的方式来构建PRM，通过蒙特卡洛采样（MCTS）来在可验证的任务上来构建PRM训练数据。为了标注具体的某一步，作者采用 fine-tune 的 LLMs decode（后文称为completer）从当前步的多个一系列的推导（多条推导轨迹），验证最终 decode 的和 golden answer 是否匹配。如果推理的步骤能够比其他步推导出更多的正确答案，这一步将被给予更高的正确分数（reward）。

  ![image-20250817195925110](./Reward%20model.assets/image-20250817195925110.png)

- 目前大多数 GRPO 都是在 response 生成完毕后再赋奖励（包括 DeepSeek-Prover-V2，也是在生成证明后验证是否可通过lean再给奖励），但是 [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/abs/2402.03300) 将 PRM 运用于 GRPO。

  - 前者，即 基于 ORM/rule-based 的 GRPO，Advantage在每个Token上是等权的
  - 而基于 PRM 的 GRPO 在每个Token上有不同的advantage，具体为当前token为后续所有分数的加和（maybe 我还要去看下论文才懂）


## 现有范式

from 生成方式：

- Scalar : model 直接生成一个 reward 数值
- Semi-Scaler : model 在生成一段分析文本后，生成一个数值
- 前两者都是 discriminative 打分，只给出一个总分，无法精确指出回复中具体哪个部分好或不好
- Generative ：model 直接生成一段分析文本，在分析文本中描述给出的 reward（一般需要固定格式训练）（代表工作：LLM-as-a-Judge）

from 作用方式：

- Pointwise ：直接对单个样本给出 reward
- Pairwise ：给出两个 response，由模型判断好坏，选出更好的回答，典型的样本格式为 \<prompt, chosen, rejected\> .

![image-20250817215521391](./Reward%20model.assets/image-20250817215521391.png)

## 构建方式

暂时只讨论纯文本模态……

### DPO方式

[Beyond Scalar Reward Model: Learning Generative Judge from Preference Data](https://arxiv.org/pdf/2410.03742)

直接在预训练模型的生成层上做DPO，让模型学会在给定正负回答对时，同时生成“更优判断＋推理理由”和“劣势判断＋推理理由”，从而取代传统的标量回归头。该方法将奖励学习和策略优化整合为一步，更加简单。

流程：

![image-20250817221740291](./Reward%20model.assets/image-20250817221740291.png)

- LLM 拿到 $(x,r^+,r^-)$ 后，Con-J 会先用当前或预训练的 LLM 给出两条判断（LLM必须较强，减少人工成本）：
  1. “为什么 $r^+$ 比 $r^-$ 好” 的“正向判断＋理由” $\text{Judg}^+$；
  2. “为什么 r^-不如 r^+” 的“负向判断＋理由” $\text{Judg}^-$
- repeated sampling : 在相同提示下多次让 LLM 生成输出，每次生成都使用不同的随机种子。这样可以得到多条判断结果。但是，如果 LLM 在所有重复采样中始终只给出单方面的判断（即所有判断都偏向 a₁ 或都偏向 a₂），那么就无法通过重复采样来构造对比式判断对。
- hint-driven sampling : 当仅靠重复采样无法获得“正向”与“反向”两类判断时，强制 LLM 生成特定答案的偏好判断。具体而言，提示中会明确指出哪一个答案是更好的，然后要求 LLM 按照相同的 JSON 格式生成判断。即分别给模型一个“正确提示”与一个“错误提示”，由此生成一条正向判断和一条反向判断，从而构造出一对判断对。

### SFT 方式

[Process reward model that think](https://arxiv.org/abs/2504.16828)

generative PRM

在不破坏通用推理能力的前提下，对一个通用模型进行有效特化。这是THINKPRM方法论所要解决的根本性问题。流程为用 Reject sampling (让 AI 生成多个答案，然后只选择最优的答案来继续训练。) 获得少量的合成数据，通过 SFT 增强 zero-shot reasoning LLM 作为reward model的能力

第一步：SFT data 合成

![image-20250818154451151](./Reward%20model.assets/image-20250818154451151.png)

- 首先，将 PRM800K 数据集中的“问题-解决方案” pair 输入给教师模型（论文中采用 QwQ-32B-Preview），以生成原始的 verification 链
- 接着进行筛选：生成的 verification 链志愿在其对每一个步骤的判断都与 PRM800K 数据集中人工标注的 label 一致时，才会被保留
- 同时，不符合预设输出格式与超出长度限制的 verification 链都会被丢弃
- 对得到数据进行预处理：删除最后一次验证决策之后的所有内容；添加特殊标记（例如 <think> 和 </think>）以标注验证推理部分。

有了高质量的训练数据，下一步就是将一个通用的 LRMs（如 R1-Distill-Qwen 系列）转化为一个专业的验证器。论文采用 LoRA 微调。

实验部分，有数据支持其 SOTA

![image-20250818161841726](./Reward%20model.assets/image-20250818161841726.png)

[GenPRM: Scaling Test-Time Compute of Process Reward Models via Generative Reasoning](https://arxiv.org/abs/2504.00891)

不仅思维链验证，并且加入代码验证，通过 test-time-scale 让小模型逆袭，为 PRM 提供新思路

![image-20250818174053432](./Reward%20model.assets/image-20250818174053432.png)

- stage 1 : 使用 Qwen2.5-7B-Instruct 为原问题生成多种解决方案，对于方案中的每一步，系统会计算其 MC 得分（从该 step 开始，生成 K 条完整的 rollout，并计算些轨迹最终得出正确答案的比例），MC得分可以被视为从当前状态出发，成功解决问题的经验概率。K 值可根据问题难度自适应。

- Relative Progress Estimation 的提出：简单地将 MC 得分 > 0 的步骤标记为“正确”（标签为1）是一种充满噪声的做法 。一个步骤可能在局部上是正确的（例如，一个无误的计算），但对于整个解题过程而言可能毫无助益，甚至是一个弯路。RPE 通过比较当前状态和上一状态的 MC 分数，用 ”进步幅度“ 来评估每一步的质量，比传统的硬标签更加可靠。

  ![image-20250818191749027](./Reward%20model.assets/image-20250818191749027.png)

  - 若进步幅度低于阈值（ϵ=0.8），则判定步骤无效；若首步错误（MC 为 0），后续步骤分数归零。

- 在通过RPE为每个步骤确定了高质量的二元标签后，用 QwQ-32B 来为每个步骤生成包含CoT分析和代码验证的结构化文本。再使用 QwQ-32B 基于自身理解对其进行判断，如果判断与先前通过RPE计算得出的标签存在任何不一致，整个接替方案被丢弃。

  - 对每个步骤，用 **<analysis> </analysis>** 标签让模型分析推理过程，详细解释段落正确或不正确的原因；同时使用 **<verify> </verify>** 标签让模型针对可以使用 python 验证的部分写出相应的验证代码，执行并将结果写入 [Code Output]。如果 <verify> 与<analysis> 验证不一致，模型会进行自我反思，直到生成的一致为止。

- 得到高质量合成数据后，对模型进行 SFT。

应用：集成到小模型使其获得超越大模型表现的能力；作为 verifier 使用 best-of-N；作为步骤级别的 Critic 模型指导策略模型迭代优化



### 类 R1 的 RL 方式训练 RM

[J1: Incentivizing Thinking in LLM-as-a-Judge via Reinforcement Learning](https://arxiv.org/abs/2505.10320)

























