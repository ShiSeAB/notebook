# Reward model

- 提出：论文 [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155) 构建一个能够对大模型Response进行打分的模型，从而在强化学习中使模型向人类偏好的方向学习。通常，奖励模型的输入包含一个问题Query和一个模型输出Response。其中偏好数据需要人类标注对多个 response 进行排序。

- 改进1：[Let's Verify Step by Step](https://arxiv.org/abs/2305.20050) 对中间状态进行监督，提出 Process Reward Model，即PRM，该模型能够对模型生成的中间状态进行打分，从而做到更加精准的credit assignment

- 改进2：[Math-Shepherd: Verify and Reinforce LLMs Step-by-step without Human Annotations](https://arxiv.org/abs/2312.08935) 使用自动的方式来构建PRM，通过蒙特卡洛采样（MCTS）来在可验证的任务上来构建PRM训练数据。为了标注具体的某一步，作者采用fine-tune的LLMs decode（后文称为completer）从当前步的多个一系列的推导（多条推导轨迹），验证最终decode的模型是否和golden answer是否匹配。如果推理的步骤能够比其他步推导出更多的正确答案，这一步将被给予更高的正确分数。














