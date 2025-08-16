# DAPO

在训练 long-CoT 推理模型时，模型分布可能会与初始模型产生显著差异，KL散度约束并无必要，所以 DAPO 算法排除了 KL 散度。Reward 采用可验证任务的最终准确率。

DAPO 公式如下：

![image-20250816211325904](./DAPO.assets/image-20250816211325904.png)