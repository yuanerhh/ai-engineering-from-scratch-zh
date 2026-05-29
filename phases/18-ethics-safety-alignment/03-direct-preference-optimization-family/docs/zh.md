# 直接偏好优化家族

> Rafailov 等人（2023）表明，RLHF 的最优解可以用偏好数据的封闭形式来表达，因此你可以跳过显式奖励模型，直接优化策略。这一洞见催生了一个家族——IPO、KTO、SimPO、ORPO、BPO——每个成员都在修复 DPO 的某个失效模式。2026 年，直接对齐算法（DAA）在前沿模型的后训练运行中比 PPO 更为普遍。但第 2 课的过优化曲线依然适用：DAA 没有逃脱古德哈特定律，只是将其咬合点移到了别处。

**类型：** 学习  
**语言：** Python（标准库、六种偏好损失变体比较器）  
**前置条件：** 第 18 阶段第 01 课（InstructGPT），第 18 阶段第 02 课（奖励黑客），第 10 阶段第 08 课（DPO 基础）  
**时间：** 约 75 分钟

## 学习目标

- 从带 KL 约束的 RLHF 最优解推导出 DPO 封闭形式。
- 阐述 IPO、KTO、SimPO、ORPO、BPO 各自修复了 DPO 的哪个失效模式。
- 区分"隐式奖励差距"与"偏好强度"，并解释为什么 IPO 的恒等映射（identity mapping）很重要。
- 解释为什么 Rafailov 等人（NeurIPS 2024）证明 DAA 尽管没有显式奖励模型仍会过优化。

## 问题所在

RLHF 目标（第 1 课）：

```
max_pi E_{x,y~pi} [ r(x, y) ] - beta * KL(pi || pi_ref)
```

有一个已知的最优解：

```
pi*(y|x) = (1/Z(x)) * pi_ref(y|x) * exp(r(x, y) / beta)
```

因此奖励由最优策略与参考策略的比值隐式定义：

```
r(x, y) = beta * log(pi*(y|x) / pi_ref(y|x)) + beta * log Z(x)
```

将这个结果代入 Bradley-Terry 偏好似然函数，配分函数 `Z(x)` 因为只依赖 `x` 而被消去。剩下的是只含策略参数的损失函数——不需要奖励模型。这就是 DPO（直接偏好优化，Direct Preference Optimization）。

问题在于：该推导假设最优解可达、偏好数据来自训练分布内、且参考策略是真实的众数锚点。这些条件没有一个能严格成立。每个家族成员修复的是不同的被违反假设。

## 核心概念

### DPO（Rafailov 等，2023）

```
L_DPO = -log sigmoid(
  beta * log(pi(y_w | x) / pi_ref(y_w | x))
  - beta * log(pi(y_l | x) / pi_ref(y_l | x))
)
```

可能出错的地方：

- 隐式奖励差距 `beta * (log(pi/pi_ref)_w - log(pi/pi_ref)_l)` 无界。微小的偏好可以产生任意大的差距。
- 损失使被选择（chosen）和被拒绝（rejected）的对数概率向相反方向移动。只要被拒绝的下降更快，它可以使被选择的绝对对数概率也下降。这就是"被选择响应退化"（Degraded Chosen Response）现象。
- 分布外偏好（罕见对 vs 罕见对）会产生任意隐式奖励。

### IPO（Azar 等，2024）

恒等偏好优化（Identity Preference Optimization）将 log-sigmoid 替换为偏好概率的恒等映射。损失变成在有界目标上的平方误差：

```
L_IPO = (log(pi(y_w | x) / pi_ref(y_w | x)) - log(pi(y_l | x) / pi_ref(y_l | x)) - 1/(2 beta))^2
```

边距由 `1/(2 beta)` 限定。偏好强度与隐式奖励差距成比例。不会爆炸。

### KTO（Ethayarajh 等，2024）

Kahneman-Tversky 优化（Kahneman-Tversky Optimization）完全去掉了成对结构。给定一个单独标注的输出和一个二元"可取"或"不可取"信号，将其映射到前景理论（prospect-theory）效用：

```
v(x, y) = sigma(beta * log(pi(y|x) / pi_ref(y|x)) - z_ref)
```

对收益和损失使用不同权重（损失厌恶）。好处：可以使用非配对数据，而非配对数据要丰富得多。

### SimPO（Meng 等，2024）

简单偏好优化（Simple Preference Optimization）将训练信号与生成过程对齐。完全去除参考策略，并对对数似然按长度归一化：

```
L_SimPO = -log sigmoid(
  (beta / |y_w|) * log pi(y_w | x)
  - (beta / |y_l|) * log pi(y_l | x)
  - gamma
)
```

加入边距 `gamma` 来稳定训练。长度归一化消除了利用 DPO 长度偏差失效模式的激励（构造上，较长的 `y_w` 会产生更大的对数概率差距）。

### ORPO（Hong 等，2024）

比值偏好优化（Odds-Ratio Preference Optimization）在标准 SFT 负对数似然损失中增加一个偏好项：

```
L_ORPO = L_NLL(y_w) + lambda * L_OR
L_OR = -log sigmoid(log(odds(y_w) / odds(y_l)))
```

不需要参考策略——SFT 项充当正则化器。从基础模型到对齐模型一步完成训练，无需独立的 SFT 检查点。

### BPO（ICLR 2026 投稿，OpenReview id=b97EwMUWu7）

识别了"被选择响应退化"（Degraded Chosen Responses）问题：DPO 保持了 `y_w > y_l` 的排名，但 `y_w` 的绝对对数概率可能下降。BPO 增加了一行修正，惩罚被选择响应的向下移动。据报告，在 Llama-3.1-8B-Instruct 的数学推理上相比 DPO 准确率提升 +10.1%。

### 普适结果：DAA 仍然会过优化

Rafailov 等人"直接对齐算法中奖励模型过优化的规律"（NeurIPS 2024）在多个数据集上、跨 KL 预算用 DPO、IPO、SLiC 训练策略。黄金奖励-vs-KL 曲线呈现与 Gao 等人相同的先达到峰值后崩溃的形状。隐式奖励在训练中查询分布外样本；KL 正则化无法稳定这一点。

DAA 没有逃脱古德哈特定律。它们将咬合点从"奖励模型被过优化"改变为"参考策略比率被过优化"。普适修复——更好的数据、集成、提前停止——对两者都适用。

### 2026 年的选择指南

- 如果有大量成对偏好数据：用 DPO 配保守的 beta，如果长度偏差明显则用 SimPO。
- 如果有非配对二元反馈：用 KTO。
- 如果想要从基础模型开始的单阶段流程：用 ORPO。
- 如果在 DPO 日志中看到被选择的对数概率退化：用 BPO。
- 如果偏好强度差异很大且 DPO 正在饱和：用 IPO。

每个实验室都会在全套方法上运行，并按任务选取最优方法。没有理由认为数学推理和安全性的最优方法是相同的。

## 实践应用

`code/main.py` 在一个玩具偏好数据集上比较六种损失（DPO、IPO、KTO、SimPO、ORPO、BPO），数据集中各对的真实偏好强度各不相同。每种损失在用一个小型 softmax 策略对相同的 500 个样本对进行优化。绘制各方法最终胜率、被选择对数概率漂移和隐式奖励分散度的对比图。

## 交付成果

本课产出 `outputs/skill-preference-loss-selector.md`。给定数据集统计信息（成对 vs 非配对、偏好强度可变 vs 均匀、长度分布）和目标（单阶段或 SFT-then-偏好），推荐一种偏好损失并报告其所防范的失效模式。

## 练习

1. 运行 `code/main.py`。报告 DPO 和 BPO 最终被选择对数概率的下降量。BPO 应保留更高的被选择绝对概率——验证这一点。

2. 修改偏好数据，使所有对具有相同的强度。六种方法中哪种最鲁棒？哪种退化了？解释 IPO 在此处的优势。

3. 使被拒绝的响应平均比被选择的响应长 2 倍。在其他条件不变的情况下，用数字展示 DPO 的长度利用问题以及 SimPO 的修复。

4. Rafailov 等人（NeurIPS 2024）声称 DAA 会过优化。复现一个单点版本：绘制被选择与被拒绝的 KL 散度，并观察 DPO 在大 beta 下的过优化现象。

5. 阅读 BPO 论文摘要（OpenReview b97EwMUWu7）。写下 BPO 相比 DPO 增加的那一行修正。与 `code/main.py` 中的实现对照确认。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| DPO | "不需要奖励模型的 RLHF" | 从封闭形式 RLHF 最优解推导出的损失；仅含策略参数 |
| 隐式奖励（Implicit reward） | "对数比率" | `beta * log(pi(y|x) / pi_ref(y|x))`——DPO 隐含的奖励 |
| IPO | "有界 DPO" | 用恒等映射替换 log-sigmoid；隐式奖励差距被 `1/(2 beta)` 限定 |
| KTO | "非配对 DPO" | 对单标注样本使用带损失厌恶的前景理论效用 |
| SimPO | "无参考策略 DPO" | 长度归一化对数似然加边距；无参考策略 |
| ORPO | "单阶段 DPO" | 负对数似然加比值偏好项；从基础模型一次完成训练 |
| BPO | "保留被选择响应的 DPO" | DPO 加上惩罚被选择响应绝对对数概率下降的修正项 |
| 被选择响应退化（Degraded Chosen） | "被选择的下降了" | DPO 使被选择对数概率下降，只要被拒绝下降得更快 |
| DAA | "直接对齐算法" | 任何跳过显式奖励模型的偏好损失方法 |

## 延伸阅读

- [Rafailov et al. — Direct Preference Optimization (NeurIPS 2023, arXiv:2305.18290)](https://arxiv.org/abs/2305.18290)
- [Azar et al. — A General Theoretical Paradigm to Understand Learning from Human Preferences (AISTATS 2024, arXiv:2310.12036)](https://arxiv.org/abs/2310.12036) — IPO
- [Ethayarajh et al. — KTO: Model Alignment as Prospect Theoretic Optimization (arXiv:2402.01306)](https://arxiv.org/abs/2402.01306)
- [Meng, Xia, Chen — SimPO (NeurIPS 2024, arXiv:2405.14734)](https://arxiv.org/abs/2405.14734)
- [Hong, Lee, Thorne — ORPO (EMNLP 2024, arXiv:2403.07691)](https://arxiv.org/abs/2403.07691)
- [BPO — Behavior Preservation Optimization (ICLR 2026 OpenReview b97EwMUWu7)](https://openreview.net/forum?id=b97EwMUWu7)
- [Rafailov et al. — Scaling Laws for RM Overoptimization in DAAs (NeurIPS 2024, arXiv:2406.02900)](https://arxiv.org/abs/2406.02900)
