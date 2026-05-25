# 扩展律

> 2020 年 Kaplan 的论文说：模型越大，损失越低。2022 年 Hoffmann 的论文说：你训练不足。算力分配给两个桶——参数和 token——而这个分配并不明显。

**类型：** 学习
**语言：** Python
**前置条件：** Phase 7 · 05（完整 Transformer）、Phase 7 · 07（GPT）
**时长：** 约 45 分钟

## 问题背景

当你有 C 个训练算力 FLOP 想要获得最好的模型时，面临两个旋钮：

1. **多少个参数（N）？** 模型越大，容量越高。
2. **多少个训练 token（D）？** 数据越多，容量利用越好。

FLOP 约等于 `6 × N × D`。你可以增大 N 减小 D，或增大 D 减小 N。哪种更好？

2022 年之前，答案是"大力推高 N"。GPT-3（2020）是 1750 亿参数，在约 3000 亿 token 上训练。每个参数约 1.7 个 token 的比例。Kaplan 扩展律支持了这一点。

Hoffmann et al.（2022）训练了一个叫 Chinchilla 的小型模型族，发现了不同的结论：最优比例更接近于**每参数 20 个 token**。GPT-3 的训练不足了 10 倍。Chinchilla（700 亿参数，1.4 万亿 token）在每个基准上都击败了 GPT-3（1750 亿，3000 亿 token），推理成本降低 2.5 倍。

2026 年是 Chinchilla 的时代——但有一个重要转变。Llama 3 8B 在 15 万亿 token 上训练，每参数 1875 个 token 的比例。比 Chinchilla 最优高出 94 倍。对于将被大规模使用的模型，推理成本比训练成本更重要，因此超训练（超过 Chinchilla 最优）以获得更小的可部署模型已成为 2026 年的默认做法。

## 核心概念

![Chinchilla 曲线：不同 N/D 比例下损失与算力的关系](../assets/scaling-laws.svg)

### Hoffmann 定律

来自 Chinchilla 论文，损失遵循：

```
L(N, D) = A / N^α + B / D^β + E
```

- `N` = 参数量（不含嵌入层）。
- `D` = 训练 token 数。
- `α ≈ 0.34`，`β ≈ 0.28`（大致对称）。
- `E ≈ 1.69`，不可约损失上界。
- `A ≈ 406`，`B ≈ 411`。

两个项在扩展时相互权衡。在固定算力（C = 6ND）下对 `N` 求导并求解：

```
N_opt ≈ 0.6 × (C/6)^0.5
D_opt ≈ 0.6 × (C/6)^0.5
D_opt / N_opt ≈ 20
```

算力最优：每参数 20 个 token。

### 为什么还是要超训练

Chinchilla 最优最小化每训练 FLOP 的训练损失。但你只支付一次训练成本；推理成本永远持续。

对于每月服务一万亿 token 的聊天机器人，推理主导总成本。Llama 的方法：训练更小，训练更长。8B 参数在 15 万亿 token 上是深度推理优化的：

- 适合消费级 GPU。
- 延迟是 700 亿 Chinchilla 最优模型的一小部分。
- 大多数任务的质量足够接近。

DeepMind 2024 年的论文（"超训练是新的最优"）将此正式化。对于推理主导的工作负载，正确的比例更接近每参数 100-500 个 token，取决于服务量。

### 涌现 vs 平滑性

说法：某些能力（算术、多步推理、思维链跟随）在某个规模下"涌现"出现。

Schaeffer et al.（2023）认为这是测量假象：涌现指标使用不连续评分（精确匹配、阈值准确率），隐藏了底层 logit 中的平滑改进。连续指标（交叉熵）显示平滑曲线。

2026 年的共识是：通过连续损失的预测是可靠的。基准跳跃通常是评分器假象。根据连续指标制定预算。

### 2026 年的全景

扩展律仍然有效，但：

| 因素 | 变化方式 |
|------|---------|
| 数据质量 | 精选"好"token（Phi 风格）将曲线移动超过 2 倍有效算力 |
| MoE | 总参数与激活 FLOP 解耦；每激活 FLOP 的扩展律 |
| 后训练 | 某些能力（指令遵循、代码）通过 SFT+RLHF 的改变超过预训练 |
| 多模态 | 图像 + 文本 token 一起扩展；每种模态单独的曲线 |
| 合成数据 | 模型生成训练数据；有效算力可以复利增长 |

Muon 优化器（Kimi Moonlight，2024）在相同数据下相比 AdamW 显示出约 2 倍有效算力提升。一些 2026 年的训练运行默认使用 Muon。改变了扩展律中的绝对常数，而非其形状。

## 动手实现

见 `code/main.py`。我们实现 Chinchilla 损失方程，并在几个算力预算下求解算力最优的 `(N, D)`。

### 步骤一：Chinchilla 损失

```python
def chinchilla_loss(N, D, A=406.4, B=410.7, alpha=0.34, beta=0.28, E=1.69):
    return A / N ** alpha + B / D ** beta + E
```

在固定 `C = 6ND` 的情况下，在 `(N, D)` 上绘制 `L` 的等高线。找到最小值。

### 步骤二：算力最优前沿

对于从 `1e17` 到 `1e25` FLOP 的算力预算，找到在约束 `6ND = C` 下最小化损失的 `(N, D)`。验证比例 `D/N ≈ 20`。

### 步骤三：超训练成本

计算训练一个 10 倍更小模型（最优 N 的 1/10，最优 D 的 10 倍）所付出的额外损失。报告相应节省的推理 FLOP（与 N 成比例）。

### 步骤四：与真实模型对比

代入 GPT-3、Chinchilla、Llama 3 8B、DeepSeek-V3（激活参数）的已知 `(N, D)` 对，比较预测损失与报告损失。

## 生产使用

你不太可能自己训练前沿模型。但扩展律告诉你：

1. **微调是否有足够数据。** 如果任务特定数据低于基础模型每参数 20 个 token，预计会在某个损失下限处达到饱和。
2. **是否选择更大的基础模型。** 如果你把所有预算都花在推理上，优先选择更小、训练更长的模型。
3. **收益递减在哪里。** 超过 Chinchilla 最优 1000 倍后，对数损失变化成为噪声。

**2026 年的研究走向：**

- **数据受限状态。** 网络上的高质量 token 数量有限（过滤后约 5-10 万亿英文 token）。前沿预训练正在接近这一上限。合成数据、多语言、多模态和 RLHF 规模化微调是下一批杠杆。
- **算力乘数技巧。** Muon 优化器、MoE、更好的数据整理——每个都移动绝对常数，不改变渐近线。
- **强化学习的扩展律。** 开放问题。早期证据表明 RL 样本有幂律，但指数与预训练非常不同。

## 上手实践

见 `outputs/skill-training-budget-estimator.md`。该技能根据算力预算、部署约束和目标损失，为新训练运行选择 `(N, D, 小时数, GPU)`。

## 练习

1. **简单。** 运行 `code/main.py`。打印算力预算 `1e20`、`1e22`、`1e24` 的 Chinchilla 最优 `(N, D)`。与真实模型表对比。
2. **中等。** 实现 Hoffmann 损失随算力变化的曲线。在算力最优前沿上绘制损失 vs `log10(C)`。确定该定律预测需要多少 FLOP 才能使交叉熵再降低 0.1。
3. **困难。** 在相同数据集上训练 5 个微型模型（10 万到 1000 万参数），拟合自己的扩展律。估计 `α` 和 `E`。你的指数与已发布的匹配程度如何？

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| 参数（N） | "模型大小" | 非嵌入层的权重数量；决定容量。 |
| Token（D） | "训练数据" | 见过的训练 token 数量；决定参数被利用的程度。 |
| 算力（C） | "花费的 FLOP" | 标准 Transformer 约为 `6 × N × D`。 |
| Chinchilla 最优（Chinchilla-optimal） | "D/N ≈ 20" | 最小化每训练 FLOP 损失的比例。 |
| 超训练（Over-training） | "超过 Chinchilla" | 花费额外训练 FLOP 来节省推理 FLOP；D/N >> 20。 |
| 不可约损失（Irreducible loss） | "下限" | 扩展律中的 `E` 项；数据本身的熵。 |
| 涌现能力（Emergent capability） | "规模下的突然跳跃" | 通常是评分器假象；连续损失是平滑的。 |
| 有效算力（Effective compute） | "训练效率乘数" | 更好的数据/优化器/架构乘以每个 FLOP 的效果。 |

## 延伸阅读

- [Kaplan et al. (2020). Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361) — 第一篇扩展律论文；训练不足
- [Hoffmann et al. (2022). Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556) — Chinchilla
- [Schaeffer et al. (2023). Are Emergent Abilities of Large Language Models a Mirage?](https://arxiv.org/abs/2304.15004) — 涌现作为测量假象
- [Sardana, Frankle (2024). Beyond Chinchilla-Optimal: Accounting for Inference in Language Model Scaling Laws](https://arxiv.org/abs/2401.00448) — 为什么 Llama 的超训练对其工作负载是正确的
- [Jordan et al. (2024). Muon: An optimizer for hidden layers in neural networks](https://kellerjordan.github.io/posts/muon/) — 2 倍算力乘数
