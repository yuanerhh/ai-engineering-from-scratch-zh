# 从零构建 Transformer——综合项目

> 十三节课。一个模型。没有捷径。

**类型：** 构建
**语言：** Python
**前置条件：** Phase 7 · 01 到 13。不要跳过。
**时长：** 约 120 分钟

## 问题背景

你已经读过每篇论文。你已经实现了注意力、多头分割、位置编码、编码器和解码器块、BERT 和 GPT 损失、MoE、KV 缓存。现在让它们在真实任务上协同工作。

综合项目：在字符级语言建模任务上端到端训练一个小型仅解码器 Transformer。它读取莎士比亚。它生成新的莎士比亚风格文本。它足够小，可在笔记本电脑上 10 分钟内训练完成。它足够正确，替换更大的数据集和更长的训练就能得到真正的语言模型。

这是本课程的"nanoGPT"。它并非原创——Karpathy 2023 年的 nanoGPT 教程是每个学生都会写至少一次的参考实现。我们借用其结构，根据我们所学内容重新工具化。

## 核心概念

![从零构建 Transformer 的块图](../assets/capstone.svg)

带注释的架构：

```
输入 token (B, N)
   │
   ▼
token 嵌入 + 位置嵌入  ◀── 第 04 课（RoPE 选项）
   │
   ▼
┌──── 块 × L ────────────────────┐
│  RMSNorm                       │  ◀── 第 05 课
│  多头注意力（因果）              │  ◀── 第 03 + 07 课（因果掩码）
│  残差                          │
│  RMSNorm                       │
│  SwiGLU FFN                    │  ◀── 第 05 课
│  残差                          │
└─────────────────────────────── ┘
   │
   ▼
最终 RMSNorm
   │
   ▼
lm_head（与 token 嵌入绑定）
   │
   ▼
logits (B, N, V)
   │
   ▼
移位一位交叉熵                    ◀── 第 07 课
```

### 我们提供的内容

- `GPTConfig` — 配置所有超参数的单一位置。
- `MultiHeadAttention` — 因果、批量处理，带可选 Flash 风格路径（PyTorch 的 `scaled_dot_product_attention`）。
- `SwiGLUFFN` — 现代 FFN。
- `Block` — 前置归一化、残差包裹的注意力 + FFN。
- `GPT` — 嵌入、堆叠块、LM 头、generate()。
- 带 AdamW、余弦学习率、梯度裁剪的训练循环。
- 莎士比亚文本上的字符级分词器。

### 我们未提供的内容

- RoPE——在第 04 课中概念性实现。这里为简单起见使用可学习位置嵌入。练习要求你换入 RoPE。
- 生成时的 KV 缓存——每个生成步骤对完整前缀重新计算注意力。更慢但更简单。练习要求你添加 KV 缓存。
- Flash Attention——PyTorch 2.0+ 在输入匹配时自动分发；我们使用 `F.scaled_dot_product_attention`。
- MoE——每块单个 FFN。你在第 11 课中看过 MoE。

### 目标指标

在 Mac M2 笔记本电脑上，4 层、4 头、d_model=128 的 GPT 在 `tinyshakespeare.txt` 上训练 2000 步：

- 训练损失从约 4.2（随机）在约 6 分钟内收敛到约 1.5。
- 采样输出看起来有莎士比亚风格：古语词汇、换行、"ROMEO:" 等专有名词涌现。
- 验证损失（最后 10% 的文本作为保留集）紧跟训练损失；在这个规模/预算下没有过拟合。

## 动手实现

本课使用 PyTorch。安装 `torch`（CPU 版本即可）。见 `code/main.py`。脚本处理：

- 如果缺少则下载 `tinyshakespeare.txt`（或读取本地副本）。
- 字节级字符分词器。
- 90/10 的训练/验证分割。
- 在支持的硬件上使用 bf16 自动转换的训练循环。
- 训练完成后进行采样。

### 步骤一：数据

```python
text = open("tinyshakespeare.txt").read()
chars = sorted(set(text))
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for c, i in stoi.items()}
encode = lambda s: [stoi[c] for c in s]
decode = lambda xs: "".join(itos[x] for x in xs)
```

65 个独特字符。微型词汇表。适合 4 字节 vocab_size。无 BPE，无分词器困扰。

### 步骤二：模型

见 `code/main.py`。块是第 05 课的教科书内容——前置归一化、RMSNorm、SwiGLU、因果 MHA。4/4/128 的参数量：约 80 万。

### 步骤三：训练循环

获取随机批次的长度为 256 的 token 窗口。前向传播。移位一位交叉熵。反向传播。AdamW 步进。记录。重复。

```python
for step in range(max_steps):
    x, y = get_batch("train")
    logits = model(x)
    loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    opt.zero_grad()
```

### 步骤四：采样

给定一个提示，反复前向传播，从 top-p logit 中采样，追加，继续。在 500 个 token 后停止。

### 步骤五：读取输出

2000 步之后：

```
ROMEO:
Away and mild will not thy friend, that thou shalt wit:
The chief that well shame and hath been his friends,
...
```

不是莎士比亚，但有莎士比亚风格。约 80 万参数、笔记本电脑上 6 分钟——明确的胜利。

## 生产使用

这个综合项目是一个参考架构。三个扩展将其推向真实产品：

1. **替换分词器。** 使用 BPE（如 `tiktoken.get_encoding("cl100k_base")`）。词汇表大小从 65 跳升至约 5 万。模型容量需要相应扩展。
2. **在更大的语料上训练。** 使用 `OpenWebText` 或 `fineweb-edu`（HuggingFace）。在单个 A100 上 100 亿 token，1.25 亿参数的 GPT 约需 24 小时。
3. **添加 RoPE + KV 缓存 + Flash Attention。** 下方练习逐步引导你完成每项。

最终得到一个 1.25 亿参数的 GPT，能流畅生成英文。不是前沿模型，但相同的代码路径——只是更大——正是 Karpathy、EleutherAI 和 Allen Institute 在 2026 年用来训练研究检查点的。

## 上手实践

见 `outputs/skill-transformer-review.md`。该技能根据前 13 课的所有内容审查从零构建的 Transformer 实现是否正确。

## 练习

1. **简单。** 运行 `code/main.py`。验证训练后模型的最终步骤验证损失在 2.0 以下。将 `max_steps` 从 2000 改为 5000——验证损失会继续改善吗？
2. **中等。** 将可学习位置嵌入替换为 RoPE。在 `MultiHeadAttention` 内部对 Q 和 K 应用旋转。训练并验证验证损失至少同样低。
3. **中等。** 在采样循环中实现 KV 缓存。在有和没有缓存的情况下生成 500 个 token。笔记本电脑上实际速度应提升 5-20×。
4. **困难。** 为模型添加第二个头，预测下下一个 token（MTP——DeepSeek-V3 的多 token 预测）。联合训练。有帮助吗？
5. **困难。** 将每块的单个 FFN 替换为 4 专家 MoE。路由器 + top-2 路由。观察在相同激活参数下验证损失如何变化。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| nanoGPT | "Karpathy 的教程仓库" | 最小化的仅解码器 Transformer 训练代码，约 300 行；规范参考。 |
| tinyshakespeare | "标准玩具语料" | 约 1.1 MB 文本；2015 年以来每个字符 LM 教程都使用它。 |
| 绑定嵌入（Tied embeddings） | "共享输入/输出矩阵" | LM 头权重 = token 嵌入矩阵的转置；节省参数，提升质量。 |
| bf16 自动转换（bf16 autocast） | "训练精度技巧" | 前向/反向以 bf16 运行，优化器状态保持 fp32；2021 年以来的标准。 |
| 梯度裁剪（Gradient clipping） | "防止峰值" | 将全局梯度范数上限设为 1.0；防止训练爆炸。 |
| 余弦学习率调度（Cosine LR schedule） | "2020 年以来的默认" | 学习率线性预热后余弦衰减至峰值的 10%。 |
| MFU | "模型 FLOP 利用率" | 实际 FLOP / 理论峰值；2026 年密集 40%、MoE 30% 算强。 |
| 验证损失（Val loss） | "保留集损失" | 模型从未见过的数据上的交叉熵；过拟合检测器。 |

## 延伸阅读

- [The Annotated Transformer (Harvard NLP)](https://nlp.seas.harvard.edu/annotated-transformer/) — 经典带注释实现
