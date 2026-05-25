# 位置编码——正弦式、RoPE、ALiBi

> 注意力对排列是不变的。"猫坐在垫子上"和"垫子上坐猫在"在没有位置信号的情况下产生相同的输出。三种算法修复了这个问题——每种对"位置"的含义做出不同的押注。

**类型：** 构建
**语言：** Python
**前置条件：** Phase 7 · 02（自注意力）、Phase 7 · 03（多头注意力）
**时长：** 约 45 分钟

## 问题背景

缩放点积注意力是顺序盲的。注意力矩阵 `softmax(Q K^T / √d) V` 由成对相似度计算得到。打乱 `X` 的行，输出的行以相同方式打乱。注意力内部没有任何东西关心位置。

对于词袋模型来说这不是 bug。对于语言、代码、音频、视频——任何顺序携带含义的事物——这是致命的。

解决方案是以某种方式将位置注入嵌入。三个时代的答案：

1. **绝对正弦式**（Vaswani 2017）。将位置的 `sin/cos` 加到嵌入上。简单，无需学习，在训练长度之外外推能力差。
2. **RoPE——旋转位置嵌入**（Su 2021）。将 Q 和 K 向量旋转与位置成比例的角度。在点积中直接编码*相对*位置。2026 年主导。
3. **ALiBi——带线性偏差的注意力**（Press 2022）。完全跳过嵌入技巧；根据距离向注意力分数添加每头线性惩罚。出色的长度外推能力。

截至 2026 年，几乎所有前沿开源模型都使用 RoPE：Llama 2/3/4、Qwen 2/3、Mistral、Mixtral、DeepSeek-V3、Kimi。少数长上下文模型使用 ALiBi 或其现代变体。绝对正弦式已成历史。

## 核心概念

![绝对正弦式 vs RoPE 旋转 vs ALiBi 距离偏差](../assets/positional-encoding.svg)

### 绝对正弦式

预计算形状为 `(max_len, d_model)` 的固定矩阵 `PE`：

```
PE[pos, 2i]   = sin(pos / 10000^(2i / d_model))
PE[pos, 2i+1] = cos(pos / 10000^(2i / d_model))
```

然后在注意力之前 `X' = X + PE[:N]`。每个维度是不同频率的正弦波。模型学习从相位模式中读取位置。超过 `max_len` 就失败了：没有告诉模型当它只看过位置 0-2047 时位置 2048 会发生什么。

### RoPE

旋转 Q 和 K 向量（不是嵌入）。对于维度对 `(2i, 2i+1)`：

```
[q'_2i    ]   [ cos(pos·θ_i)  -sin(pos·θ_i) ] [q_2i   ]
[q'_2i+1  ] = [ sin(pos·θ_i)   cos(pos·θ_i) ] [q_2i+1 ]

θ_i = base^(-2i / d_head)，base 默认为 10000
```

对位置 `pos_k` 处的键应用相同旋转。点积 `q'_m · k'_n` 变成仅关于 `(m - n)` 的函数。即：**注意力分数只依赖于相对距离**，即使旋转是以绝对位置为键的。精妙的技巧。

扩展 RoPE：`base` 可以被缩放（NTK 感知、YaRN、LongRoPE），无需重训练即可外推到更长的上下文。Llama 3 通过这种方式将上下文从 8K 扩展到 128K。

### ALiBi

跳过嵌入技巧。直接对注意力分数施加偏置：

```
attn_score[i, j] = (q_i · k_j) / √d  -  m_h · |i - j|
```

其中 `m_h` 是特定头的斜率（例如 `1 / 2^(8·h/H)`）。更近的 token 得到提升；更远的 token 受到惩罚。没有训练成本。论文表明长度外推优于正弦式，在原始训练长度上与 RoPE 匹配。

### 2026 年如何选择

| 变体 | 外推能力 | 训练成本 | 使用者 |
|------|---------|---------|--------|
| 绝对正弦式 | 差 | 免费 | 原始 Transformer、早期 BERT |
| 可学习绝对 | 无 | 极小 | GPT-2、GPT-3 |
| RoPE | 良好（配合缩放） | 免费 | Llama 2/3/4、Qwen 2/3、Mistral、DeepSeek-V3、Kimi |
| RoPE + YaRN | 出色 | 微调阶段 | Qwen2-1M、Llama 3.1 128K |
| ALiBi | 出色 | 免费 | BLOOM、MPT、Baichuan |

RoPE 胜出是因为它无需改变架构即可嵌入注意力，编码相对位置，其 `base` 超参数为长上下文微调提供了清晰的旋钮。

## 动手实现

### 步骤一：正弦式编码

见 `code/main.py`。4 行计算：

```python
def sinusoidal(N, d):
    pe = [[0.0] * d for _ in range(N)]
    for pos in range(N):
        for i in range(d // 2):
            theta = pos / (10000 ** (2 * i / d))
            pe[pos][2 * i]     = math.sin(theta)
            pe[pos][2 * i + 1] = math.cos(theta)
    return pe
```

在第一个注意力层之前将其加到嵌入矩阵上。

### 步骤二：对 Q、K 应用 RoPE

RoPE 在 Q 和 K 上原地操作。对每对维度：

```python
def apply_rope(x, pos, base=10000):
    d = len(x)
    out = list(x)
    for i in range(d // 2):
        theta = pos / (base ** (2 * i / d))
        c, s = math.cos(theta), math.sin(theta)
        a, b = x[2 * i], x[2 * i + 1]
        out[2 * i]     = a * c - b * s
        out[2 * i + 1] = a * s + b * c
    return out
```

关键：对位置 `m` 处的 Q 和位置 `n` 处的 K 应用相同函数。它们的点积在每对坐标上获得 `cos((m-n)·θ_i)` 因子。注意力免费学习相对位置。

### 步骤三：ALiBi 斜率和偏差

```python
def alibi_bias(n_heads, seq_len):
    # slope_h = 2 ** (-8 * h / n_heads) 对于 h = 1..n_heads
    slopes = [2 ** (-8 * (h + 1) / n_heads) for h in range(n_heads)]
    bias = []
    for m in slopes:
        row = [[-m * abs(i - j) for j in range(seq_len)] for i in range(seq_len)]
        bias.append(row)
    return bias  # 在 softmax 之前加到注意力分数上
```

将 `bias[h]` 加到头 `h` 的 `(seq_len, seq_len)` 注意力分数矩阵上，然后执行 softmax。

### 步骤四：验证 RoPE 的相对距离属性

取两个随机向量 `a, b`。分别旋转 `(pos_a, pos_b)`，然后旋转 `(pos_a + k, pos_b + k)`。两次点积必须在浮点误差范围内匹配。这个属性正是 RoPE 的全部意义——它对绝对偏移是不变的，只有相对间隔才重要。

## 生产使用

PyTorch 2.5+ 在 `torch.nn.functional` 中内置了 RoPE 工具。大多数生产代码使用 `flash_attn` 或 `xformers`，RoPE 在注意力核内部应用。

```python
from transformers import AutoModel
model = AutoModel.from_pretrained("meta-llama/Llama-3.2-3B")
# model.config.rope_scaling → {"type": "yarn", "factor": 32.0, "original_max_position_embeddings": 8192}
```

**2026 年的长上下文技巧：**

- **NTK 感知插值。** 当从 4K 扩展到 16K+ 时，将 `base` 重新缩放为 `base * (scale_factor)^(d/(d-2))`。
- **YaRN。** 更智能的插值，保留长上下文上的注意力熵。Llama 3.1 128K 使用它。
- **LongRoPE。** 微软 2024 年的方法，使用进化搜索选择每维度缩放因子。Phi-3-Long 使用它。
- **位置插值 + 微调。** 只需将位置按扩展因子缩小并微调 10-50 亿 token。效果出奇地好。

## 上手实践

见 `outputs/skill-positional-encoding-picker.md`。该技能根据目标上下文长度、外推需求和训练预算，为新模型选择编码策略。

## 练习

1. **简单。** 对 `max_len=512, d=128` 将正弦式 `PE` 矩阵绘制为热力图。确认"随着维度索引增大，条纹变宽"的模式。
2. **中等。** 实现 NTK 感知 RoPE 缩放。在长度 256 的序列上训练一个微型 LM，然后在有和没有缩放的长度 1024 上测试。测量困惑度。
3. **困难。** 在同一个注意力模块中实现 ALiBi 和 RoPE。在长度 512 的复制任务上训练 4 层 Transformer。在测试时外推到 2048。比较退化程度。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| 位置编码（Positional encoding） | "告诉注意力关于顺序" | 任何加到嵌入或注意力中编码位置的信号。 |
| 正弦式（Sinusoidal） | "原始的那种" | 以几何频率的 `sin/cos` 加到嵌入上；不能外推。 |
| RoPE | "旋转嵌入" | 将 Q、K 旋转与位置相关的角度；点积编码相对距离。 |
| ALiBi | "线性偏差技巧" | 向注意力分数加 `-m·|i-j|`；不需要嵌入，外推效果好。 |
| base | "RoPE 的旋钮" | RoPE 中的频率缩放器；在推理时增大以扩展上下文。 |
| NTK 感知 | "一种 RoPE 缩放技巧" | 重新缩放 `base`，使高频维度在上下文扩展时不被压缩。 |
| YaRN | "高级版" | 每维度的插值+外推，保留注意力熵。 |
| 外推（Extrapolation） | "在训练长度之外有效" | 位置方案能否在超过训练中看到的 `max_len` 后提供正确输出？ |

## 延伸阅读

- [Vaswani et al. (2017). Attention Is All You Need §3.5](https://arxiv.org/abs/1706.03762) — 原始正弦式
- [Su et al. (2021). RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864) — RoPE 论文
- [Press, Smith, Lewis (2021). Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation](https://arxiv.org/abs/2108.12409) — ALiBi
- [Peng et al. (2023). YaRN: Efficient Context Window Extension of Large Language Models](https://arxiv.org/abs/2309.00071) — 最先进的 RoPE 缩放
- [Chen et al. (2023). Extending Context Window of Large Language Models via Positional Interpolation](https://arxiv.org/abs/2306.15595) — Meta 的 Llama 2 长上下文论文
- [Ding et al. (2024). LongRoPE: Extending LLM Context Window Beyond 2 Million Tokens](https://arxiv.org/abs/2402.13753) — Phi-3-Long 使用的微软方法
- [HuggingFace Transformers — `modeling_rope_utils.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_rope_utils.py) — 每种 RoPE 缩放方案的生产级实现
