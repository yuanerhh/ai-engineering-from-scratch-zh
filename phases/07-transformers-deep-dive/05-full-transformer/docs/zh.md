# 完整 Transformer——编码器 + 解码器

> 注意力是主角。其余一切——残差、归一化、前馈、交叉注意力——都是让你深度堆叠的脚手架。

**类型：** 构建
**语言：** Python
**前置条件：** Phase 7 · 02（自注意力）、Phase 7 · 03（多头注意力）、Phase 7 · 04（位置编码）
**时长：** 约 75 分钟

## 问题背景

单个注意力层是一个特征提取器，而非完整模型。每层一次矩阵乘法对于语言来说容量不足。你需要深度——但没有正确的"管道"，深度会崩溃。

2017 年 Vaswani 论文封装了六个设计决策，将单层注意力变成可堆叠的块。此后每个 Transformer——仅编码器（BERT）、仅解码器（GPT）、编码器-解码器（T5）——都继承了同样的骨架。2026 年的块已被改进（RMSNorm、SwiGLU、前置归一化、RoPE），但骨架完全相同。

本课讲骨架。后续课程进行专化——第 06 课讲编码器，第 07 课讲解码器，第 08 课讲编码器-解码器。

## 核心概念

![编码器和解码器块内部结构及连线](../assets/full-transformer.svg)

### 六大组件

1. **嵌入 + 位置信号。** Token → 向量。位置通过 RoPE（现代）或正弦式（经典）注入。
2. **自注意力（Self-attention）。** 每个位置与其他每个位置进行注意力计算。解码器中有掩码。
3. **前馈网络（FFN，Feed-Forward Network）。** 逐位置的两层 MLP：`W_2 · activation(W_1 · x)`。默认扩展比为 4×。
4. **残差连接（Residual connection）。** `x + sublayer(x)`。没有残差，梯度在约 6 层后消失。
5. **层归一化（Layer normalization）。** `LayerNorm` 或 `RMSNorm`（现代）。稳定残差流。
6. **交叉注意力（Cross-attention，仅解码器）。** 查询来自解码器，键和值来自编码器输出。

### 编码器块（BERT、T5 编码器使用）

```
x → LN → MHA(self) → + → LN → FFN → + → out
                     ^              ^
                     |              |
                     └── 残差 ──────┘
```

编码器是双向的，无掩码。所有位置看到所有位置。

### 解码器块（GPT、T5 解码器使用）

```
x → LN → MHA(掩码 self) → + → LN → MHA(交叉到编码器) → + → LN → FFN → + → out
```

解码器每块有三个子层。中间那个——交叉注意力——是信息从编码器流向解码器的唯一通道。在纯仅解码器架构（GPT）中，交叉注意力被省略，只有掩码自注意力 + FFN。

### 前置归一化 vs 后置归一化

原始论文：`x + sublayer(LN(x))` vs `LN(x + sublayer(x))`。后置归一化在 2019 年前后失去青睐——在没有仔细预热的情况下难以深度训练。前置归一化（`LN` 在子层*之前*）是 2026 年的默认方案：Llama、Qwen、GPT-3+、Mistral 均使用它。

### 2026 年现代化块

Vaswani 2017 搭载了 LayerNorm + ReLU。现代栈替换了两者。生产块的实际样貌：

| 组件 | 2017 | 2026 |
|------|------|------|
| 归一化 | LayerNorm | RMSNorm |
| FFN 激活 | ReLU | SwiGLU |
| FFN 扩展 | 4× | 2.6×（SwiGLU 使用三个矩阵，总参数量相当） |
| 位置 | 正弦式绝对位置 | RoPE |
| 注意力 | 完整 MHA | GQA（或 MLA） |
| 偏置项 | 有 | 无 |

RMSNorm 去掉了 LayerNorm 的均值中心化（少一次减法），节省计算，经验上同样稳定。SwiGLU（`Swish(W1 x) ⊙ W3 x`）在 Llama、PaLM 和 Qwen 的论文中持续以约 0.5 点困惑度优于 ReLU/GELU FFN。

### 参数量

对于一个 `d_model = d`、FFN 扩展比为 `r` 的单块：

- MHA：`4 · d²`（Q、K、V、O 投影）
- FFN（SwiGLU）：`3 · d · (r · d)` ≈ `3rd²`
- 归一化：可忽略

在 `d = 4096, r = 2.6, layers = 32`（大致 Llama 3 8B）时，总计：`32 · (4·4096² + 3·2.6·4096²) ≈ 32 · (16 + 32) M = ~每层 1.5B 参数 × 32 ≈ 7B`（加上嵌入和头部）。与公布数字吻合。

## 动手实现

### 步骤一：基础构建块

使用第 03 课的微型 `Matrix` 类（已复制到本文件以保持独立性）：

- `layer_norm(x, eps=1e-5)` — 减去均值，除以标准差。
- `rms_norm(x, eps=1e-6)` — 除以 RMS。无均值减法。
- `gelu(x)` 和 `silu(x) * W3 x`（SwiGLU）。
- `ffn_swiglu(x, W1, W2, W3)`。
- `encoder_block(x, params)` 和 `decoder_block(x, enc_out, params)`。

完整连线见 `code/main.py`。

### 步骤二：连线 2 层编码器和 2 层解码器

堆叠它们。将编码器输出传入每个解码器交叉注意力。在输出投影前添加最终 LN。

```python
def encode(tokens, params):
    x = embed(tokens, params.emb) + sinusoidal(len(tokens), params.d)
    for block in params.encoder_blocks:
        x = encoder_block(x, block)
    return x

def decode(target_tokens, encoder_out, params):
    x = embed(target_tokens, params.emb) + sinusoidal(len(target_tokens), params.d)
    for block in params.decoder_blocks:
        x = decoder_block(x, encoder_out, block)
    return x
```

### 步骤三：在玩具示例上运行前向传播

将 6 个 token 的源序列和 5 个 token 的目标序列传入。验证输出形状为 `(5, vocab)`。无需训练——本课关注架构，而非损失。

### 步骤四：换用 RMSNorm + SwiGLU

将 LayerNorm 和 ReLU-FFN 替换为 RMSNorm 和 SwiGLU。确认形状仍然匹配。这是仅需一次函数替换的 2026 年现代化改造。

## 生产使用

PyTorch/TF 参考实现：`nn.TransformerEncoderLayer`、`nn.TransformerDecoderLayer`。但大多数 2026 年的生产代码会自己实现块，原因是：

- Flash Attention 在注意力内部调用，而非通过 `nn.MultiheadAttention`。
- GQA / MLA 不在标准库参考中。
- RoPE、RMSNorm、SwiGLU 不是 PyTorch 的默认实现。

HF `transformers` 有干净的参考块值得阅读：`modeling_llama.py` 是 2026 年规范的仅解码器块。约 500 行，值得通读一遍。

**编码器 vs 解码器 vs 编码器-解码器——何时选择：**

| 需求 | 选择 | 示例 |
|------|------|------|
| 分类、嵌入、文本问答 | 仅编码器 | BERT、DeBERTa、ModernBERT |
| 文本生成、对话、代码、推理 | 仅解码器 | GPT、Llama、Claude、Qwen |
| 结构化输入 → 结构化输出（翻译、摘要） | 编码器-解码器 | T5、BART、Whisper |

仅解码器赢得了语言任务，因为它扩展最简洁，同时处理理解和生成。编码器-解码器在输入有明确"源序列"身份（翻译、语音识别、结构化任务）时仍是最优方案。

## 上手实践

见 `outputs/skill-transformer-block-reviewer.md`。该技能根据 2026 年默认标准审查新的 Transformer 块实现，并标出缺失的部分（前置归一化、RoPE、RMSNorm、GQA、FFN 扩展比）。

## 练习

1. **简单。** 计算 `d_model=512, n_heads=8, ffn_expansion=4, swiglu=True` 时 encoder_block 的参数量。通过实现该块并使用 `sum(p.numel() for p in block.parameters())` 验证。
2. **中等。** 从后置归一化切换到前置归一化。初始化两者，在随机输入上经过 12 层堆叠后测量激活范数。后置归一化的激活应该爆炸；前置归一化的应该保持有界。
3. **困难。** 在玩具复制任务（将 `x` 反转复制）上实现 4 层编码器-解码器。训练 100 步，报告损失。换用 RMSNorm + SwiGLU + RoPE——损失有所下降吗？

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| 块（Block） | "一个 Transformer 层" | 归一化 + 注意力 + 归一化 + FFN 的堆叠，包裹在残差连接中。 |
| 残差（Residual） | "跳跃连接" | `x + f(x)` 输出；使梯度能够在深层栈中流动。 |
| 前置归一化（Pre-norm） | "先归一化，再传入子层" | 现代：`x + sublayer(LN(x))`。无需预热即可深度训练。 |
| RMSNorm | "没有均值的 LayerNorm" | 除以 RMS；少一次操作，经验稳定性相同。 |
| SwiGLU | "大家都切换的 FFN" | `Swish(W1 x) ⊙ W3 x → W2`。在 LM 困惑度上优于 ReLU/GELU。 |
| 交叉注意力（Cross-attention） | "解码器看编码器的方式" | Q 来自解码器、K/V 来自编码器输出的 MHA。 |
| FFN 扩展（FFN expansion） | "中间 MLP 有多宽" | 隐藏层宽度与 d_model 之比，通常 4（LayerNorm）或 2.6（SwiGLU）。 |
| 无偏置（Bias-free） | "去掉 +b 项" | 现代栈省略线性层中的偏置；略微提升困惑度，模型更小。 |

## 延伸阅读

- [Vaswani et al. (2017). Attention Is All You Need](https://arxiv.org/abs/1706.03762) — 原始块规格
- [Xiong et al. (2020). On Layer Normalization in the Transformer Architecture](https://arxiv.org/abs/2002.04745) — 为什么前置归一化在深度上优于后置归一化
- [Zhang, Sennrich (2019). Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467) — RMSNorm
- [Shazeer (2020). GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202) — SwiGLU 论文
- [HuggingFace `modeling_llama.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py) — 2026 年规范仅解码器块
