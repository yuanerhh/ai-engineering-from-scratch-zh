# GPT——因果语言建模

> BERT 看两侧。GPT 只看过去。三角掩码是现代 AI 中最具影响力的一行代码。

**类型：** 构建
**语言：** Python
**前置条件：** Phase 7 · 02（自注意力）、Phase 7 · 05（完整 Transformer）、Phase 7 · 06（BERT）
**时长：** 约 75 分钟

## 问题背景

语言模型回答一个问题：给定前 `t-1` 个 token，token `t` 的概率分布是什么？在这个信号上训练——下一个 token 预测——你就能得到一个可以逐个 token 生成任意文本的模型。

要并行地在整个序列上端到端训练它，每个位置的预测必须只依赖于更早的位置。否则模型就会通过偷看答案而轻易作弊。

因果掩码（causal mask）做到了这一点。它是一个上三角的 `-inf` 值矩阵，在 softmax 之前加到注意力分数上。softmax 后，那些位置变为 0。每个位置只能关注自身和更早的位置。而且由于一次应用于整个序列，一次前向传播就能得到 N 个并行的下一个 token 预测。

GPT-1（2018）、GPT-2（2019）、GPT-3（2020）、GPT-4（2023）、GPT-5（2024）、Claude、Llama、Qwen、Mistral、DeepSeek、Kimi——它们都是具有同样核心循环的仅解码器因果 Transformer。只是更大、更好的数据和更好的 RLHF。

## 核心概念

![因果掩码产生三角形注意力矩阵](../assets/causal-attention.svg)

### 掩码

给定长度为 `N` 的序列，构建一个 `N × N` 矩阵：

```
M[i, j] = 0       如果 j <= i
M[i, j] = -inf    如果 j > i
```

在 softmax 之前将 `M` 加到原始注意力分数上。`exp(-inf) = 0`，因此被掩码位置的权重贡献为零。注意力矩阵的每一行只是之前位置上的概率分布。

实现成本：一次 `torch.tril()` 调用。计算时间：纳秒级。对该领域的影响：一切。

### 并行训练，串行推理

**训练：** 一次前向传播整个 `(N, d_model)` 序列，计算 N 个交叉熵损失（每个位置一个），求和，反向传播。沿序列并行。这就是 GPT 训练可以扩展的原因——一次 GPU 传播处理一批 100 万 token。

**推理：** 逐个 token 生成。输入 `[t1, t2, t3]`，得到 `t4`。输入 `[t1, t2, t3, t4]`，得到 `t5`。输入 `[t1, t2, t3, t4, t5]`，得到 `t6`。KV 缓存（第 12 课）保存 `t1…tn` 的隐藏状态，避免每步重新计算。但推理时的串行深度 = 输出长度。这是自回归税，也是每个 LLM 延迟瓶颈所在。

### 损失——移位一位

给定 token `[t1, t2, t3, t4]`：

- 输入：`[t1, t2, t3]`
- 目标：`[t2, t3, t4]`

对每个位置 `i`，计算 `-log P(target_i | inputs[:i+1])`。求和。这是整个序列的交叉熵。

你听过的每个 Transformer 语言模型都在这个损失上训练。预训练、微调、SFT——同一损失，不同数据。

### 解码策略

训练后，采样选择比人们认为的更重要。

| 方法 | 作用 | 使用时机 |
|------|------|---------|
| 贪婪（Greedy） | 每步取 argmax | 确定性任务、代码补全 |
| 温度（Temperature） | 将 logit 除以 T 后采样 | 创意任务，T 越高多样性越强 |
| Top-k | 只从前 k 个 token 中采样 | 消除低概率长尾 |
| Top-p（核采样） | 从累积概率 ≥ p 的最小集合中采样 | 2020 年以来的默认；适应分布形状 |
| Min-p | 保留 `p > min_p * max_p` 的 token | 2024 年以来；比 top-p 更好地拒绝长尾 |
| 投机解码（Speculative decoding） | 草稿模型提出 N 个 token，大模型验证 | 相同质量下延迟降低 2-3× |

2026 年，min-p + 温度 0.7 是开放权重模型的合理默认值。投机解码是任何生产推理栈的基本要求。

### 是什么让"GPT 秘方"奏效

1. **仅解码器。** 无编码器开销。每层一次注意力 + FFN 传播。
2. **规模扩展。** 1.24 亿 → 15 亿 → 1750 亿 → 万亿。Chinchilla 扩展律（第 13 课）告诉你如何分配算力。
3. **上下文学习（In-context learning）。** 在约 60 亿-130 亿参数时涌现。模型无需微调即可遵循少样本示例。
4. **RLHF。** 基于人类偏好的后训练将原始预训练文本转化为对话助手。
5. **前置归一化 + RoPE + SwiGLU。** 大规模下的稳定训练。

核心架构自 GPT-2 以来变化不大。所有有趣的事情都发生在数据、规模和后训练上。

## 动手实现

### 步骤一：因果掩码

见 `code/main.py`。一行代码：

```python
def causal_mask(n):
    return [[0.0 if j <= i else float("-inf") for j in range(n)] for i in range(n)]
```

在 softmax 之前将其加到注意力分数上。这就是整个机制。

### 步骤二：类 GPT 的 2 层模型

堆叠两个解码器块（掩码自注意力 + FFN，无交叉注意力）。添加 token 嵌入、位置编码和反嵌入（与 token 嵌入矩阵绑定——GPT-2 以来的标准技巧）。

### 步骤三：端到端下一个 token 预测

在 20 token 的玩具词汇表上，在每个位置生成 logit。计算与移位一位目标的交叉熵损失。无梯度——这是前向传播的健全性检查。

### 步骤四：采样

实现贪婪、温度、top-k、top-p、min-p。在固定提示上运行每种方法并比较输出。采样函数只需 10 行代码。

## 生产使用

PyTorch，2026 年用法：

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")
tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")

prompt = "Attention is all you need because"
inputs = tok(prompt, return_tensors="pt")
out = model.generate(
    **inputs,
    max_new_tokens=64,
    temperature=0.7,
    top_p=0.9,
    do_sample=True,
)
print(tok.decode(out[0]))
```

在底层，`generate()` 运行前向传播，提取最后位置的 logit，采样下一个 token，追加，重复。每个生产 LLM 推理栈（vLLM、TensorRT-LLM、llama.cpp、Ollama、MLX）都以大量优化实现同样的循环——批量预填充、连续批处理、KV 缓存分页、投机解码。

**GPT vs BERT，各一行：** GPT 预测 `P(x_t | x_{<t})`。BERT 预测 `P(x_masked | x_unmasked)`。损失决定了模型是否能生成。

## 上手实践

见 `outputs/skill-sampling-tuner.md`。该技能为新的生成任务选择采样参数，并标出何时需要确定性解码。

## 练习

1. **简单。** 运行 `code/main.py`，验证因果注意力矩阵在 softmax 后是下三角的。抽查：第 3 行应只在第 0-3 列有权重。
2. **中等。** 实现宽度为 4 的束搜索（beam search）。在 10 个短提示上比较 beam-4 与贪婪解码的困惑度。束搜索总是更好吗？（提示：通常对翻译是这样，但对开放式对话不是。）
3. **困难。** 实现投机解码：使用 2 层微型模型作为草稿，6 层模型作为验证器。测量 100 个长度为 64 的补全的实际加速比。确认输出与验证器的贪婪解码匹配。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| 因果掩码（Causal mask） | "那个三角形" | 加到注意力分数上的上三角 `-inf` 矩阵，使位置 `i` 只能看到位置 `≤ i`。 |
| 下一个 token 预测（Next-token prediction） | "损失函数" | 每个位置上模型分布对真实下一个 token 的交叉熵。 |
| 自回归（Autoregressive） | "一次生成一个" | 将输出送回作为输入；训练时并行，生成时不并行。 |
| Logit | "softmax 前的分数" | LM 头在 softmax 前的原始输出；采样在这上面进行。 |
| 温度（Temperature） | "创意旋钮" | 将 logit 除以 T；T→0 = 贪婪，T→∞ = 均匀分布。 |
| Top-p | "核采样" | 截断分布到总和 ≥p 的最小集合；从剩余部分采样。 |
| Min-p | "比 top-p 更好" | 保留 `p ≥ min_p × max_p` 的 token；根据分布的尖锐程度自适应截断。 |
| 投机解码（Speculative decoding） | "草稿 + 验证" | 廉价模型提出 N 个 token；大模型并行验证。 |
| 教师强制（Teacher forcing） | "训练技巧" | 训练时输入真实的前一个 token 而非模型的预测。每个 seq2seq LM 的标准做法。 |

## 延伸阅读

- [Radford et al. (2018). Improving Language Understanding by Generative Pre-Training](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf) — GPT-1
- [Radford et al. (2019). Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) — GPT-2
- [Brown et al. (2020). Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165) — GPT-3 与上下文学习
- [Leviathan, Kalman, Matias (2023). Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192) — 投机解码论文
- [HuggingFace `modeling_llama.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py) — 规范因果 LM 参考代码
