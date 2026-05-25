# 为什么是 Transformer——RNN 的问题

> RNN 逐个处理 token。Transformer 同时处理所有 token。这一单一的架构押注改变了 2017 年后深度学习的所有扩展曲线。

**类型：** 学习
**语言：** Python
**前置条件：** Phase 3（深度学习核心）、Phase 5 · 09（序列到序列）、Phase 5 · 10（注意力机制）
**时长：** 约 45 分钟

## 问题背景

2017 年之前，地球上每个最先进的序列模型——语言、翻译、语音——都是循环神经网络（RNN）。LSTM 和 GRU 在 ImageNet 级别的翻译基准上独领风骚长达五年。它们是每个人手中唯一的工具。

它们有三个致命弱点。顺序计算意味着你无法沿时间轴并行化：token `t+1` 需要来自 token `t` 的隐藏状态。1024 个 token 的序列意味着在每个时钟周期可以执行 100 万次浮点运算的 GPU 上做 1024 个串行步骤。在为并行设计的硬件上，训练墙钟时间与序列长度线性增长。

梯度消失意味着 50 个 token 之前的信息已经被 50 个非线性压缩处理过了。门控循环单元（LSTM、GRU）软化了这种压缩，但从未消除。长程依赖——"我去年夏天在飞往京都的飞机上读的那本书……"——经常失败。

固定宽度隐藏状态意味着编码器在解码器看到任何内容之前，将整个源序列压缩成单一向量。无论源序列是 5 个 token 还是 500 个，瓶颈的形状是一样的。

2017 年的论文《Attention Is All You Need》提出了激进的方案：完全放弃循环。让每个位置并行地与其他每个位置进行注意力计算。用一次大型矩阵乘法训练，而不是 1024 个串行矩阵乘法。

到 2026 年，这一结果主导了每种模态。语言（GPT-5、Claude 4、Llama 4），视觉（ViT、DINOv2、SAM 3），音频（Whisper），生物学（AlphaFold 3），机器人学（RT-2）。相同的块，不同的输入。

## 核心概念

![RNN 串行计算 vs Transformer 并行注意力](../assets/rnn-vs-transformer.svg)

**循环作为瓶颈。** RNN 计算 `h_t = f(h_{t-1}, x_t)`。每一步依赖于前一步。在计算 `h_4` 之前无法计算 `h_5`。在拥有 10000+ 并行核心的现代 GPU 上，这在长序列上浪费了 99% 的硅片。

**注意力作为广播。** 自注意力为每一对 `(i, j)` 同时计算 `output_i = sum_j(a_ij * v_j)`。整个 N×N 注意力矩阵在一次批量矩阵乘法中填充完成。没有任何步骤依赖于另一步骤。GPU 喜欢这种方式。

**加速不是常数倍。** 它是 `O(N)` 串行深度和 `O(1)` 串行深度之间的差异。实际上，在 N=512 的匹配硬件上，Transformer 每轮训练比 RNN 快 5-10 倍，随着序列长度增加差距不断扩大，直到遇到注意力的 `O(N²)` 内存壁（Flash Attention 后来修复了这个问题——见第 12 课）。

**Transformer 的代价。** 注意力内存以 `O(N²)` 扩展。2K 上下文没问题。128K 上下文就需要滑动窗口、RoPE 外推、Flash Attention 分块或线性注意力变体。循环在时间和内存上都是 `O(N)`；Transformer 用内存换时间，然后通过并行性赢回时间。

**归纳偏置的转变。** RNN 假设局部性和近期性。Transformer 什么都不假设——每一对都是注意力的候选。这就是为什么 Transformer 需要更多数据才能训练好，但一旦拥有足够数据就能扩展得更远。Chinchilla（2022）将这一点形式化：给定足够的 token，在参数量相同的情况下 Transformer 总是优于 RNN。

## 动手实现

这里没有神经网络——我们在数值上模拟核心瓶颈，让你在笔记本上感受到差距。

### 步骤一：测量串行深度

见 `code/main.py`。我们构建两个函数。一个将序列编码为加法链（串行，像 RNN）。一个将其编码为并行规约（广播，像注意力）。相同的数学，不同的依赖图。

```python
def rnn_style(xs):
    h = 0.0
    for x in xs:
        h = 0.9 * h + x   # 无法并行化：h 依赖于前一个 h
    return h

def attention_style(xs):
    return sum(xs) / len(xs)  # 每个 x 都是独立的
```

我们在长达 100,000 个元素的序列上为两者计时。RNN 版本是 O(N) 且在单个 CPU 流水线上运行。即使在纯 Python 中，注意力风格的规约在长度 ≥ 1,000 时也能胜出，因为 Python 的 `sum()` 在 C 中实现，每步无解释器开销。

### 步骤二：计算理论操作数

两种算法都做 N 次加法。区别在于*依赖深度*：在下一步可以开始之前必须顺序发生多少操作。RNN 深度 = N。注意力深度 = 树规约时 log(N)，并行扫描时 = 1。深度而非操作数决定 GPU 时间。

### 步骤三：长序列上的经验扩展

我们打印一个使 O(N) 差距可见的计时表。在 2026 年的 Mac 笔记本上，1000 个元素以下的序列太快无法测量。100,000 个元素的序列显示清晰的线性扫描。将其扩展到带有 12 层等效 LSTM 的 16,384 token Transformer，你就能看到为什么 2016 年训练墙钟时间是一个瓶颈。

## 生产使用

2026 年什么时候还应该选 RNN：

| 场景 | 选择 |
|------|------|
| 流式推理，逐个 token，恒定内存 | RNN 或状态空间模型（Mamba、RWKV） |
| 超长序列（>100 万 token），注意力内存爆炸 | 线性注意力、Mamba 2、Hyena |
| 无矩阵乘法加速器的边缘设备 | 深度可分离 RNN 在 FLOPs/瓦特上仍胜出 |
| 其他所有情况（训练、批量推理、最高 128K 上下文） | Transformer |

像 Mamba 这样的状态空间模型（SSM）本质上是具有结构化参数化的 RNN，使它们兼具两者优势：`O(N)` 扫描内存，通过选择性扫描进行并行训练。它们以更好的长上下文扩展性恢复了 90% 的 Transformer 质量。2026 年大多数前沿实验室训练混合 SSM+Transformer 模型（例如 Jamba、Samba）——循环并未消亡，它是一个组件。

## 上手实践

见 `outputs/skill-architecture-picker.md`。该技能根据长度、吞吐量和训练预算约束，为新的序列问题选择架构。它应该始终拒绝在不说明权衡的情况下为超过 10 亿 token 的训练运行推荐纯 RNN。

## 练习

1. **简单。** 取 `code/main.py` 中的 `rnn_style`，将标量隐藏状态替换为长度为 64 的向量隐藏状态。重新测量。串行开销随隐藏状态维度增长多少？
2. **中等。** 用纯 Python 实现并行前缀和（Hillis-Steele 扫描）。验证它在长度 1024 上与串行扫描产生相同数值输出。计算深度。
3. **困难。** 将注意力风格的规约移植到 GPU 上的 PyTorch。当序列长度从 64 扫到 65,536 时，对两者计时。绘图并解释曲线形状。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| 循环（Recurrence） | "RNN 是顺序的" | 步骤 `t` 依赖于步骤 `t-1` 的计算，强制沿时间轴串行执行。 |
| 串行深度（Serial depth） | "图有多深" | 最长的依赖操作链；即使在无限硬件上也限制了墙钟时间。 |
| 注意力（Attention） | "让 token 互相查看" | 加权和 `sum_j a_ij v_j`，其中 `a_ij` 来自位置 i 和 j 之间的相似度分数。 |
| 上下文窗口（Context window） | "模型能看多少" | 注意力层可以作为输入的位置数；二次内存成本在此处扩展。 |
| 归纳偏置（Inductive bias） | "架构内置的假设" | 关于数据形态的先验；CNN 假设平移不变性，RNN 假设近期性。 |
| 状态空间模型（State-space model） | "有代数支撑的 RNN" | 通过结构化状态空间矩阵参数化以进行并行训练的循环网络。 |
| 二次瓶颈（Quadratic bottleneck） | "为什么上下文这么贵" | 注意力内存 = 序列长度的 `O(N²)`；Flash Attention 隐藏了常数，但不是扩展规律。 |

## 延伸阅读

- [Vaswani et al. (2017). Attention Is All You Need](https://arxiv.org/abs/1706.03762) — 终结主流 NLP 中循环的论文
- [Bahdanau, Cho, Bengio (2014). Neural MT by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473) — 注意力的诞生地，被附加到 RNN 上
- [Hochreiter, Schmidhuber (1997). Long Short-Term Memory](https://www.bioinf.jku.at/publications/older/2604.pdf) — 原始 LSTM 论文，留作记录
- [Gu, Dao (2023). Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752) — 对 Transformer 的现代循环答案
