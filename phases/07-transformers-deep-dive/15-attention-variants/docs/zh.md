# 注意力变体——滑动窗口、稀疏、差分

> 完整注意力是一个圆。每个 token 看到每个 token，内存付出代价。四种变体改变圆的形状，恢复一半的成本。

**类型：** 构建
**语言：** Python
**前置条件：** Phase 7 · 02（自注意力）、Phase 7 · 03（多头注意力）、Phase 7 · 12（KV 缓存 / Flash Attention）
**时长：** 约 60 分钟

## 问题背景

完整注意力在序列长度上的内存和计算成本均为 `O(N²)`。对于 128K 上下文的 Llama 3 70B，每层有 160 亿个注意力条目，乘以 80 层。Flash Attention（第 12 课）隐藏了 `O(N²)` 的激活内存，但不改变算术成本——每个 token 仍然关注每个其他 token。

三类变体改变注意力矩阵本身的拓扑：

1. **滑动窗口注意力（SWA，Sliding Window Attention）。** 每个 token 只关注固定窗口内的邻居，而非完整前缀。内存和计算降至 `O(N · W)`，其中 `W` 是窗口大小。Gemma 2/3、Mistral 7B 的前几层、Phi-3-Long。
2. **稀疏 / 块注意力。** 只对选定的 `(i, j)` 对打分；其余强制为零权重。Longformer、BigBird、OpenAI 稀疏 Transformer。
3. **差分注意力。** 使用独立的 Q/K 投影计算两个注意力图，然后相减。消除将权重渗透到前几个 token 的"注意力汇聚"。微软 DIFF Transformer（2024）。

它们可以共存。2026 年的前沿模型通常混合使用：大多数层是 SWA-1024，每五层是全局完整注意力，还有少数差分头用于清理检索。Gemma 3 的 5:1 SWA 与全局比是当前的教科书默认方案。

## 核心概念

### 滑动窗口注意力（SWA）

位置 `i` 处的每个查询只关注 `[i - W, i]`（因果 SWA）或 `[i - W/2, i + W/2]`（双向）中的位置。窗口外的 token 在分数矩阵中得到 `-inf`。

```
完整因果：               滑动窗口（W=4）：
位置 0-7               位置 0-7，W=4
    0 1 2 3 4 5 6 7        0 1 2 3 4 5 6 7
0 | x                0 |  x
1 | x x              1 |  x x
2 | x x x            2 |  x x x
3 | x x x x          3 |  x x x x
4 | x x x x x        4 |    x x x x
5 | x x x x x x      5 |      x x x x
6 | x x x x x x x    6 |        x x x x
7 | x x x x x x x x  7 |          x x x x
```

对于 `N = 8192` 和 `W = 1024`，分数矩阵预期有 1024 × 8192 个非零行——减少 8 倍。

**KV 缓存随 SWA 缩小。** 每层只需保留最后 `W` 个 token 的 K 和 V。对于 Gemma-3 类配置（1024 窗口，128K 上下文），KV 缓存缩小 128 倍。

**质量代价。** 仅有 SWA 的 Transformer 在长程检索上有困难。修复方案：SWA 层与完整注意力层交错。Gemma 3 使用 5:1 的 SWA:全局比例。Mistral 7B 使用因果 SWA 栈，信息通过重叠窗口"向前流动"——每层将有效感受野扩展 `W`，L 层后模型可以关注 `L × W` 个 token。

### 稀疏 / 块注意力

提前选定 `N × N` 的稀疏模式。三种典型形状：

- **局部 + 步进（OpenAI 稀疏 Transformer）。** 关注最后 `W` 个 token 加上每隔 `stride` 个 token 的历史。同时捕获局部和长程，计算量为 `O(N · sqrt(N))`。
- **Longformer / BigBird。** 局部窗口 + 少量全局 token（如 `[CLS]`，关注所有人也被所有人关注）+ 随机稀疏链接。在相同质量下实现约 2 倍上下文。
- **原生稀疏注意力（DeepSeek，2025）。** 学习哪些 `(Q, K)` 块重要；在核级别跳过零块。Flash Attention 兼容。

稀疏注意力是一个核工程故事。数学简单（遮蔽分数矩阵）；优势来自永不将零条目加载到 SRAM。Flash Attention-3 和 2026 年的 FlexAttention API 使自定义稀疏模式在 PyTorch 中成为一等公民。

### 差分注意力（DIFF Transformer，2024）

常规注意力有"注意力汇聚"问题：softmax 迫使每行总和为 1，因此不想关注任何特定内容的 token 会将权重倾倒在第一个 token（或前几个）上。这窃取了本应给予真实内容的容量。

差分注意力通过计算**两个**注意力图并相减来解决这个问题：

```
A1 = softmax(Q1 K1^T / √d)
A2 = softmax(Q2 K2^T / √d)
DiffAttn = (A1 - λ · A2) V
```

其中 `λ` 是可学习的标量（通常 0.5-0.8）。A1 捕获真实内容权重；A2 捕获汇聚。相减消除汇聚，将权重重新分配给相关 token。

报告结果（微软 2024）：困惑度降低 5-10%，在相同训练长度下有效上下文延长 1.5-2 倍，针头检索（needle-in-haystack）更精准。

### 变体对比

| 变体 | 计算 | KV 缓存 | 质量 vs 完整注意力 | 生产使用 |
|------|------|---------|-------------------|---------|
| 完整注意力 | O(N²) | 每层 O(N) | 基准 | 每个模型的默认层 |
| SWA（窗口 1024） | O(N·W) | 每层 O(W) | 困惑度 -0.1，配合全局层效果好 | Gemma 2/3、Phi-3-Long |
| 局部 + 步进稀疏 | O(N·√N) | 混合 | 类似 SWA | OpenAI 稀疏 Transformer、Longformer |
| BigBird（局部 + 全局 + 随机） | O(N) 近似 | 混合 | 在 2× 上下文下匹配完整注意力 | 早期长上下文 BERT |
| 原生稀疏（DeepSeek-V3.2） | O(N · 激活比例) | O(N) | 困惑度差距在 0.05 以内 | DeepSeek-V3.2，2025 |
| 差分 | O(2·N²) | O(2N) | 困惑度 -5% 到 -10% | DIFF Transformer，2026 年初期模型 |

## 动手实现

见 `code/main.py`。我们实现一个因果掩码比较器，在玩具序列上并排展示完整、SWA、局部+步进和差分注意力。

### 步骤一：完整因果掩码（基准）

```python
def causal_mask(n):
    return [[0.0 if j <= i else float("-inf") for j in range(n)] for i in range(n)]
```

第 07 课的基准。下三角；对角线上方零权重。

### 步骤二：滑动窗口因果掩码

```python
def swa_mask(n, window):
    M = [[float("-inf")] * n for _ in range(n)]
    for i in range(n):
        lo = max(0, i - window + 1)
        for j in range(lo, i + 1):
            M[i][j] = 0.0
    return M
```

一个参数——`window`。当 `window >= n` 时，恢复完整因果注意力。当 `window = 1` 时，每个 token 只关注自身。

### 步骤三：局部 + 步进稀疏掩码

```python
def strided_mask(n, window, stride):
    M = [[float("-inf")] * n for _ in range(n)]
    for i in range(n):
        lo = max(0, i - window + 1)
        for j in range(lo, i + 1):
            M[i][j] = 0.0
        for j in range(0, i + 1, stride):
            M[i][j] = 0.0
    return M
```

密集局部窗口加上每隔 `stride` 个 token 的历史记录。感受野随附加层以对数步骤增长。

### 步骤四：差分注意力

```python
def diff_attention(Q1, K1, Q2, K2, V, lam):
    A1 = softmax_causal(Q1 @ K1.T / sqrt_d)
    A2 = softmax_causal(Q2 @ K2.T / sqrt_d)
    return (A1 - lam * A2) @ V
```

两次注意力传播，用可学习的混合系数相减。代码中我们比较单一 vs 差分注意力的汇聚热力图，观察汇聚的消失。

### 步骤五：KV 缓存大小

打印 `N = 131072` 时每个变体每层的缓存大小。SWA 和稀疏变体下降 10-100 倍。差分加倍。有意识地支付内存代价。

## 生产使用

2026 年的生产模式：

```python
from transformers import AutoModelForCausalLM
# Gemma 3 以 5:1 混合 SWA（window=1024）和全局层。
model = AutoModelForCausalLM.from_pretrained("google/gemma-3-27b-it")
# print(model.config.sliding_window, model.config.layer_types)
```

PyTorch 2.5+ 中的 FlexAttention 接受掩码函数：

```python
from torch.nn.attention.flex_attention import flex_attention, create_block_mask

def swa_pattern(b, h, q_idx, kv_idx):
    return (q_idx - kv_idx < 1024) & (q_idx >= kv_idx)

mask = create_block_mask(swa_pattern, B=batch, H=heads, Q_LEN=n, KV_LEN=n)
out = flex_attention(q, k, v, block_mask=mask)
```

这会编译成自定义 Triton 核。对于常见模式，速度与 FlashAttention-3 相差在 10% 以内，掩码函数是 Python 可调用对象。

**何时选择各变体：**

- **纯完整注意力** — 上下文约 16K 以内的所有层，或检索质量至关重要时。
- **SWA + 全局混合** — 长上下文（>32K），训练和推理受内存限制。2026 年 32K 以上的默认方案。
- **稀疏块注意力** — 自定义核，自定义模式。保留给专门工作负载（检索、音频）。
- **差分注意力** — 任何注意力汇聚污染有害的工作负载（长上下文 RAG、针头检索）。

## 上手实践

见 `outputs/skill-attention-variant-picker.md`。该技能根据目标上下文长度、检索需求和训练/推理算力概况，为新模型选择注意力拓扑。

## 练习

1. **简单。** 运行 `code/main.py`。验证 `window=4` 的 SWA 在每行中将最后 4 个 token 以外的所有内容归零。验证 `window=n` 逐位复现完整因果注意力。
2. **中等。** 在第 07 课的综合项目中实现 `window=1024` 的因果 SWA。在 tinyshakespeare 上训练 1000 步。与完整注意力相比，验证损失回退了多少？峰值内存下降了多少？
3. **困难。** 在综合项目模型中实现 Gemma-3 风格的 5:1 层混合（5 个 SWA，1 个全局）。在相同参数下与纯 SWA 和纯全局基准比较损失、内存和生成质量。
4. **困难。** 使用每头可学习的 `λ` 实现差分注意力。在合成检索任务（一根针，2000 个干扰项）上训练。测量与相同参数的单注意力基准相比的检索准确率。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| 滑动窗口注意力（SWA） | "局部注意力" | 每个查询关注其最后 `W` 个 token；KV 缓存缩小至 `O(W)`。 |
| 有效感受野（Effective receptive field） | "模型能看多远" | 在窗口为 `W` 的 `L` 层 SWA 栈中，最多 `L × W` 个 token。 |
| Longformer / BigBird | "局部 + 全局 + 随机" | 带有少数始终关注全局 token 的稀疏模式；早期长上下文方法。 |
| 原生稀疏注意力（Native Sparse Attention） | "DeepSeek 的核技巧" | 学习块级稀疏性；在核级别跳过零块，同时保持质量。 |
| 差分注意力（Differential attention） | "两个图，一个相减" | DIFF Transformer：从第一个注意力图中减去可学习 `λ` 倍的第二个注意力图，以消除注意力汇聚。 |
| 注意力汇聚（Attention sink） | "权重渗漏到 token 0" | Softmax 归一化迫使行总和为 1；无信息查询将权重倾倒在位置 0 上。 |
| FlexAttention | "掩码即 Python" | PyTorch 2.5+ API，将任意掩码函数编译成 FlashAttention 形状的核。 |
| 层类型混合（Layer type mix） | "5:1 SWA 与全局" | 在栈中交错稀疏和完整注意力层，以较低内存保持质量。 |

## 延伸阅读

- [Beltagy, Peters, Cohan (2020). Longformer: The Long-Document Transformer](https://arxiv.org/abs/2004.05150) — 滑动窗口 + 全局 token 的经典论文
- [Zaheer et al. (2020). Big Bird: Transformers for Longer Sequences](https://arxiv.org/abs/2007.14062) — 局部 + 全局 + 随机
- [Child et al. (2019). Generating Long Sequences with Sparse Transformers](https://arxiv.org/abs/1904.10509) — OpenAI 的局部 + 步进模式
- [Gemma Team (2024). Gemma 2: Improving Open Language Models at a Practical Size](https://arxiv.org/abs/2408.00118) — 1:1 SWA:全局混合
- [Gemma Team (2025). Gemma 3 technical report](https://arxiv.org/abs/2503.19786) — 5:1 混合，window=1024，当前教科书默认
- [Ye et al. (2024). Differential Transformer](https://arxiv.org/abs/2410.05258) — DIFF Transformer 论文
- [Yuan et al. (2025). Native Sparse Attention](https://arxiv.org/abs/2502.11089) — DeepSeek-V3.2 的可学习稀疏注意力
- [PyTorch — FlexAttention blog and docs](https://pytorch.org/blog/flexattention/) — 生产使用中掩码即可调用模式的 API 参考
