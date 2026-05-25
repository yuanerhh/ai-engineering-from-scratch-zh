# 专家混合（MoE）

> 一个密集的 700 亿 Transformer 对每个 token 激活所有参数。一个 6710 亿的 MoE 每个 token 只激活 370 亿，并在每个基准上击败它。稀疏性是十年来最重要的扩展理念。

**类型：** 构建
**语言：** Python
**前置条件：** Phase 7 · 05（完整 Transformer）、Phase 7 · 07（GPT）
**时长：** 约 45 分钟

## 问题背景

密集 Transformer 在推理时的 FLOP 等于其参数量（前向传播乘以 2）。扩大密集模型，每个 token 都付全部代价。到 2024 年，前沿已经碰到了算力瓶颈：要明显更聪明，每个 token 需要指数级更多的 FLOP。

专家混合（Mixture of Experts，MoE）打破了这一联系。将每个 FFN 替换为 `E` 个独立专家 + 一个路由器，路由器为每个 token 选择 `k` 个专家。总参数量 = `E × FFN_size`。每个 token 的激活参数量 = `k × FFN_size`。2026 年典型配置：`E=256`，`k=8`。存储随 `E` 扩展，计算随 `k` 扩展。

2026 年的前沿几乎完全是 MoE：DeepSeek-V3（6710 亿总参数 / 370 亿激活），Mixtral 8×22B，Qwen2.5-MoE，Llama 4，Kimi K2，gpt-oss。在 Artificial Analysis 的独立排行榜上，前 10 名开源模型全部是 MoE。

## 核心概念

![MoE 层：路由器为每个 token 从 E 个专家中选 k 个](../assets/moe.svg)

### FFN 替换

密集 Transformer 块：

```
h = x + attn(norm(x))
h = h + FFN(norm(h))
```

MoE 块：

```
h = x + attn(norm(x))
scores = router(norm(h))              # (N_tokens, E)
top_k = argmax_k(scores)              # 每个 token 从 E 个中选 k 个
h = h + sum_{e in top_k}(
        gate(scores[e]) * Expert_e(norm(h))
    )
```

每个专家都是独立的 FFN（通常是 SwiGLU）。路由器是单个线性层。每个 token 选择自己的 `k` 个专家，获得它们输出的门控混合。

### 负载均衡问题

如果路由器将 90% 的 token 发给专家 3，其他专家就会饥饿。已尝试过三种修复方案：

1. **辅助负载均衡损失**（Switch Transformer、Mixtral）。添加与专家使用方差成比例的惩罚。有效，但增加了超参数和第二个梯度信号。
2. **专家容量 + token 丢弃**（早期 Switch）。每个专家最多处理 `C × N/E` 个 token；溢出 token 跳过该层。损害质量。
3. **无辅助损失均衡**（DeepSeek-V3）。添加一个可学习的每专家偏置，移动路由器的 top-k 选择。偏置在训练损失之外更新，不惩罚主目标。2024 年的重大突破。

DeepSeek-V3 的方法：每次训练步骤后，对每个专家，检查其使用量是否高于或低于目标。以 `±γ` 微调偏置。选择使用 `scores + bias`。用于门控的专家概率使用原始的 `scores` 不变。将路由与表达解耦。

### 共享专家

DeepSeek-V2/V3 还将专家分为*共享*和*路由*两类。每个 token 经过所有共享专家。路由专家通过 top-k 选择。共享专家捕获常识；路由专家专门化。V3 运行 1 个共享专家加 256 个路由专家中的 top-8。

### 细粒度专家

经典 MoE（GShard、Switch）：每个专家与完整 FFN 一样宽。`E` 较小（8-64），`k` 较小（1-2）。

现代细粒度 MoE（DeepSeek-V3、Qwen-MoE）：每个专家更窄（1/8 FFN 大小）。`E` 很大（256+），`k` 更大（8+）。总参数量相同，但组合数量扩展得更快。`C(256, 8) = 400 万亿` 种每个 token 可能的"专家"组合。质量提升，延迟持平。

### 成本概况

每个 token，每层：

| 配置 | 每 token 激活参数量 | 总参数量 |
|------|--------------------|---------|
| Mixtral 8×22B | 约 390 亿 | 1410 亿 |
| Llama 3 70B（密集） | 700 亿 | 700 亿 |
| DeepSeek-V3 | 370 亿 | 6710 亿 |
| Kimi K2（MoE） | 约 320 亿 | 1T |

DeepSeek-V3 在几乎每个基准上都击败 Llama 3 70B（密集），同时每个 token 使用**更少的激活 FLOP**。更多参数 = 更多知识。更多激活 FLOP = 每个 token 更多计算。MoE 将两者解耦。

### 代价：内存

无论哪些专家被激活，所有专家都驻留在 GPU 上。一个 6710 亿模型的 fp16 权重需要约 1.3 TB VRAM。前沿 MoE 部署需要专家并行性——跨 GPU 分片专家，跨网络路由 token。延迟由全对全通信主导，而非矩阵乘法。

## 动手实现

见 `code/main.py`。纯标准库的紧凑 MoE 层，包含：

- `n_experts=8` 个类 SwiGLU 专家（每个一个线性层，用于说明）
- top-k=2 路由
- softmax 归一化的门控权重
- 通过每专家偏置的无辅助损失均衡

### 步骤一：路由器

```python
def route(hidden, W_router, top_k, bias):
    scores = [sum(h * w for h, w in zip(hidden, W_router[e])) for e in range(len(W_router))]
    biased = [s + b for s, b in zip(scores, bias)]
    top_idx = sorted(range(len(biased)), key=lambda i: -biased[i])[:top_k]
    # 对所选专家的原始分数做 softmax
    chosen = [scores[i] for i in top_idx]
    m = max(chosen)
    exps = [math.exp(c - m) for c in chosen]
    s = sum(exps)
    gates = [e / s for e in exps]
    return top_idx, gates
```

偏置影响选择，不影响门控权重。这就是 DeepSeek-V3 的技巧——偏置纠正负载不均衡，而不引导模型的预测。

### 步骤二：让 100 个 token 经过路由器

追踪哪些专家被激活了多少次。没有偏置时，使用量是倾斜的。有偏置更新循环（对过度使用的专家 `-γ`，对使用不足的专家 `+γ`），使用量在几次迭代后收敛到均匀分布。

### 步骤三：参数量对比

打印 MoE 配置的"密集等效量"。DeepSeek-V3 形状：256 个路由 + 1 个共享，8 个激活，d_model=7168。总参数量令人瞠目。激活参数量是密集 Llama 3 70B 的七分之一。

## 生产使用

HuggingFace 加载：

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained("mistralai/Mixtral-8x22B-v0.1")
```

2026 年生产推理：vLLM 原生支持 MoE 路由。SGLang 拥有最快的专家并行路径。两者都自动处理 top-k 选择和专家并行。

**何时选择 MoE：**
- 你想要以较低的每 token 推理成本获得前沿质量。
- 你拥有 VRAM / 专家并行基础设施。
- 你的工作负载是 token 密集型（对话、代码），而非上下文密集型（长文档）。

**何时不选择 MoE：**
- 边缘部署——任何激活的 FLOP 都要付出完整存储代价。
- 延迟关键的单用户服务——专家路由增加开销。
- 小模型（<70 亿参数）——MoE 的质量优势只在超过某个算力阈值（约 60 亿激活参数）时才出现。

## 上手实践

见 `outputs/skill-moe-configurator.md`。该技能根据参数预算、训练 token 数和部署目标，为新 MoE 选择 E、k 和共享专家布局。

## 练习

1. **简单。** 运行 `code/main.py`。观察无辅助损失的偏置更新如何在 50 次迭代中均衡专家使用量。
2. **中等。** 将可学习路由器替换为基于哈希的路由器（确定性，无学习）。比较质量和均衡性。为什么可学习路由器更好？
3. **困难。** 实现 GRPO 风格的"rollout 匹配路由"（DeepSeek-V3.2 技巧）：记录推理期间哪些专家被激活，在梯度计算期间强制相同的路由。在玩具策略梯度设置上测量效果。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| 专家（Expert） | "众多 FFN 之一" | 独立的前馈网络；参数专用于 FFN 计算的稀疏切片。 |
| 路由器（Router） | "门控" | 对每个 token 相对于每个专家打分的微型线性层；top-k 选择。 |
| Top-k 路由（Top-k routing） | "每个 token k 个激活专家" | 每个 token 的 FFN 计算经过恰好 k 个专家，由门控加权。 |
| 辅助损失（Auxiliary loss） | "负载均衡惩罚" | 惩罚专家使用不均匀的额外损失项。 |
| 无辅助损失（Auxiliary-loss-free） | "DeepSeek-V3 的技巧" | 仅通过路由器选择上的每专家偏置进行均衡；无额外梯度。 |
| 共享专家（Shared expert） | "始终激活" | 每个 token 都经过的额外专家；捕获常识。 |
| 专家并行（Expert parallelism） | "按专家分片" | 将不同专家分布到不同 GPU；跨网络路由 token。 |
| 稀疏性（Sparsity） | "激活参数 < 总参数" | 比率 `k × expert_size / (E × expert_size)`；DeepSeek-V3 约 37/671 ≈ 5.5%。 |

## 延伸阅读

- [Shazeer et al. (2017). Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer](https://arxiv.org/abs/1701.06538) — 这个想法的起源
- [Fedus, Zoph, Shazeer (2022). Switch Transformer: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity](https://arxiv.org/abs/2101.03961) — Switch，经典 MoE
- [Jiang et al. (2024). Mixtral of Experts](https://arxiv.org/abs/2401.04088) — Mixtral 8×7B
- [DeepSeek-AI (2024). DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437) — MLA + 无辅助损失 MoE + MTP
- [Wang et al. (2024). Auxiliary-Loss-Free Load Balancing Strategy for Mixture-of-Experts](https://arxiv.org/abs/2408.15664) — 基于偏置的均衡论文
- [Dai et al. (2024). DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models](https://arxiv.org/abs/2401.06066) — 本课路由器使用的细粒度 + 共享专家分拆
- [Kim et al. (2022). DeepSpeed-MoE: Advancing Mixture-of-Experts Inference and Training](https://arxiv.org/abs/2201.05596) — 原始共享专家论文
