# KV 缓存、Flash Attention 与推理优化

> 训练是并行的、受 FLOP 限制的。推理是串行的、受内存限制的。瓶颈不同，技巧不同。

**类型：** 构建
**语言：** Python
**前置条件：** Phase 7 · 02（自注意力）、Phase 7 · 05（完整 Transformer）、Phase 7 · 07（GPT）
**时长：** 约 75 分钟

## 问题背景

朴素的自回归解码器生成 `N` 个 token 需要 `O(N²)` 的工作量：每步都要对完整前缀重新计算注意力。对于 4K token 的响应，这是 1600 万次注意力操作，其中大多数是冗余的。前缀 token 的每个隐藏状态一旦计算就是确定性的——你只需要用新 token 的查询对之前所有内容的缓存键和值运行一次。

除此之外，注意力本身会移动大量数据。标准注意力需要具象化一个 N×N 的分数矩阵、N×d 的 softmax 输出、N×d 的最终输出——对 HBM 的读写过于频繁。对于 N≥2K，注意力在受 FLOP 限制之前就已受内存限制。经典注意力核对现代 GPU 的利用率不到 10-25%。

Dao et al. 的两项优化将前沿推理从"慢"推向"快"：

1. **KV 缓存。** 存储每个前缀 token 的 K 和 V 向量。每个新 token 的注意力只需一个查询对缓存键。推理从每步 `O(N²)` 降为 `O(N)`。
2. **Flash Attention。** 将注意力计算分块，使完整的 N×N 矩阵永远不会接触 HBM。所有 softmax + 矩阵乘法在 SRAM 中完成。A100 上实际加速 2-4×；H100 上 FP8 加速 5-10×。

2026 年两者都已普及。每个生产推理栈（vLLM、TensorRT-LLM、SGLang、llama.cpp）都以它们为前提。每个前沿模型都启用了 Flash Attention。

## 核心概念

![KV 缓存增长和 Flash Attention 分块](../assets/kv-cache-flash-attn.svg)

### KV 缓存数学

每个解码器层，每个 token，每个头：

```
每 token 每层字节数 = 2 * d_head * dtype_size
                     ^
                     K 和 V
```

对于 32 层、32 头、d_head=128、fp16 的 7B 模型：

```
每 token 每层 = 2 * 128 * 2 = 512 字节
每 token（32 层）= 16 KB
32K 上下文 = 512 MB
```

对于 Llama 3 70B（80 层，d_head=128，GQA 有 8 个 KV 头）：

```
每 token 每层 = 2 * 8 * 128 * 2 = 4096 字节（4 KB）
32K 上下文 = 10.4 GB
```

这 10 GB 就是为什么 Llama 3 70B 在 128K 上下文下、批大小为 1 时，光 KV 缓存就需要占用 40 GB A100 的大部分。

**GQA 是 KV 缓存的胜利。** MHA 有 64 个头需要 32 GB。MLA 压缩得更多。

### Flash Attention——分块技巧

标准注意力：

```
S = Q @ K^T          （HBM 读，N×N，HBM 写）
P = softmax(S)       （HBM 读，HBM 写）
O = P @ V            （HBM 读，HBM 写）
```

三次 HBM 往返。在 H100 上，HBM 带宽为 3 TB/s；SRAM 为 30 TB/s。每次 HBM 往返比全片上处理慢 10 倍。

Flash Attention：

```
对于 Q 的每个块（块大小约 128 × 128）：
    将 Q_tile 加载到 SRAM
    对于 K、V 的每个块：
        将 K_tile、V_tile 加载到 SRAM
        计算 S_tile = Q_tile @ K_tile^T     （SRAM）
        运行 softmax 聚合                    （SRAM）
        累加到 O_tile                        （SRAM）
    将 O_tile 写入 HBM
```

每个块只有一次 HBM 往返。总内存占用从 `O(N²)` 降至 `O(N)`。反向传播从前向传播重新计算某些值，而非存储它们——另一个内存优势。

**数值技巧。** 运行 softmax 跨块维护 `(最大值, 求和)` 以使最终归一化精确。不是近似——Flash Attention 计算出与标准注意力逐位相同的输出（fp16 非结合性误差除外）。

**版本演进：**

| 版本 | 年份 | 关键变化 | 参考硬件加速比 |
|------|------|---------|--------------|
| Flash 1 | 2022 | 分块 SRAM 核 | A100 上 2× |
| Flash 2 | 2023 | 更好的并行性，因果优先排序 | A100 上 3× |
| Flash 3 | 2024 | Hopper 异步，FP8 | H100 上 1.5-2×（约 740 TFLOPs FP16） |
| Flash 4 | 2026 | Blackwell 5 级流水线，软件 exp2 | 推理优先（最初仅前向传播） |

Flash 4 发布时仅支持前向传播。训练仍使用 Flash 3。GQA 和 varlen 对 Flash 4 的支持仍在推进中（2026 年中期）。

### 投机解码——另一个延迟优势

廉价模型提出 N 个 token。大模型并行验证所有 N 个。如果验证接受了 k 个 token，你用 1 次大模型前向传播换来了 k 次生成。代码和散文上典型 k=3-5。

2026 年默认方案：
- **EAGLE 2 / Medusa。** 共享验证器隐藏状态的集成草稿头。2-3× 加速，无质量损失。
- **用草稿模型的投机解码。** 消费者硬件上 2-4× 加速。
- **前瞻解码（Lookahead decoding）。** Jacobi 迭代；无需草稿模型。小众但免费。

### 连续批处理

经典批量推理：等待最慢的序列完成，然后开始新批次。短响应提前完成时浪费 GPU。

连续批处理（最初由 Orca 发布，现在在 vLLM、TensorRT-LLM、SGLang 中）：旧请求完成后立即将新请求换入批次。典型对话工作负载吞吐量提升 5-10×。

### PagedAttention——KV 缓存作为虚拟内存

vLLM 的标志性功能。KV 缓存以 16 个 token 的块分配；页表将逻辑位置映射到物理块。支持在并行采样（束搜索、并行采样）中共享 KV、用于提示缓存的前缀热换，以及内存碎片整理。与朴素连续分配相比，吞吐量提升 4×。

## 动手实现

见 `code/main.py`。我们实现：

1. 朴素的 `O(N²)` 增量解码器。
2. `O(N)` KV 缓存解码器。
3. 模拟 Flash Attention 运行最大值算法的分块 softmax。

### 步骤一：KV 缓存

```python
class KVCache:
    def __init__(self, n_layers, n_heads, d_head):
        self.K = [[[] for _ in range(n_heads)] for _ in range(n_layers)]
        self.V = [[[] for _ in range(n_heads)] for _ in range(n_layers)]

    def append(self, layer, head, k, v):
        self.K[layer][head].append(k)
        self.V[layer][head].append(v)

    def read(self, layer, head):
        return self.K[layer][head], self.V[layer][head]
```

简单：在每层、每头的列表中保持增长的每 token K、V 向量。

### 步骤二：分块 softmax

```python
def tiled_softmax_dot(q, K, V, tile=4):
    """Flash Attention 风格的 softmax(qK^T)V，使用运行最大值/求和。"""
    m = float("-inf")
    s = 0.0
    out = [0.0] * len(V[0])
    for start in range(0, len(K), tile):
        k_block = K[start:start + tile]
        v_block = V[start:start + tile]
        scores = [sum(qi * ki for qi, ki in zip(q, k)) for k in k_block]
        new_m = max(m, *scores)
        exp_old = math.exp(m - new_m) if m != float("-inf") else 0.0
        exp_new = [math.exp(sc - new_m) for sc in scores]
        s = s * exp_old + sum(exp_new)
        for j in range(len(out)):
            out[j] = out[j] * exp_old + sum(e * v[j] for e, v in zip(exp_new, v_block))
        m = new_m
    return [o / s for o in out]
```

与一次性 `softmax(qK) V` 的输出逐位相同，但任何时候的工作集都是一个 `tile × d_head` 块，而非完整的 `N × d_head`。

### 步骤三：比较朴素解码与缓存解码，生成 100 个 token

统计注意力操作次数。朴素：`O(N²)` = 5050。缓存：`O(N)` = 100。代码打印两者。

## 生产使用

```python
# HuggingFace transformers 在仅解码器的 generate() 上自动启用 KV 缓存。
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-3B",
    attn_implementation="flash_attention_2",  # Hopper 上使用 FA3
    torch_dtype="bfloat16",
)
# generate() 自动使用 KV 缓存
```

vLLM 生产部署：

```bash
pip install vllm
vllm serve meta-llama/Llama-3.1-70B-Instruct \
    --tensor-parallel-size 4 \
    --max-model-len 32768 \
    --enable-prefix-caching \
    --kv-cache-dtype fp8
```

跨请求的前缀缓存是 2026 年的重大优势——相同的系统提示、少样本示例或长上下文文档可跨调用复用 KV。对于具有重复工具提示的智能体工作负载，前缀缓存通常带来 5× 的吞吐量提升。

## 上手实践

见 `outputs/skill-inference-optimizer.md`。该技能为新推理部署选择注意力实现、KV 缓存策略、量化和投机解码。

## 练习

1. **简单。** 运行 `code/main.py`。确认朴素解码器和缓存解码器产生相同输出；注意操作次数的差异。
2. **中等。** 实现前缀缓存：给定一个提示 P 和多个补全，对 P 运行一次前向传播以填充 KV 缓存，然后按每个补全分支。测量与为每个补全重新编码 P 相比的加速比。
3. **困难。** 实现玩具 PagedAttention：KV 缓存以固定 16 个 token 的块存储，带有空闲列表。序列完成后，将其块归还给池。模拟 1000 次不同长度的对话补全。比较与连续分配相比的内存碎片。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| KV 缓存（KV cache） | "使解码快速的技巧" | 存储每个前缀 token 的 K 和 V；新查询对其进行注意力，而非重新计算。 |
| HBM | "GPU 主内存" | 高带宽内存；H100 上 80 GB，B200 上 192 GB。约 3 TB/s 带宽。 |
| SRAM | "片上内存" | 每 SM 的快速内存，H100 上每 SM 约 256 KB。约 30 TB/s 带宽。 |
| Flash Attention | "分块注意力核" | 不在 HBM 中具象化 N×N 的情况下计算注意力。 |
| 连续批处理（Continuous batching） | "无等待批处理" | 将完成的序列换出，将新序列换入，无需排空批次。 |
| PagedAttention | "vLLM 的标志" | KV 缓存以固定块分配，带页表；消除碎片。 |
| 前缀缓存（Prefix caching） | "复用长提示" | 跨请求缓存共享前缀的 KV；智能体大幅削减成本。 |
| 投机解码（Speculative decoding） | "草稿 + 验证" | 廉价草稿模型提出 token；大模型一次验证 k 个。 |

## 延伸阅读

- [Dao et al. (2022). FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135) — Flash 1
- [Dao (2023). FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning](https://arxiv.org/abs/2307.08691) — Flash 2
- [Shah et al. (2024). FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision](https://arxiv.org/abs/2407.08608) — Flash 3
- [FlashAttention-4 release notes (Dao-AILab, 2026)](https://github.com/Dao-AILab/flash-attention) — Blackwell 5 级流水线和软件 exp2 技巧
- [Kwon et al. (2023). Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180) — vLLM 论文
- [Leviathan et al. (2023). Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192) — 投机解码
- [Li et al. (2024). EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty](https://arxiv.org/abs/2401.15077) — EAGLE-1/2 集成草稿方法论文
- [Cai et al. (2024). Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads](https://arxiv.org/abs/2401.10774) — Medusa 方法
- [vLLM docs — PagedAttention](https://docs.vllm.ai/en/latest/design/kernel/paged_attention.html) — 16 token 块和页表设计的权威深度剖析
