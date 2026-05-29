# vLLM 服务内部原理：PagedAttention、连续批处理、分块预填充

> vLLM 在 2026 年的主导地位源于三个相互叠加的默认特性，而非单一技巧。PagedAttention 始终开启。连续批处理在解码迭代之间将新请求注入活跃批次。分块预填充将长提示词切片，使解码 token 永不饥饿。三者全开，单台 H100 SXM5 上的 Llama 3.3 70B FP8 在 128 并发下可达 2200-2400 token/秒——比 vLLM 自身的默认设置高约 25%，比朴素 PyTorch 循环高 3-4 倍。本课以你能绘图讲解的深度阅读调度器和注意力内核，并在 `code/main.py` 中以 vLLM 的方式调度预填充和解码，实现一个玩具连续批处理器。

**类型：** 学习
**编程语言：** Python（标准库，玩具连续批处理调度器）
**前置知识：** Phase 17 · 01（模型服务）、Phase 11（LLM 工程）
**预计时间：** 约 75 分钟

## 学习目标

- 将 PagedAttention 解释为 KV 缓存分配器：块、块表，以及为什么在生产负载下碎片化率低于 4%。
- 在迭代级别绘制连续批处理图：完成的序列如何离开批次，新序列如何无需排空即可加入。
- 用一句话描述分块预填充，并指出它保护的是哪个延迟指标（提示：是首 token 延迟的尾部，而非平均吞吐量）。
- 命名 2026 年 vLLM v0.18.0 中同时启用所有优化时会遇到的"坑"。

## 问题背景

朴素的 PyTorch 服务循环每次处理一个请求：分词、预填充、解码直到 EOS，返回。一个用户时没问题。一百个用户时，就是一队耐心等待的人。显而易见的修复——静态批处理——将每个请求填充到窗口中最长提示词的长度，将每次解码填充到最长预期输出的长度，并让整个批次等待最慢的序列。你为从未使用的填充付费，快速请求等待慢速请求。

vLLM 同时解决三个问题。PagedAttention 阻止 KV 缓存碎片化像经典连续分配那样吃掉 60-80% 的 GPU 内存。连续批处理让请求在每次解码迭代之间加入和离开批次，使批次始终充满真实工作。分块预填充将 32k token 的提示词分成约 512 token 的切片，与解码交错，使长提示词不会冻结 GPU 上所有其他序列的每个解码 token。

2026 年的生产默认值是三者全开。你需要理解每个的作用，因为失败模式全在调度器上，不在模型上。

## 核心概念

### PagedAttention 作为虚拟内存系统

KV 缓存每个序列占用 `层数 × 2 × 头数 × 头维度 × 序列长度 × 每元素字节数`。对于 8192 token 的 Llama 3.3 70B（BF16），约为每序列 1.25 GB。如果你为每个请求预留 8192 个槽但平均请求只用 1500 个 token，你浪费了约 82% 的预留 HBM。经典批处理就是这种浪费。

PagedAttention 借用操作系统虚拟内存的思想。KV 缓存不是每序列连续的，而是以固定大小的块（默认 16 个 token）分配。每个序列有一个块表，将其逻辑 token 位置映射到物理块 ID。当序列增长超出已分配的块时，再添加一个块。当序列完成时，其块返回池中。

碎片化从 60-80%（经典）降至 4% 以下（PagedAttention）。你不需要用标志启用 PagedAttention——它是 vLLM 唯一内置的分配器。调节旋钮是 `--gpu-memory-utilization`（默认 0.9），告诉 vLLM 在加载权重和激活后为 KV 块保留多少 HBM。

### 迭代级连续批处理

旧的"动态批处理"等待一个窗口（比如 10ms）填满批次，然后运行预填充 + 解码 + 解码 + 解码，直到每个序列完成。快速序列早早离开，在 GPU 完成慢速序列时处于空闲。

连续批处理在每个解码步骤之间运行。将运行中的序列集合称为 `RUNNING` 列表。每次迭代：

1. `RUNNING` 中刚到达 EOS 或 max_tokens 的序列被移除。
2. 调度器查看等待队列。如果有空闲 KV 块，就接纳新序列（预填充或恢复）。
3. 对 `RUNNING` 中现有的所有内容执行前向传递，每个序列发出一个新 token。

批次大小永远不会填充到固定数量。处于不同输出位置的序列共享一次融合前向传递。在 2026 年的 vLLM 中这被称为 `V1 调度器`。关键不变量：调度器每次解码迭代运行一次，而非每个请求运行一次。

### 分块预填充保护首 token 延迟的尾部

预填充是计算密集型的。在一台 H100 上，Llama 3.3 70B 的 32k token 提示词需要约 800ms 的纯预填充时间。预填充运行时，批次中所有其他序列的解码 token 都在等待。在服务循环中，一个长提示词的首 token 延迟（TTFT）变成了数十个其他用户的令牌间延迟（ITL）抖动。

分块预填充将预填充分成固定大小的块（默认 512 token），将每个块作为一个单元调度。在块之间，调度器可以将解码序列推进一个 token。你用轻微的绝对预填充延迟增加（每块几毫秒）换取了低得多的解码时间抖动。在已发布的基准测试中，混合负载下 P99 ITL 从约 50ms 降至约 15ms。

### 三个默认值相互配合

所有三个特性互相假设对方存在。PagedAttention 给调度器一个细粒度的 KV 资源来权衡。连续批处理需要那个细粒度资源，这样接纳新序列不会强制全局重新排列。分块预填充是调度器在同一个 `RUNNING` 列表上做的决策——它是又一个调度器策略，而非独立系统。

你不需要知道每个标志。你需要知道调度器优化什么：在 KV 块预算约束下的 goodput，受分块预填充切片约束。

### 2026 年 v0.18.0 的坑

在 vLLM v0.18.0 中，你不能将 `--enable-chunked-prefill` 与草稿模型投机解码（`--speculative-model`）结合使用。有文档记录的例外是 V1 调度器中的 N-gram GPU 投机解码。不读发布说明就打开所有标志的团队会在启动时遇到运行时错误，而非软性回归。如果你认为分块预填充的增益值得投机解码，那么 2026 年正确答案通常是 EAGLE-3（不带分块预填充），而非无法编译的草稿模型加分块预填充。

### 需要记住的数字

- Llama 3.3 70B FP8、H100 SXM5、128 并发、三者全开：2200-2400 token/秒。
- 同款模型，默认 vLLM（无分块预填充）：约 1800 token/秒。
- 同款模型，朴素 PyTorch 前向循环：约 600 token/秒。
- 生产负载下 PagedAttention 的 KV 碎片化浪费：< 4%。
- 混合负载下 P99 ITL：带分块预填充约 15ms，不带约 50ms。

### 调度器示意

```
while True:
    finished = [s for s in RUNNING if s.is_done()]
    for s in finished: release_blocks(s); RUNNING.remove(s)

    while WAITING and have_free_blocks_for(WAITING[0]):
        s = WAITING.pop(0)
        allocate_initial_blocks(s)
        RUNNING.append(s)

    # 在一个批次中调度预填充块 + 解码
    batch = []
    for s in RUNNING:
        if s.in_prefill:
            batch.append(next_prefill_chunk(s))   # 例如 512 个 token
        else:
            batch.append(decode_one_token(s))     # 1 个 token

    run_forward(batch)                            # 一次融合 GPU 调用
```

`code/main.py` 正是这个循环，用标准库 Python 实现，带假 token 计数和假前向延迟。运行它可以看到分块预填充在长预填充期间如何保持解码序列活跃。

## 动手实践

`code/main.py` 模拟一个带可切换特性的 vLLM 风格调度器。运行它可以看到：

- `NAIVE` 模式：每次一个请求，无批处理。
- `STATIC` 模式：填充并等待，经典批处理。
- `CONTINUOUS` 模式：迭代级接纳和释放。
- `CONTINUOUS + CHUNKED` 模式：预填充切片与解码交错。

输出显示总吞吐量（虚拟秒内的 token 数）、平均首 token 延迟和 P99 ITL。在混合流量下，`CONTINUOUS + CHUNKED` 行应该占优。

## 产出技能

本课产出 `outputs/skill-vllm-scheduler-reader.md`。给定服务配置（批次大小、KV 内存利用率、分块预填充大小、投机配置），产出调度器诊断，指出三个默认值中哪个是瓶颈以及应调整什么。

## 练习

1. 运行 `code/main.py`。在包含混合长短请求的工作负载上比较 `STATIC` 和 `CONTINUOUS`。吞吐量差距来自哪里——预填充效率、解码效率还是尾部延迟？
2. 修改玩具调度器添加 `--max-num-batched-tokens`。对于在 H100 上运行 Llama 3.3 70B FP8 的正确值是多少？（提示：它是 KV 块大小和空闲块数量的函数，不是裸 HBM。）
3. 重新阅读 vLLM v0.18.0 发布说明。哪些标志组合是互斥的？列出它们。
4. 计算 1000 个请求的跟踪（平均 1500 输出 token，标准差 600 token）在以下情况下的 KV 缓存碎片化浪费：(a) 按请求连续分配，最大 8192；(b) 带 16 token 块的 PagedAttention。
5. 用一段话解释为什么分块预填充单独来看有助于 P99 ITL 但不改善吞吐量。吞吐量提升在实践中从哪里来？

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| PagedAttention | "KV 技巧" | KV 缓存的固定大小块分配器；碎片化 < 4% |
| 块表（Block table） | "页表" | 每序列从逻辑 token 位置到物理 KV 块的映射 |
| 连续批处理（Continuous batching） | "动态批处理，但做对了" | 每次解码迭代做接纳/释放决策 |
| 分块预填充（Chunked prefill） | "预填充分割" | 将长预填充分成 512 token 切片，与解码交错 |
| TTFT | "首 token 时间" | 预填充 + 队列 + 网络；长提示词时以预填充为主 |
| ITL | "令牌间延迟" | 连续解码 token 之间的时间；以批次大小为主 |
| Goodput | "满足 SLO 的吞吐量" | 每秒仍满足 TTFT 和 ITL 目标的请求中的 token 数 |
| V1 调度器（V1 scheduler） | "新调度器" | vLLM 2026 调度器；N-gram 投机解码是兼容分块预填充的路径 |
| `--gpu-memory-utilization` | "内存旋钮" | 权重和激活之后为 KV 块保留的 HBM 比例 |

## 延伸阅读

- [vLLM 文档——投机解码](https://docs.vllm.ai/en/latest/features/spec_decode/) — 分块预填充与投机解码兼容性的官方来源
- [vLLM 发布说明（NVIDIA）](https://docs.nvidia.com/deeplearning/frameworks/vllm-release-notes/index.html) — 2026 年发布节奏和版本特定行为
- [vLLM 博客——PagedAttention](https://blog.vllm.ai/2023/06/20/vllm.html) — 仍然是理解分配器思路的原始文章
- [PagedAttention 论文（arXiv:2309.06180）](https://arxiv.org/abs/2309.06180) — 碎片化分析和调度器设计
- [Aleksa Gordic——vLLM 内部](https://www.aleksagordic.com/blog/vllm) — 带火焰图的 V1 调度器详细介绍
