# Blackwell 上的 TensorRT-LLM：FP8 与 NVFP4

> TensorRT-LLM 仅支持 NVIDIA，但在 Blackwell 上称霸。在带 Dynamo 编排的 GB200 NVL72 上，SemiAnalysis InferenceX 于 2026 年第一二季度测量到 120B 模型的成本为每百万 token 0.012 美元，而 H100 + vLLM 为 0.09 美元/百万——7 倍经济差距。整个技术栈由三种浮点精度方案叠加而成：FP8 对 KV 缓存和注意力内核仍然关键，因为它们需要那个动态范围；NVFP4（4 位微缩放）处理权重和激活；多 token 预测（MTP）和分离式预填充/解码在此基础上再增加 2-3 倍。Day-0 模型支持直接加载 FP4 权重而无需训练后转换。2026 年工程团队需要注意的陷阱：TRT-LLM 是封闭的 NVIDIA 技术栈，采用它意味着用可移植性换取吞吐量。在承诺之前，先在你的模型和硬件混合上算好账。

**类型：** 学习
**编程语言：** Python（标准库，玩具 FP8/NVFP4 内存与成本计算器）
**前置知识：** Phase 17 · 04（vLLM 服务内部原理）、Phase 10 · 13（量化）
**预计时间：** 约 75 分钟

## 学习目标

- 解释为什么即使权重使用 NVFP4，FP8 对 KV 缓存和注意力仍然关键。
- 计算前沿模型在 BF16、FP8 和 NVFP4 下的 HBM 占用，并分析节省来自何处。
- 命名 TRT-LLM 利用的 Blackwell 专用特性（Day-0 FP4、MTP、分离式服务、全对全通信原语）。
- 判断何时值得为 7 倍成本差距接受 TRT-LLM 的 NVIDIA 锁定。

## 问题背景

2026 年推理经济学的前沿是"每美元多少 token"。答案取决于四个叠加的选择：硬件代（Hopper H100/H200 vs Blackwell B200/GB200）、精度（BF16 → FP8 → NVFP4）、服务引擎（vLLM vs SGLang vs TRT-LLM）和编排（普通 vs 分离式 vs Dynamo）。

在 Hopper + vLLM 上，120B MoE 每百万 token 运行成本约为 0.09 美元。在 Blackwell + TRT-LLM + Dynamo 上，同款模型约为 0.012 美元——便宜 7 倍。部分差距来自硬件（Blackwell 每 GPU 的 LLM 吞吐量比 Hopper 高 11-15 倍）。部分来自技术栈：FP4 权重、MTP 草稿、分离式预填充/解码和 NVLink 5 全对全通信用于 MoE 专家路由。

你无法在 NVIDIA 技术栈之外复制这一结果。这就是权衡——用可移植性换取经济性。理解哪个技术栈选择贡献了差距的哪部分，正是本课的重点。

## 核心概念

### 为什么 FP8 仍然是 KV 缓存的底线

2026 年一个常见错误：假设 NVFP4 适用于所有地方。它不适用。KV 缓存需要 FP8（8 位浮点），因为它存储的注意力键值跨越很大的动态范围。将 KV 量化到 FP4 会导致灾难性的精度损失——分布的尾部被截断，注意力分数崩溃。FP8 的指数位给 KV 缓存提供了它需要的范围。

NVFP4（2025-2026 年）应用于权重和激活。微缩放：每个权重块有自己的缩放因子，所以小块可以跨越不同的动态范围，而不会损失每张量缩放精度。对于激活，FP4 能保持是因为激活在层内是小范围的。

典型的 Blackwell 配置：

- 权重：NVFP4（4 位微缩放）。
- 激活：NVFP4。
- KV 缓存：FP8。
- 注意力累加器：FP32（softmax 稳定性）。

### TRT-LLM 使用的 Blackwell 专用原语

- **Day-0 FP4 权重**：模型提供商直接发布 FP4 权重；TRT-LLM 无需训练后转换即可加载。FP4 无需 AWQ/GPTQ 步骤。
- **多 token 预测（MTP）**：与 EAGLE 相同的思路（Phase 17 · 05），但集成在 TRT-LLM 构建中。
- **分离式服务**：预填充和解码在独立的 GPU 池上运行，KV 缓存通过 NVLink 或 InfiniBand 传输。与 Dynamo 相同的思路（Phase 17 · 20）。
- **全对全通信原语**：NVLink 5 将 MoE 专家通信延迟降低 3 倍（相比 Hopper）。TRT-LLM 的 MoE 内核针对此进行了调优。
- **NVFP4 + MXFP8 微缩放**：Blackwell Tensor Core 上硬件加速的缩放因子处理。

### 需要记住的数字

- HGX B200 通过 TRT-LLM 在 GPT-OSS-120B 上的每百万 token 成本：0.02 美元。
- GB200 NVL72 通过 Dynamo（编排 TRT-LLM）：0.012 美元/百万 token。
- H100 + vLLM 在可比工作负载上：约 0.09 美元/百万 token。
- TRT-LLM 三个月更新带来的吞吐量增益：2.8 倍（2026 年）。
- 每 GPU LLM 吞吐量，Blackwell vs Hopper：11-15 倍。
- MLPerf Inference v6.0（2026 年 4 月）：Blackwell 主导所有提交任务。

### FP4 的实际质量代价

NVFP4 很激进。在推理密集型工作负载（思维链、数学、长上下文代码生成）上，FP4 权重会产生可见的质量下降。按块校准可以缓解但无法消除。发布推理模型的团队通常使用 FP8 权重 + FP4 激活作为折中，或全程坚持 H200 + FP8。

规则：在承诺 NVFP4 权重之前，始终在你的评估集上验证任务质量。

### 为什么这是一个 NVIDIA 锁定决策

TRT-LLM 是 C++ + CUDA + 闭源内核。模型需要为特定 GPU SKU 编译。不支持 AMD、Intel、ARM。如果你的基础设施策略是多供应商，TRT-LLM 在 TRT-LLM 服务层上不可行——你仍然可以在混合硬件上用 vLLM 服务。如果你是纯 NVIDIA，7 倍差距值得接受锁定。

### 2026 年实践方案

对于年推理账单超过 1 亿美元的情况，在 Hopper + vLLM 上运行每年浪费 7-10 倍。将成本主导的工作负载迁移到 Blackwell + TRT-LLM + Dynamo。将实验层保留在 H100 + vLLM 上以便快速迭代模型。在生产之前验证每个 NVFP4 转换模型的质量。

### 分离式服务的加成

TRT-LLM 的分离式服务（独立的预填充和解码池）在 Phase 17 · 20 中深入介绍。在 Blackwell 上，乘数叠加：FP4 权重 × MTP 加速 × 分离式部署 × 缓存感知路由。7 倍数字假设这个完整技术栈。

## 动手实践

`code/main.py` 计算三种技术栈下的 HBM 占用、解码吞吐量（内存带宽受限区间）和每百万 token 成本：H100 + BF16 + vLLM、H100 + FP8 + vLLM、B200 + NVFP4/FP8 + TRT-LLM。运行它以查看叠加效果以及每项改变贡献了多少差距。

## 产出技能

本课产出 `outputs/skill-trtllm-blackwell-advisor.md`。给定工作负载、模型大小和年度 token 量，决定 Blackwell + TRT-LLM 技术栈是否值得接受 NVIDIA 锁定。

## 练习

1. 运行 `code/main.py`。对于有 30% 激活参数的 120B MoE，计算 H100 BF16、H100 FP8 和 B200 NVFP4/FP8 上内存带宽受限的解码吞吐量。最大跳跃来自哪里？
2. 一个客户在 H100 + vLLM 上每年花费 200 万美元。给定 7 倍经济差距，他们需要购买多少 Blackwell GPU 才能在 12 个月内摊销迁移到 TRT-LLM 的成本？
3. NVFP4 权重转换后，你在 MATH 上看到 3 分的精度下降。命名两条恢复路径：一条质量优先（保留 FP8 权重），一条成本优先（用域内数据校准）。
4. 阅读 MLPerf v6.0 推理结果。哪个任务 Blackwell 对 Hopper 的差距最小，为什么？
5. 计算 128k 上下文下 NVFP4 权重 + FP8 KV 缓存的 405B 模型所需 HBM。能放在单台 GB200 NVL72 节点上吗？

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| FP8 | "8 位浮点" | 8 位浮点；因动态范围用于 KV 缓存和注意力 |
| NVFP4 | "4 位微缩放" | NVIDIA 的 4 位微缩放 FP 格式；Blackwell 上的权重和激活 |
| MXFP8 | "MX 八位" | 微缩放 FP8 变体；Blackwell Tensor Core 上硬件加速 |
| Day-0 FP4 | "直接发布 FP4 权重" | 模型提供商直接发布 FP4 权重；无训练后转换步骤 |
| MTP | "多 token 预测" | TRT-LLM 的集成投机解码草稿（Phase 17 · 05） |
| 分离式服务（Disaggregated serving） | "拆分预填充/解码" | 预填充和解码在独立 GPU 池；KV 通过 NVLink/IB 传输 |
| 全对全（All-to-all） | "MoE 专家通信" | 将 token 路由到专家 GPU 的通信模式；NVLink 5 降低 3 倍 |
| InferenceX | "SemiAnalysis 推理基准" | 2026 年行业认可的每 token 成本基准 |

## 延伸阅读

- [NVIDIA——Blackwell Ultra MLPerf Inference v6.0](https://developer.nvidia.com/blog/nvidia-blackwell-ultra-sets-new-inference-records-in-mlperf-debut/) — 2026 年 4 月 MLPerf 结果
- [NVIDIA——Blackwell 上的 MoE 推理](https://developer.nvidia.com/blog/delivering-massive-performance-leaps-for-mixture-of-experts-inference-on-nvidia-blackwell/) — NVLink 5 全对全和 MoE 内核
- [TensorRT-LLM 概述](https://nvidia.github.io/TensorRT-LLM/overview.html) — 官方引擎文档
- [NVIDIA——介绍 Dynamo](https://developer.nvidia.com/blog/introducing-nvidia-dynamo-a-low-latency-distributed-inference-framework-for-scaling-reasoning-ai-models/) — TRT-LLM 之上的分离式编排
- [MLPerf 推理](https://mlcommons.org/benchmarks/inference-datacenter/) — 发布 Blackwell 数字的基准测试套件
