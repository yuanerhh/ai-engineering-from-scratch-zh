# 生产量化——AWQ、GPTQ、GGUF K-quant、FP8、MXFP4/NVFP4

> 量化格式不是通用选择——它是硬件、服务引擎和工作负载的函数。GGUF Q4_K_M 或 Q5_K_M 主导 CPU 和边缘端，通过 llama.cpp 和 Ollama 交付。GPTQ 在 vLLM 中多 LoRA 场景胜出。带 Marlin-AWQ 内核的 AWQ 在 7B 级模型上达到约 741 token/秒，INT4 格式中 Pass@1 最好——2026 年数据中心生产默认值。FP8 在 Hopper、Ada 和 Blackwell 上是中间地带——接近无损且被广泛支持。NVFP4 和 MXFP4（Blackwell 微缩放）是激进的，需要按块验证。两个陷阱会坑团队：校准数据集必须与部署域匹配；KV 缓存与权重量化是独立的——"我的模型现在只有 4GB"这种 AWQ 课程忘记了生产批次大小下 10-30GB 的 KV 缓存。

**类型：** 学习
**编程语言：** Python（标准库，跨格式的玩具内存与吞吐量比较器）
**前置知识：** Phase 10 · 13（量化基础）、Phase 17 · 04（vLLM 服务内部原理）
**预计时间：** 约 75 分钟

## 学习目标

- 命名六种生产量化格式及其在 2026 年的适用场景。
- 根据硬件（CPU vs GPU，Hopper vs Blackwell）、引擎（vLLM、TRT-LLM、llama.cpp）和工作负载（常规聊天、推理、多 LoRA）选择格式。
- 计算所选格式节省的权重内存，以及未触及的 KV 缓存。
- 命名导致量化模型在领域流量上性能下降的校准数据集陷阱。

## 问题背景

量化减少内存和 HBM 带宽，这正是解码所需要的。FP16 70B 模型权重为 140GB。将权重量化到 INT4（AWQ 或 GPTQ），模型降至 35GB——可放入一块 H100，并留有 KV 缓存空间。这很重要，因为在 128 并发、2k 上下文下，仅 KV 缓存就需要 20-30GB。

但量化不是免费的。激进量化会降低质量，尤其在推理密集任务上。不同格式适用于不同引擎。不同硬件原生支持不同精度。2026 年的格式大杂烩是真实存在的，你无法照搬他人的选择——必须根据你的技术栈选择。

## 核心概念

### 六种格式

| 格式 | 位数 | 适用场景 | 引擎 |
|------|------|---------|------|
| GGUF Q4_K_M / Q5_K_M | 4-5 | CPU、边缘、笔记本 | llama.cpp、Ollama |
| GPTQ | 4-8 | vLLM 上的多 LoRA | vLLM、TGI |
| AWQ | 4 | 数据中心 GPU 生产 | vLLM（Marlin-AWQ）、TGI |
| FP8 | 8 | Hopper/Ada/Blackwell 数据中心 | vLLM、TRT-LLM、SGLang |
| MXFP4 | 4 | Blackwell 多用户 | TRT-LLM |
| NVFP4 | 4 | Blackwell 多用户 | TRT-LLM |

### GGUF——CPU/边缘默认值

GGUF 是文件格式，而非本质上的量化方案——它将 K-quant 变体（Q2_K、Q3_K_M、Q4_K_M、Q5_K_M、Q6_K、Q8_0）打包在一个容器中。Q4_K_M 和 Q5_K_M 是生产默认值——4-5 位下接近 BF16 质量。CPU 或边缘端服务的最佳选择，因为 llama.cpp 是迄今最快的 CPU 推理引擎。

vLLM 中的吞吐量损失：约 93 token/秒（7B）——该格式未针对 GPU 内核优化。当部署目标是 CPU/边缘时使用 GGUF，否则不用。

### GPTQ——vLLM 中的多 LoRA

GPTQ 是带校准步骤的训练后量化算法。Marlin 内核使其在 GPU 上速度很快（相比非 Marlin GPTQ 提速 2.6 倍）。7B 模型约 712 token/秒。

独特优势：GPTQ-Int4 在 vLLM 中支持 LoRA 适配器。如果你在服务一个基础模型加 10-50 个微调变体（每个作为 LoRA），GPTQ 是你的路径。截至 2026 年初，NVFP4 尚不支持 LoRA。

### AWQ——数据中心 GPU 默认值

激活感知权重量化（Activation-aware Weight Quantization）。在量化过程中保护约 1% 最显著的权重。Marlin-AWQ 内核：相比朴素方法提速 10.9 倍。7B 模型约 741 token/秒，INT4 格式中 Pass@1 最好。

新 GPU 服务选择 AWQ，除非需要多 LoRA（GPTQ）或激进 Blackwell FP4（NVFP4）。

### FP8——可靠的中间地带

8 位浮点。接近无损。被广泛支持。Hopper Tensor Core 原生加速 FP8。Blackwell 继承。当质量不可妥协时（推理、医疗、代码生成），FP8 是 2026 年的安全默认值。内存节省是 INT4 的一半，但质量风险远低。

### MXFP4 / NVFP4——Blackwell 激进方案

微缩放 FP4。每个权重块有自己的缩放因子。激进但在 Blackwell Tensor Core 上硬件加速。相比 FP8 每 token 字节数减半——Phase 17 · 07 中的经济优势。

注意事项：
- 截至 2026 年初尚不支持 LoRA。
- 推理密集工作负载上质量下降可见。
- 每个模型在评估集上单独验证。

### 校准陷阱

AWQ 和 GPTQ 需要校准数据集——通常是 C4 或 WikiText。对于领域模型（代码、医疗、法律），用通用网络文本校准会让算法对哪些权重需要保护做出错误决策。HumanEval 上的 Pass@1 可能下降几个点。

修复：用领域内数据校准。通常几百个领域样本就够了。发布前在评估集上测试。

### KV 缓存陷阱

AWQ 将权重压缩到 4 位。KV 缓存是独立的，保持在 FP16/FP8。对于带 AWQ 的 70B 模型：

- 权重：约 35GB（INT4，从 140GB 压缩）。
- 128 并发 × 2k 上下文的 KV 缓存：约 20GB。
- 激活：约 5GB。
- 总计：约 60GB——适合 H100 80GB。

朴素地"我把模型量化到 4GB"忘记了另外 30-50GB。全面规划 HBM 预算。

另外，KV 缓存量化（FP8 KV 或 INT8 KV）是一个不同的选择，有自己的权衡——它直接影响注意力精度，不是免费的优化。

### AWQ INT4 对推理有害

思维链、数学、长上下文代码生成——这些会受到激进量化的明显影响。AWQ INT4 在 MATH 上损失约 3-5 分。对于推理密集工作负载，使用 FP8 或 BF16；接受内存成本。

### 2026 年选择指南

- CPU/边缘服务：GGUF Q4_K_M。完毕。
- GPU 服务，常规聊天，无 LoRA：AWQ。
- GPU 服务，多 LoRA：GPTQ + Marlin。
- 推理工作负载：FP8。
- Blackwell 数据中心，已验证质量：NVFP4 + FP8 KV。
- 不确定：在每个候选格式上运行 1000 样本评估。

## 动手实践

`code/main.py` 计算一系列模型大小下六种格式的内存占用（权重 + KV + 激活）和相对吞吐量。显示 KV 缓存何时占主导，权重压缩何时有价值，以及 FP8 何时是安全选择。

## 产出技能

本课产出 `outputs/skill-quantization-picker.md`。给定硬件、模型大小、工作负载类型和质量容忍度，选择格式并产出校准/验证方案。

## 练习

1. 运行 `code/main.py`。对于 128 并发、2k 上下文的 70B 模型，计算每种格式的总 HBM 使用量。哪种格式能放入一块 H100 80GB？
2. 你有一个 7B 编程模型。选择格式并说明理由。如果你对质量容忍度判断有误，恢复路径是什么？
3. 计算为医疗领域模型校准 AWQ 所需的校准数据集大小。为什么更多数据并不总是更好？
4. 阅读 Marlin-AWQ 内核论文或发布说明。用三句话解释为什么 AWQ 在 7B 上达到 741 token/秒而原始 GPTQ 约为 712。
5. 何时有意义将 AWQ 权重与 FP8 KV 缓存组合，而非保持 KV 在 BF16？

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| GGUF | "llama.cpp 格式" | 打包 K-quant 变体的文件格式；CPU/边缘默认值 |
| Q4_K_M | "Q4 K M" | 4 位 K-quant 中等；生产 GGUF 默认值 |
| GPTQ | "GPTQ" | 带校准的训练后 INT4；在 vLLM 中支持 LoRA |
| AWQ | "AWQ" | 激活感知 INT4；Marlin 内核；INT4 中 Pass@1 最好 |
| Marlin 内核（Marlin kernels） | "快速 INT4 内核" | Hopper 上 INT4 的自定义 CUDA 内核；10 倍加速 |
| FP8 | "8 位浮点" | Hopper/Ada/Blackwell 上的安全精度默认值 |
| MXFP4 / NVFP4 | "微缩放 4 位" | Blackwell 带每块缩放因子的 4 位 FP |
| 校准数据集（Calibration dataset） | "校准数据" | 用于选择量化参数的输入文本；必须与域匹配 |
| KV 缓存量化（KV cache quantization） | "KV INT8" | 与权重独立的选择；影响注意力精度 |

## 延伸阅读

- [VRLA Tech——LLM 量化 2026](https://vrlatech.com/llm-quantization-explained-int4-int8-fp8-awq-and-gptq-in-2026/) — 对比基准测试
- [Jarvis Labs——vLLM 量化完整指南](https://jarvislabs.ai/blog/vllm-quantization-complete-guide-benchmarks) — 按格式分的吞吐量数字
- [PremAI——GGUF vs AWQ vs GPTQ vs bitsandbytes 2026](https://blog.premai.io/llm-quantization-guide-gguf-vs-awq-vs-gptq-vs-bitsandbytes-compared-2026/) — 逐格式选择指南
- [vLLM 文档——量化](https://docs.vllm.ai/en/latest/features/quantization/index.html) — 支持的格式和标志
- [AWQ 论文（arXiv:2306.00978）](https://arxiv.org/abs/2306.00978) — AWQ 原始公式
- [GPTQ 论文（arXiv:2210.17323）](https://arxiv.org/abs/2210.17323) — GPTQ 原始公式
