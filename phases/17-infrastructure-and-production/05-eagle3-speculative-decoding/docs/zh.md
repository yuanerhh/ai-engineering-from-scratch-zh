# 生产环境中的 EAGLE-3 投机解码

> 投机解码将快速草稿模型与目标模型配对。草稿提议 K 个 token；目标在一次前向传递中验证；被接受的 token 实际上是免费的。2026 年，EAGLE-3 是生产级变体——它在目标模型的隐藏状态而非原始 token 上训练草稿头，将通用聊天的接受率 alpha 推入 0.6-0.8 区间。正确的问题不是"草稿有多快"，而是"在我的流量上 alpha 是多少？"如果 alpha 低于约 0.55，投机解码在高并发下会产生净负效果，因为每次被拒绝的草稿都需要额外的目标前向传递。本课教你先测量 alpha，再启用标志。

**类型：** 学习
**编程语言：** Python（标准库，玩具接受率模拟器）
**前置知识：** Phase 17 · 04（vLLM 服务内部原理）、Phase 10 · 18（多 token 预测）
**预计时间：** 约 60 分钟

## 学习目标

- 命名投机解码的三代演进，并解释 EAGLE-3 相比 EAGLE-2 和经典草稿模型的改进。
- 定义接受率 alpha，根据 alpha 和 K（草稿长度）计算预期加速比，并识别你的目标并发下的盈亏平衡 alpha。
- 解释为什么 2026 年 vLLM 中投机解码是可选项（非默认），以及为什么不测量 alpha 就启用是生产反模式。
- 写出测量方案：使用哪个基准测试、哪种提示词分布、哪个并发点、依据哪个指标决策。

## 问题背景

解码是内存带宽受限的。在运行 Llama 3.3 70B FP8 的 H100 上，每个解码 token 读取约 140 GB/s 的权重并发出一个 token。GPU 计算在解码期间几乎是空闲的——瓶颈是 HBM 带宽，而非矩阵乘法吞吐量。

投机解码利用这一差距。用廉价的草稿模型生成 K 个候选 token，然后让目标模型在一次前向传递中验证所有 K 个。每个被验证的 token 实际上是免费的（摊销到目标本来需要执行的批次 K 前向传递中）。

经典草稿模型方法使用同一家族的较小模型（用 Llama 3.2 1B 为 Llama 3.3 70B 起草）。它有效，但接受率一般——较小模型的分布与目标分布存在偏差。EAGLE、EAGLE-2、EAGLE-3 在目标模型内部状态上直接训练轻量级草稿头，使草稿的分布更紧密地跟踪目标。这就是为什么 alpha 从草稿模型的 0.4 提升到 EAGLE-3 的 0.6-0.8。

陷阱是：EAGLE-3 在 2026 年的 vLLM 中是可选项。必须显式设置 `speculative_config`。没有该标志，就没有加速。不测量真实流量的 alpha 就启用它的团队，往往会看到尾部延迟变差，而非变好。

## 核心概念

### 投机解码实际带来什么

没有投机解码时，每 token 成本是一次目标前向传递。在草稿长度 K 和接受率 alpha 下使用投机解码，每次目标前向传递的预期 token 数为 `1 + K * alpha`。加速比为 `(1 + K * alpha) / (1 + epsilon)`，其中 epsilon 是草稿加验证的开销。对于 K=5、alpha=0.7：`(1 + 5*0.7) / (1 + 0.1) = 4.5 / 1.1 = 4.1x`。实际数字集中在 2-3x，因为生产流量上 alpha 很少那么高，而且 epsilon 在高批次大小下会增长。

### 为什么 alpha 是唯一重要的指标

被拒绝的 token 不会消失——它们强制对第一个被拒绝的 token 进行第二次目标前向传递。在 alpha 降至 0.4 的工作负载上，你要付出草稿开销加验证加重跑的代价。在高并发（比如 256 并发）下，解码批次已经足够大，"仅目标"与"目标加验证"之间的内存带宽差距缩小。在大多数 2026 年硬件上，alpha 低于 0.55 时，投机解码是净负效果。

Alpha 因工作负载而异。在 ShareGPT 风格的通用聊天上，用 ShareGPT 训练的 EAGLE-3 达到 0.6-0.8。在领域特定流量（代码、医疗、法律）上，在通用数据上训练的草稿头下降到 0.4-0.6。训练领域特定草稿头可以恢复 alpha——与目标微调相比，这是一项轻量、快速的训练工作。

### EAGLE 各代一览

- **经典草稿模型**：同家族的较小模型。Alpha 0.3-0.5。基础设施简单——加载两个模型，草稿每次目标前向传递执行 K 次前向。
- **EAGLE-1（2024年）**：在目标隐藏状态（最后一层）上训练的单一草稿头。Alpha 约 0.5-0.6。目标之上的小参数开销。
- **EAGLE-2（2025年）**：自适应草稿长度和基于树的草稿（在一次目标传递中验证多个分支）。Alpha 约 0.6-0.7。更复杂的草稿调度器。
- **EAGLE-3（2025-2026年）**：在多个目标层（而非仅最后一层）上训练的草稿头，对齐更好。通用聊天 alpha 约 0.6-0.8。

### 2026 年生产方案

1. 纯粹部署目标模型。在目标并发下测量基线首 token 延迟、ITL 和吞吐量。
2. 通过 vLLM `speculative_config` 启用 EAGLE-3 草稿。重新运行基准测试。
3. 记录接受率 alpha。vLLM V1 将其报告为 `spec_decode_metrics.accepted_tokens_per_request`。除以请求的草稿长度得到 alpha。
4. 如果生产流量分布下 alpha < 0.55，禁用投机解码或训练领域特定 EAGLE-3 草稿。
5. 在生产并发下重新运行。确认 P99 ITL 没有变差。

### 生产陷阱：P99 尾部延迟

均值 ITL 随投机解码下降。如果不调优，P99 可能变差。被拒绝的草稿触发两次传递序列（草稿 + 验证失败 + 重跑）。在满批次下，这两次传递串行执行。要监控 P99 ITL，而非 P50。

### EAGLE-3 已在哪里部署

Google 在 2025 年将投机解码部署在 AI Overviews 中（同等质量，更快响应）。vLLM V1 将 `speculative_config` 作为文档化接口；V1 中的 N-gram GPU 投机解码是与分块预填充兼容的变体。SGLang 支持 EAGLE-3 作为前缀密集工作负载的推荐草稿路径。

### 一行盈亏平衡数学

预期加速比：`S(alpha, K) = (1 + K*alpha) / (1 + verify_overhead)`。令 `S = 1` 求解 alpha：`alpha_breakeven = verify_overhead / K`。对于典型 verify_overhead 约 0.15 和 K=5：`alpha_breakeven = 0.03`。但这是裸解码数学。在高并发下，验证开销上升，解码批次已经在序列间摊销内存读取，所以实际 alpha_breakeven 在实践中升至约 0.45-0.55。

### 何时不使用投机解码

- 延迟无关紧要的批量离线生成。使用纯目标模型。
- 非常短的输出（50 token 以下）。草稿开销和验证成本占主导。
- 没有领域训练草稿头的专业领域。Alpha 太低。
- vLLM v0.18.0 加草稿模型投机解码加 `--enable-chunked-prefill`。此组合无法编译。有文档记录的例外是 V1 中的 N-gram GPU 投机解码。

## 动手实践

`code/main.py` 在一系列 alpha 值和草稿长度 K 上模拟带/不带投机解码的解码循环。打印盈亏平衡 alpha、测量的加速比和尾部行为。在几个（alpha, K）组合上运行，准确看到投机解码何时不再划算。

## 产出技能

本课产出 `outputs/skill-eagle3-rollout.md`。给定目标模型、流量分布描述和并发目标，产出分阶段的 EAGLE-3 上线计划——基准测试基线、启用配置、测量 alpha、以 alpha >= 0.55 为门控、监控 P99 ITL。

## 练习

1. 运行 `code/main.py`。在 K=5 时，需要多大 alpha 才能实现 2x 加速？3x 加速？对 verify_overhead 的敏感度如何？
2. 假设生产流量分为 70% 通用聊天、30% 代码。通用聊天用 ShareGPT 训练的 EAGLE-3 达到 alpha 0.7；代码达到 alpha 0.4。混合 alpha 是多少，投机解码是净正效果吗？
3. 阅读 vLLM `speculative_config` 文档。命名三种模式（草稿模型、EAGLE、N-gram），哪种与分块预填充兼容？
4. 你看到启用 EAGLE-3 后均值 ITL 下降 25%，但 P99 ITL 上升 15%。诊断并提出缓解措施。
5. 计算 Llama 3.3 70B 的 EAGLE-3 草稿头的内存成本。与使用 Llama 3.2 1B 作为经典草稿相比如何？

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| 投机解码（Speculative decoding） | "草稿加验证" | 用廉价模型提议 K 个 token，在一次目标前向中验证所有 K 个 |
| 接受率 alpha（Acceptance rate alpha） | "投机接受率" | 目标接受的草稿 token 比例；唯一重要的指标 |
| 草稿长度 K（Draft length K） | "投机 K" | 每次目标前向中草稿提议的 token 数；通常 4-8 |
| 验证开销 epsilon（Verify overhead epsilon） | "投机开销" | 相比纯目标前向的验证和重跑额外成本；随批次增长 |
| EAGLE-3 | "最新 EAGLE" | 2025-2026 年变体；在多个目标层上训练草稿头；通用聊天 alpha 0.6-0.8 |
| `speculative_config` | "vLLM 投机配置" | vLLM V1 中的显式可选项；没有该配置就没有加速 |
| N-gram 投机解码（N-gram spec decode） | "N-gram 草稿" | 在提示词中使用 N-gram 查找的 GPU 侧草稿；兼容分块预填充 |
| 盈亏平衡 alpha（Break-even alpha） | "零收益 alpha" | 投机解码零加速的 alpha；在生产并发下监控 |
| 草稿被拒两次传递（Rejected-draft two-pass） | "重跑成本" | 草稿被拒时两次目标前向传递；驱动 P99 尾部延迟 |

## 延伸阅读

- [vLLM——投机解码文档](https://docs.vllm.ai/en/latest/features/spec_decode/) — `speculative_config` 和 V1 中分块预填充兼容性的权威来源
- [vLLM 投机配置 API](https://docs.vllm.ai/en/latest/api/vllm/config/speculative/) — 精确的字段集
- [EAGLE 论文（arXiv:2401.15077）](https://arxiv.org/abs/2401.15077) — 原始 EAGLE 草稿头公式
- [EAGLE-2 论文（arXiv:2406.16858）](https://arxiv.org/abs/2406.16858) — 自适应草稿和树
- [UC Berkeley EECS-2025-224](https://www2.eecs.berkeley.edu/Pubs/TechRpts/2025/EECS-2025-224.html) — 带投机解码的高效 LLM 系统
- [BentoML——投机解码](https://bentoml.com/llm/inference-optimization/speculative-decoding) — 生产上线检查清单
