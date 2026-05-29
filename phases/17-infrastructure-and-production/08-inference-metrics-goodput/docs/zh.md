# 推理指标——TTFT、TPOT、ITL、Goodput、P99

> 四个指标决定推理部署是否有效。TTFT 是预填充加队列加网络。TPOT（等价于 ITL）是每 token 内存带宽受限的解码成本。端到端延迟是 TTFT 加 TPOT 乘以输出长度。吞吐量是整个集群每秒聚合的 token 数。但对产品真正重要的是 goodput——同时满足所有 SLO 的请求比例。高吞吐量 + 低 goodput 意味着你在处理永远无法及时到达用户的 token。2026 年 TRT-LLM 上 Llama-3.1-8B-Instruct 的参考数字：均值 TTFT 162ms，均值 TPOT 7.33ms，均值端到端 1093ms。始终报告 P50、P90、P99——永远不要只报告均值。还要警惕测量陷阱：GenAI-Perf 将 TTFT 排除在 ITL 计算之外，LLMPerf 将其包含在内；两个工具对同一次运行的 TPOT 意见不一致。

**类型：** 学习
**编程语言：** Python（标准库，玩具百分位计算器和 goodput 报告器）
**前置知识：** Phase 17 · 04（vLLM 服务内部原理）
**预计时间：** 约 60 分钟

## 学习目标

- 精确定义 TTFT、TPOT、ITL、E2E、吞吐量和 goodput，并说明每个指标测量的组件。
- 解释为什么均值对 LLM 服务是错误的统计量，以及如何读取 P50/P90/P99。
- 构建多约束 SLO（例如 TTFT<500ms 且 TPOT<15ms 且 E2E<2s）并计算其 goodput。
- 命名两个对同一次运行的 TPOT 意见不一致的基准工具，并解释原因。

## 问题背景

"我们的吞吐量是每秒 15,000 个 token。"然后呢？如果 40% 的请求超过 2 秒端到端延迟，用户已经放弃了会话。吞吐量单独不能告诉你产品是否正常运行。

推理有多个延迟维度，每个的失败方式都不同。预填充是计算密集型的，随提示词长度扩展。解码是内存带宽受限的，随批次大小扩展。排队延迟是操作问题。网络是物理距离问题。你需要针对每个的独立指标，需要百分位数，还需要一个综合的"用户是否得到了他们期望的"——这就是 goodput。

## 核心概念

### TTFT——首 token 时间

`TTFT = 队列时间 + 网络请求 + 预填充时间`

当提示词很长时，预填充占主导。在 H100 上 Llama-3.3-70B FP8 处理 32k token 提示词需要约 800ms 的纯预填充。队列时间是负载下的调度器行为。网络请求是包含 TLS 的线路时间。TTFT 是用户在任何内容流式返回之前看到的延迟。

### TPOT / ITL——令牌间延迟

同一量的多种名称。`TPOT`（每输出 token 时间）、`ITL`（令牌间延迟）、`每 token 解码延迟`——都是一回事。它是首 token 之后连续流式 token 之间的时间。

`TPOT = (解码前向时间 + 调度器开销) / 产生的 token 数`

在同款 Llama-3.3-70B H100 技术栈上启用分块预填充时，TPOT 均值约 7ms。不启用分块预填充时，相邻序列的长预填充期间，TPOT 可飙升至 50ms。监控 P99，而非均值。

### 端到端延迟

`E2E = TTFT + TPOT × 输出 token 数 + 网络响应`

对于长输出（>500 token），E2E 以 TPOT 为主。对于带长提示词的短输出，E2E 以 TTFT 为主。报告以输出长度为条件的 E2E。

### 吞吐量

`吞吐量 = 总输出 token 数 / 耗用时间`

聚合指标。告诉你集群效率。不告诉你单个请求的健康状况。

### Goodput——你真正关心的指标

`goodput = 满足 (TTFT <= a) 且 (TPOT <= b) 且 (E2E <= c) 的请求比例`

SLO 是多约束。只有当所有约束都满足时，请求才是"好的"。Goodput 是这一比例。高吞吐量 + 60% goodput 是失败。较低吞吐量 + 99% goodput 才是目标。

2026 年，goodput 是 MLPerf Inference v6.0 提交中使用的指标，也是 AI 平台提供商内部 SLA 跟踪的指标。

### 为什么均值是错误的统计量

LLM 延迟分布是右偏的。带一个长预填充邻居的解码批次可能以 TPOT 约 7ms 发出 500 个 token，以 TPOT 约 60ms 发出 20 个 token。均值 TPOT 为 9ms。P99 TPOT 为 65ms。用户经常遇到 P99——这就是他们离开的原因。

始终报告三元组（P50、P90、P99）。对于用户体验，P99 是你优化的那个。

### 参考数字——2026 年 TRT-LLM 上的 Llama-3.1-8B-Instruct

- 均值 TTFT：162ms
- 均值 TPOT：7.33ms
- 均值 E2E：1093ms
- P99 TPOT：根据分块预填充配置，变化在 10-25ms 之间。

这些是 NVIDIA 发布的参考点。随模型大小（70B 会显示 3-5 倍）、硬件（H100 vs B200 约 3 倍）和负载而变化。

### 测量陷阱

2026 年两个最常用的基准工具对同一次运行的 TPOT 意见不一致：

- **NVIDIA GenAI-Perf**：将 TTFT 排除在 ITL 计算之外。ITL 从第 2 个 token 开始。
- **LLMPerf**：包含 TTFT。ITL 从第 1 个 token 开始。

对于一个 TTFT 500ms、100 个输出 token 总解码 700ms 的请求，GenAI-Perf 报告 `ITL = 700/99 = 7.07ms`，LLMPerf 报告 `ITL = 1200/100 = 12.00ms`。工具选择改变了数字。

始终声明使用的工具。始终公布定义。

### 构建 SLO

2026 年面向消费者的 70B 聊天模型合理 SLO：

- TTFT P99 <= 800ms。
- TPOT P99 <= 25ms。
- <300 token 输出的 E2E P99 <= 3s。
- Goodput 目标 >= 99%。

企业 SLO 收紧 TTFT（200-400ms）而放松 E2E。关键是写下来，测量所有三个，并将 goodput 作为单一综合指标追踪。

### 如何测量

- 运行真实流量或真实合成流量（LLMPerf 带 `--mean-input-tokens 800 --stddev-input-tokens 300 --mean-output-tokens 150`）。
- 基准测试运行目标为 2 倍峰值并发。
- 运行 30-50 次迭代，对合并样本取百分位数。
- 发布时附带工具名称、工具版本、模型、硬件、并发、提示词分布。

## 动手实践

`code/main.py` 是一个玩具 goodput 计算器。生成合成延迟分布，应用 SLO，计算 goodput。还展示 GenAI-Perf 与 LLMPerf 在同一追踪上的 TPOT 差异。

## 产出技能

本课产出 `outputs/skill-slo-goodput-gate.md`。给定工作负载和 SLO，产出一个 CI/CD 就绪的基准测试方案，以 goodput 而非吞吐量作为部署门控。

## 练习

1. 运行 `code/main.py`。生成带 1% 尾部峰值的分布。将 P99 TPOT 从 30ms 收紧到 15ms 时，goodput 如何变化？
2. 供应商报价"H100 上 Llama 3.3 70B 每秒 15,000 token"。在信任之前提出三个问题。
3. 为什么分块预填充保护 P99 TPOT 但不改善均值 TPOT？
4. 为语音助手（首 token 被听见，而非被读取）构建消费者 SLO。哪个指标对用户最可见？
5. 阅读 LLMPerf README 和 GenAI-Perf 文档。找出两个工具不一致的另外三个指标。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| TTFT | "首 token 时间" | 队列 + 网络 + 预填充；长提示词时以预填充为主 |
| TPOT | "每输出 token 时间" | 首 token 之后每 token 的内存带宽受限解码成本 |
| ITL | "令牌间延迟" | 在大多数工具中与 TPOT 相同（并非所有——见 GenAI-Perf） |
| E2E | "端到端" | TTFT + TPOT × 输出长度；加上响应侧网络 |
| 吞吐量（Throughput） | "token/秒" | 集群效率；没有延迟百分位数时毫无意义 |
| Goodput | "SLO 达成率" | 同时满足所有 SLO 约束的请求比例 |
| P99 | "尾部延迟" | 百分之一最差延迟；用户体验指标 |
| SLO 多约束（SLO multi-constraint） | "联合约束" | 所有三个延迟边界的 AND；任一违反即请求失败 |
| GenAI-Perf vs LLMPerf | "工具陷阱" | 工具对 ITL 是否包含 TTFT 意见不一 |

## 延伸阅读

- [NVIDIA NIM——LLM 基准测试指标](https://docs.nvidia.com/nim/benchmarking/llm/latest/metrics.html) — TTFT、ITL、TPOT 的规范定义
- [Anyscale——LLM 服务基准测试指标](https://docs.anyscale.com/llm/serving/benchmarking/metrics) — 替代定义和测量方案
- [BentoML——LLM 推理指标](https://bentoml.com/llm/inference-optimization/llm-inference-metrics) — 真实部署上的实际测量
- [LLMPerf](https://github.com/ray-project/llmperf) — 基于 Ray 的开源基准测试
- [GenAI-Perf](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/client/src/c++/perf_analyzer/genai-perf/README.html) — NVIDIA 的基准测试工具
- [MLPerf 推理](https://mlcommons.org/benchmarks/inference-datacenter/) — 行业认可的基于 goodput 的基准测试
