# LLM API 负载测试——为什么 k6 和 Locust 会说谎

> 传统负载测试工具并非为流式响应、可变输出长度、token 级指标或 GPU 饱和设计。两个陷阱会坑倒大多数团队。GIL 陷阱：Locust 的 token 级测量在 Python GIL 下运行分词，与高并发下的请求生成相互竞争；分词积压会夸大报告的令牌间延迟——瓶颈是你的客户端，不是服务器。提示词均一性陷阱：在循环中使用相同提示词只测试了 token 分布上的一个点；真实流量有可变长度和多样化的前缀匹配。LLMPerf 通过 `--mean-input-tokens` + `--stddev-input-tokens` 解决了这个问题。2026 年工具映射：LLM 专用工具（GenAI-Perf、LLMPerf、LLM-Locust、guidellm）用于 token 级精度；**k6 v2026.1.0** + **k6 Operator 1.0 GA（2025 年 9 月）**——流式感知、通过 TestRun/PrivateLoadZone CRD 实现 Kubernetes 原生分布式，最适合 CI/CD 卡口；Vegeta 用于 Go 语言恒定速率饱和测试；Locust 2.43.3 只有配合 LLM-Locust 扩展才能用于流式。负载模式：稳态、爬坡、尖峰（自动扩缩容测试）、浸泡（内存泄漏）。

**类型：** 构建
**编程语言：** Python（标准库，玩具真实提示词生成器 + 延迟收集器）
**前置知识：** Phase 17 · 08（推理指标）、Phase 17 · 03（GPU 自动扩缩容）
**预计时间：** 约 75 分钟

## 学习目标

- 解释两个反模式（GIL 陷阱、提示词均一性陷阱），说明它们为何使通用负载测试工具在 LLM API 上说谎。
- 针对不同用途选择工具：LLMPerf（性能跑分）、k6 + 流式扩展（CI 卡口）、guidellm（大规模合成测试）、GenAI-Perf（NVIDIA 参考工具）。
- 设计四种负载模式（稳态、爬坡、尖峰、浸泡），并说出每种模式捕获的故障类型。
- 使用输入 token 的均值 + 标准差构建真实提示词分布，而非固定长度。

## 问题背景

你用 k6 测试了 LLM 端点，500 并发用户，撑住了。你上线了。在生产环境 200 个实际用户时服务崩了——P99 TTFT 爆炸，GPU 跑满了。

发生了两件事。第一，k6 发送了 500 个相同的提示词——你的请求合并和前缀缓存让它看起来像在处理 500 个并发解码，但实际上只在处理一个。第二，k6 无法像用户体验的那样追踪流式响应上的令牌间延迟；它看到一个 HTTP 连接，不是 500 个以不同间隔到达的 token。

LLM 负载测试是一门独立的学科。

## 核心概念

### GIL 陷阱（Locust）

Locust 使用 Python，在 GIL 下在客户端运行分词。高并发下分词器排在请求生成后面排队。报告的令牌间延迟包含了客户端分词积压。你以为服务器慢，其实是测试框架的问题。

修复：LLM-Locust 扩展将分词移到独立进程，或使用编译语言框架（k6、使用 tokenizers.rs 的 LLMPerf）。

### 提示词均一性陷阱

所有已知负载测试工具都让你配置一个提示词。在 10,000 次迭代的循环测试中，每次发送完全相同的提示词。服务器每次看到相同前缀——前缀缓存命中率趋近 100%，吞吐量看起来很棒。

修复：从提示词分布中采样。LLMPerf 使用 `--mean-input-tokens 500 --stddev-input-tokens 150`——多样的长度，多样的内容。

### 四种负载模式

1. **稳态（Steady-state）** — 恒定 RPS 持续 30-60 分钟。捕获：基线性能回归。
2. **爬坡（Ramp）** — 15 分钟内 RPS 从 0 线性增加到目标。捕获：容量临界点、预热异常。
3. **尖峰（Spike）** — 突然 3-10 倍 RPS 持续 2 分钟后恢复。捕获：自动扩缩容延迟、队列饱和、冷启动影响。
4. **浸泡（Soak）** — 稳态持续 4-8 小时。捕获：内存泄漏、连接池漂移、可观测性溢出。

### 2026 年工具映射

**LLMPerf**（Anyscale）— Python 但有 Rust 支持的分词。均值/标准差提示词。流式感知。性能跑分的最佳默认选择。

**NVIDIA GenAI-Perf** — NVIDIA 参考工具。使用 Triton 客户端；全面的指标覆盖。注意其 ITL 排除 TTFT；LLMPerf 的 ITL 包含 TTFT。两个工具对同一服务器产生不同的 TPOT 报告。

**LLM-Locust**（TrueFoundry）— 修复 GIL 陷阱的 Locust 扩展。熟悉的 Locust DSL + 流式指标。

**guidellm** — 大规模合成基准测试。

**k6 v2026.1.0** + **k6 Operator 1.0 GA（2025 年 9 月）**：
- k6 本身（Go，编译型，无 GIL）增加了流式感知指标。
- k6 Operator 使用 TestRun / PrivateLoadZone CRD 实现 Kubernetes 原生分布式测试。
- 最适合 CI/CD 卡口和 SLA 测试。

**Vegeta** — Go 语言，比 k6 更简单。恒定速率 HTTP 饱和测试。不具备 LLM 感知能力，但适合网关/速率限制测试。

**Locust 2.43.3 原版** — 对 LLM 有 GIL 陷阱。只有配合 LLM-Locust 扩展才能用。

### CI 中的 SLA 卡口

在 PR 上运行 k6：

- 每次在基线 RPS 下运行 30-50 次迭代。
- 卡口：P50/P95 TTFT，5xx < 5%，TPOT 在阈值以下。
- 违反时中断构建。

### 真实提示词分布

从真实流量样本（如果有）或已发布分布（如聊天的 ShareGPT 提示词，代码的 HumanEval）构建。将均值 + 标准差传给 LLMPerf。不惜一切避免使用单一提示词的循环测试。

### 需要记住的数字

- k6 Operator 1.0 GA：2025 年 9 月。
- k6 v2026.1.0：流式感知指标。
- 典型 LLMPerf 运行：在并发 X 下 100-1000 次请求。
- 典型 CI 卡口：每次 PR 30-50 次迭代。
- 四种模式：稳态、爬坡、尖峰、浸泡。

## 动手实践

`code/main.py` 用真实提示词分布模拟负载测试，测量有效 TPOT，并演示均一提示词陷阱。

## 产出技能

本课产出 `outputs/skill-load-test-plan.md`。给定工作负载和 SLA，选择工具并设计四种负载模式。

## 练习

1. 运行 `code/main.py`。对比均一分布和真实分布——差距在哪里？
2. 编写 k6 脚本用于 CI 卡口：100 并发、运行 5 分钟，TTFT P95 < 800ms。
3. 浸泡测试显示内存每小时增长 50MB。列出三个原因以及区分它们所需的监控手段。
4. 从 10 RPS 尖峰到 100 RPS。如果部署了 Karpenter + vLLM 生产技术栈（Phase 17 · 03 + 18），预期恢复时间是多少？
5. GenAI-Perf 报告 TPOT=6ms；LLMPerf 对同一服务器报告 TPOT=11ms。解释原因。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| LLMPerf | "LLM 测试框架" | Anyscale 基准测试工具，流式感知 |
| GenAI-Perf | "NVIDIA 工具" | NVIDIA 参考测试框架 |
| LLM-Locust | "LLM 版 Locust" | 修复 GIL 陷阱的 Locust 扩展 |
| guidellm | "合成基准" | 大规模合成测试工具 |
| k6 Operator | "K8s 版 k6" | 基于 CRD 的分布式 k6 |
| GIL 陷阱（GIL trap） | "Python 客户端开销" | 分词积压夸大报告的延迟 |
| 提示词均一性陷阱（Prompt-uniformity trap） | "单提示词谎言" | 相同提示词循环命中缓存，夸大吞吐量 |
| 稳态（Steady-state） | "恒定负载" | 固定 RPS 持续 N 分钟 |
| 爬坡（Ramp） | "线性增加" | 在持续时间内从 0 增加到目标 |
| 尖峰（Spike） | "突发测试" | 突然乘以倍数然后恢复 |
| 浸泡（Soak） | "长时间测试" | 数小时用于泄漏检测 |

## 延伸阅读

- [TianPan——LLM 应用负载测试](https://tianpan.co/blog/2026-03-19-load-testing-llm-applications)
- [PremAI——2026 年 LLM 负载测试](https://blog.premai.io/load-testing-llms-tools-metrics-realistic-traffic-simulation-2026/)
- [NVIDIA NIM——LLM 推理基准测试入门](https://docs.nvidia.com/nim/large-language-models/1.0.0/benchmarking.html)
- [TrueFoundry——LLM-Locust](https://www.truefoundry.com/blog/llm-locust-a-tool-for-benchmarking-llm-performance)
- [LLMPerf](https://github.com/ray-project/llmperf)
- [k6 Operator](https://github.com/grafana/k6-operator)
