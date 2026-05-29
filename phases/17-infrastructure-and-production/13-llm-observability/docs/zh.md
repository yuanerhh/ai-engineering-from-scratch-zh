# LLM 可观测性技术栈选择

> 2026 年可观测性市场分为两类。开发平台（LangSmith、Langfuse、Comet Opik）将监控与评估、提示词管理、会话回放捆绑在一起。网关/仪表化工具（Helicone、SigNoz、OpenLLMetry、Phoenix）专注于遥测。Langfuse 是 MIT 许可核心，OSS 平衡性强（免费云层每月 5 万事件）。Phoenix 是 Elastic License 2.0 下的 OpenTelemetry 原生工具——漂移/RAG 可视化出色，但不适合持久化生产后端。Arize AX 使用零拷贝 Iceberg/Parquet 集成，声称比整体式可观测性便宜 100 倍。LangSmith 在 LangChain/LangGraph 生态中领先，每用户每月 39 美元，企业版才能自托管。Helicone 是基于代理的方案，15-30 分钟即可完成设置，每月 10 万请求免费，但智能体追踪深度有限。常见生产模式：网关（Helicone/Portkey）+ 评估平台（Phoenix/TruLens），通过 OpenTelemetry 粘合。

**类型：** 学习
**编程语言：** Python（标准库，玩具追踪采样模拟器）
**前置知识：** Phase 17 · 08（推理指标）、Phase 14（智能体工程）
**预计时间：** 约 60 分钟

## 学习目标

- 区分开发平台（捆绑：评估 + 提示词 + 会话）与网关/遥测工具（仅追踪 + 指标）。
- 将六个主要工具（Langfuse、LangSmith、Phoenix、Arize AX、Helicone、Opik）映射到其许可证、定价和适用场景。
- 解释允许你将网关工具与独立评估平台结合的 OpenTelemetry 粘合模式。
- 命名 2026 年的成本差异化因素（Arize AX 的零拷贝方案 vs 整体式摄入），并说明约 100 倍的乘数。

## 问题背景

你上线了 LLM 功能，它能用。但你对提示词失败、工具循环、延迟回归、成本飙升或提示词缓存命中率毫无可见性。你 Google"LLM 可观测性"，得到八个工具，都声称以三种不同价位解决相同问题。

它们解决的不是同一个问题。LangSmith 回答"这个 LangGraph 运行为什么失败？"Phoenix 回答"我的 RAG 管道是否在漂移？"Helicone 回答"哪个应用在烧 token？"Langfuse 回答"我能自托管整个系统吗？"不同工具，不同受众。

选择涉及四个维度：技术栈（LangChain？原始 SDK？多供应商？）、许可证容忍度（仅 MIT？Elastic 可以？商业可以？）、预算（免费层？每月 100 美元？1000 美元？）和自托管（必须？最好有？永不？）。

## 核心概念

### 两类工具

**开发平台**将可观测性与评估、提示词管理、数据集版本控制、会话回放捆绑。你运行实验，看哪个提示词有效，对新提示词与旧胜者进行数据集回归测试。LangSmith、Langfuse、Comet Opik。

**网关/遥测工具**对推理调用进行仪表化——提示词、响应、token、延迟、模型、成本。Helicone、SigNoz、OpenLLMetry、Phoenix。极简主义。可以通过 OpenTelemetry 与独立评估工具结合。

### Langfuse——OSS 平衡

- 核心 Apache/MIT 许可；通过 Docker 自托管。
- 云免费层：每月 5 万事件。付费：团队版每月 29 美元。
- 评估、提示词管理、追踪、数据集。对四个开发平台功能都有合理覆盖。
- 适用场景：你想要 LangSmith 级别的功能，但必须自托管或保持 OSS 许可。

### Phoenix（Arize）——遥测优先，OpenTelemetry 原生

- Elastic License 2.0；自托管简单。
- RAG 和漂移可视化出色。嵌入空间散点图作为一等特性发布。
- 不是作为持久化生产后端设计的——主要是开发时可观测性。
- 适用场景：RAG 管道开发、漂移调试，与独立网关配对用于生产。

### Arize AX——规模化方案

- 商业。通过 Iceberg/Parquet 进行零拷贝数据湖集成。
- 声称在规模下比整体式可观测性（Datadog 级别）便宜约 100 倍。原理：你将追踪存储在 S3 上自己的 Parquet 中；Arize 直接读取。
- 适用场景：每天 >1000 万次追踪，已有数据湖，想要 LLM 专属仪表板而不用 Datadog 定价。

### LangSmith——LangChain/LangGraph 优先

- 商业，每用户每月 39 美元。仅企业版可自托管。
- 对 LangChain 和 LangGraph 技术栈是同类最佳。如果你不在这两个技术栈上，吸引力较小。
- 适用场景：团队已承诺 LangChain，愿意付费。

### Helicone——基于代理的最低可行方案

- 将你的 `OPENAI_API_BASE` 换成 Helicone 代理，15-30 分钟完成设置。
- MIT 许可；每月 10 万请求免费，付费版每月 20 美元起。
- 包含故障转移、缓存、速率限制——也充当网关。
- 智能体/多步骤追踪深度有限。
- 适用场景：快速启动、单栈应用、需要网关 + 可观测性合二为一。

### Opik（Comet）——OSS 开发平台

- Apache 2.0，完全 OSS。
- 功能集与 Langfuse 类似，有 Comet 传承。
- 适用场景：已在 Comet 上的 ML 团队，希望在同一面板中获得 LLM 可观测性。

### SigNoz——OpenTelemetry 优先的完整 APM

- Apache 2.0。通过 OpenTelemetry 处理通用 APM 加 LLM。
- 适用场景：跨服务和 LLM 调用的统一可观测性。

### 粘合剂：OpenTelemetry + GenAI 语义约定

OpenTelemetry 在 2025 年末发布了 GenAI 语义约定（`gen_ai.system`、`gen_ai.request.model`、`gen_ai.usage.input_tokens`）。使用 OTel 的工具可以互操作。正在形成的生产模式：

1. 从每个 LLM 调用发出带 GenAI 约定的 OTel。
2. 日常路由到网关（Helicone / Portkey）。
3. 同时发送到评估平台（Phoenix / Langfuse）进行回归测试。
4. 归档到数据湖（Iceberg）通过 Arize AX 或 DuckDB 进行长期分析。

### 陷阱：在错误层进行仪表化

在智能体框架内部进行仪表化（例如添加 LangSmith 追踪）会将你与该框架耦合。在 HTTP/OpenAI-SDK 层进行仪表化（通过 OpenLLMetry 或你的网关）是可移植的。

### 采样——你无法保留所有内容

在每天 >100 万请求时，完整追踪保留成本超过 LLM 调用本身。按规则采样：100% 错误、100% 高成本、5% 成功。始终保留聚合数据；为长尾保留原始数据。

### 需要记住的数字

- Langfuse 免费云：每月 5 万事件。
- LangSmith：每用户每月 39 美元。
- Helicone 免费：每月 10 万请求。
- Arize AX 声称：规模下比整体式便宜约 100 倍。
- OpenTelemetry GenAI 约定：2025 年发布，2026 年广泛采用。

## 动手实践

`code/main.py` 在留存策略（100% 摄入、采样、采样 + 错误）之间模拟每天 100 万次追踪。报告存储成本以及每种策略下丢失的内容。

## 产出技能

本课产出 `outputs/skill-observability-stack.md`。给定技术栈、规模、预算、许可证立场，选择工具。

## 练习

1. 你在 LangChain 上的团队想要 OSS 自托管可观测性。选择 Langfuse 或 Opik 并说明理由。
2. 每天 500 万次追踪，Datadog 报价每月 15 万美元，计算 Arize AX 的盈亏平衡。
3. 设计你的组织指南应该在每个 LLM 调用上强制要求的 OpenTelemetry GenAI 属性集。
4. 论证 Phoenix 单独是否足以用于生产。何时不够用？
5. Helicone 有 20ms 代理开销。在 P99 TTFT 300ms 时，这可以接受吗？如果 SLA 是 100ms 呢？

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| OpenLLMetry | "LLM 的 OTel" | 面向 LLM 的开源 OpenTelemetry 仪表化 |
| GenAI 约定（GenAI conventions） | "OTel 属性" | LLM 调用的标准 OTel 属性名称 |
| LangSmith | "LangChain 可观测性" | 与 LangChain 生态系统捆绑的商业平台 |
| Langfuse | "OSS LangSmith" | 功能集类似的 MIT OSS |
| Phoenix | "Arize 开发工具" | OpenTelemetry 原生开发/评估平台 |
| Arize AX | "规模化可观测性" | 商业零拷贝 Iceberg/Parquet 可观测性 |
| Helicone | "代理可观测性" | 收集 LLM 遥测 + 网关功能的 HTTP 代理 |
| Opik | "Comet LLM" | 来自 Comet 的 Apache 2.0 OSS 开发平台 |
| 会话回放（Session replay） | "追踪重跑" | 重播带工具调用的完整智能体会话 |
| 评估（Eval） | "离线测试" | 在标记数据集上运行候选模型/提示词 |

## 延伸阅读

- [SigNoz——2026 年 LLM 可观测性工具 Top 榜](https://signoz.io/comparisons/llm-observability-tools/)
- [Langfuse——Arize AX 替代方案分析](https://langfuse.com/faq/all/best-phoenix-arize-alternatives)
- [PremAI——设置 Langfuse、LangSmith、Helicone、Phoenix](https://blog.premai.io/llm-observability-setting-up-langfuse-langsmith-helicone-phoenix/)
- [OpenTelemetry GenAI 语义约定](https://opentelemetry.io/docs/specs/semconv/gen-ai/)
- [Arize Phoenix 文档](https://docs.arize.com/phoenix)
- [Helicone 文档](https://docs.helicone.ai/)
