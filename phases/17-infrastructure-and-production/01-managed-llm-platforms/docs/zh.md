# 托管 LLM 平台——Bedrock、Vertex AI、Azure OpenAI

> 三家超大规模云厂商，三种截然不同的策略。AWS Bedrock 是模型市场——Claude、Llama、Titan、Stability、Cohere 共享一个 API。Azure OpenAI 是 OpenAI 独家合作关系，加上用于专用容量的预置吞吐量单元（PTU）。Vertex AI 以 Gemini 为核心，提供最佳的长上下文和多模态能力。2026 年，Artificial Analysis 测量到 Azure OpenAI 在 Llama 3.1 405B 等效部署上的中位首 token 延迟约为 50ms，Bedrock 约为 75ms——PTU 解释了这一差距，因为专用容量胜过共享按需服务。决策规则不是"哪个最快"，而是"哪个模型目录和 FinOps 界面与我的产品相匹配"。本课教你在权衡利弊有据可查的情况下做出选择，而非凭直觉。

**类型：** 学习
**编程语言：** Python（标准库，玩具成本与延迟比较器）
**前置知识：** Phase 11（LLM 工程）、Phase 13（工具与协议）
**预计时间：** 约 60 分钟

## 学习目标

- 命名三种平台策略（市场式 vs 独家合作 vs Gemini 优先），并将每种策略与产品用例对应。
- 解释 Azure OpenAI 中预置吞吐量单元（PTU）能买到什么，以及为什么按需 Bedrock 在 405B 规模上通常慢约 25ms。
- 绘制每个平台的 FinOps 归因界面图（Bedrock 应用推理配置文件 vs Vertex 每团队项目 vs Azure 范围 + PTU 预留）。
- 写下"双提供商最低要求"策略，并解释为什么单一供应商锁定是 2026 年代价最高的错误。

## 问题背景

你为产品选择了 Claude 3.7 Sonnet，现在需要部署它。你可以直接调用 Anthropic API，也可以通过 AWS Bedrock 调用，或者通过网关调用。直接调用 API 最简单；Bedrock 增加了 BAA、VPC 端点、IAM 和 CloudWatch 归因。网关增加了故障转移、统一计费以及跨提供商的速率限制。

更深层的问题是目录。如果你的产品同时需要 Claude、Llama 和 Gemini，除非同时使用 Bedrock + Vertex + Azure OpenAI，否则无法从一个地方全部获得。三家超大规模云厂商并非可互换——它们各自对谁拥有模型层押下了不同的赌注。

本课梳理三种赌注、延迟差距、FinOps 差距和锁定风险。

## 核心概念

### 三种策略

**AWS Bedrock** — 模型市场。Claude（Anthropic）、Llama（Meta）、Titan（AWS 自有）、Stability（图像）、Cohere（嵌入）、Mistral，加上图像和嵌入子目录。一个 API，一个 IAM 界面，一个 CloudWatch 导出。Bedrock 的赌注是：客户对可选性的需求大于对单一模型的需求。

**Azure OpenAI** — 独家合作关系。你可以在 Azure 数据中心使用 GPT-4/4o/5/o 系列、DALL·E、Whisper 以及 OpenAI 模型的微调。"Azure OpenAI Service"目录中没有非 OpenAI 模型——那些归属于 Azure AI Foundry（独立产品）。Azure 的赌注是：OpenAI 仍然是前沿，客户希望在这一特定关系上有企业级管控。

**Vertex AI** — Gemini 优先，其他次之。Gemini 1.5/2.0/2.5 Flash 和 Pro，加上 Model Garden（第三方）。Vertex 的赌注是多模态长上下文——100 万 token 的 Gemini 上下文窗口是其差异化所在。

### 规模延迟差距

Artificial Analysis 持续运行基准测试。在等效 Llama 3.1 405B 部署（共享按需）上，Azure OpenAI 中位首 token 延迟约为 50ms；Bedrock 约为 75ms。这一差距不是 AWS 的失败——而是容量模型的差异。Azure 销售 PTU（预置吞吐量单元），为你的租户预留 GPU 容量。Bedrock 等效方案（预置吞吐量）存在，但每单元起价约 21 美元/小时，大多数客户仍使用共享按需服务。

共享按需容量与所有其他客户的流量竞争。专用容量则不然。如果你的产品 SLA 要求 P99 首 token 延迟 < 100ms，你要么在 Azure 上购买 PTU，要么购买 Bedrock 预置吞吐量，要么接受默认的延迟抖动。

### 预置吞吐量经济学

Azure PTU：预留的推理计算块。对于可预测工作负载，最多可节省约 70% 的按需费用。无论流量如何，每小时固定收费——即使空闲也要付费。盈亏平衡点通常在 40-60% 的持续利用率附近。

Bedrock 预置吞吐量：根据模型和地区，每小时 21-50 美元。数学类似——盈亏平衡点在峰值利用率约一半处。需要按月承诺。

Vertex 预置容量按 Gemini SKU 销售；定价因模型和地区而异，公开宣传较少。

### FinOps 界面——真正的差异化因素

**Bedrock 应用推理配置文件**是市场中最清晰的归因方式。用 `team`、`product`、`feature` 标签标记配置文件；将所有模型调用路由经过它；CloudWatch 无需后处理即可按配置文件分解成本。2025 年新增，仍是最精细的超大规模云原生方案。

**Vertex** 归因采用每团队项目加全局标签的方式。你将每个团队建模为一个 GCP 项目，在每个资源上打标签，并使用 BigQuery 账单导出 + DataStudio 进行汇总。工作量更大，但 BigQuery 允许对成本数据执行任意 SQL 查询。

**Azure** 依赖订阅/资源组范围加标签，PTU 预留作为一等成本对象。标签从资源组继承，而非从请求继承，因此按请求归因需要 Application Insights 自定义指标或在网关中添加请求头标记。

规律：Bedrock 原生最清晰，Vertex 通过 BigQuery 最灵活，Azure 除非做好仪表化否则最不透明。

### 锁定是 2026 年的风险

当一个模型占主导地位时，绑定单一超大规模云厂商是合理的。2026 年前沿每月都在演进——某个季度是 Claude 3.7，下个季度是 Gemini 2.5，再下个季度是 GPT-5。锁定一个平台就会被排除在三分之二的前沿之外。

工作团队采用的模式：任何产品关键 LLM 调用的双提供商最低要求。Bedrock 加 Azure OpenAI 是常见组合——一个用 Claude，另一个用 GPT，互为故障转移，前置同一个网关。成本增幅可忽略不计，因为网关会路由到最优方案；而在中断期间（如 Azure OpenAI 2025 年 1 月事故、AWS us-east-1 故障）的可用性提升则是决定性的。

### 数据驻留、BAA 和受监管行业

Bedrock：大多数地区提供 BAA；支持 VPC 端点；内置护栏。金融科技的常用默认选择。
Azure OpenAI：HIPAA、SOC 2、ISO 27001；欧盟数据驻留；企业受监管行业的默认选择。
Vertex：HIPAA、GDPR、按地区数据驻留；Google Cloud 的合规堆栈。

三者均满足基本要求。区别在于数据保留策略、日志处理方式，以及滥用监控是否读取你的流量（大多数默认开启；企业版可关闭）。

### 需要记住的数字

- Azure OpenAI 在 Llama 3.1 405B 等效部署上的中位首 token 延迟：约 50ms（使用 PTU）。
- Bedrock 按需中位首 token 延迟：约 75ms。
- Bedrock 预置吞吐量：每单元每小时 21-50 美元。
- Azure PTU 盈亏平衡点：约 40-60% 持续利用率。
- 高利用率下 PTU 相比按需节省：最高 70%。

## 动手实践

`code/main.py` 在合成工作负载上比较三个平台——对按需 vs PTU 经济学、首 token 延迟抖动和成本归因精度建模。运行它，看看 PTU 在哪里划算，以及市场的模型广度在哪里胜过延迟差距。

## 产出技能

本课产出 `outputs/skill-managed-platform-picker.md`。给定工作负载画像（所需模型、首 token 延迟 SLA、日均量、合规要求），推荐主要平台、备用方案和 FinOps 仪表化计划。

## 练习

1. 运行 `code/main.py`。对于 70B 级模型，在多大持续利用率下 Azure PTU 优于按需？计算盈亏平衡点，并与宣传的 40-60% 区间对比。
2. 你的产品同时需要 Claude 3.7 Sonnet 和 GPT-4o。设计双提供商部署——哪个服务哪个超大规模云厂商，前置什么网关，故障转移策略是什么？
3. 受监管的医疗客户要求 BAA、美国东部数据驻留和 P99 首 token 延迟 < 100ms。选择一个平台并用三个具体特性说明理由。
4. 你发现本月 Bedrock 账单在流量不变的情况下增加了 4 倍。没有应用推理配置文件时如何找到原因？有了配置文件需要多长时间？
5. 阅读 Azure OpenAI 和 Bedrock 定价页面。对于每月 1 亿 token 的 Claude 工作负载，哪个更便宜——直接调用 Anthropic API、Bedrock 按需，还是 Bedrock 预置吞吐量？

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| Bedrock | "AWS LLM 服务" | 覆盖 Claude、Llama、Titan、Mistral、Cohere 的模型市场 |
| Azure OpenAI | "Azure 的 ChatGPT" | Azure 数据中心中具有企业管控的独家 OpenAI 模型 |
| Vertex AI | "Google 的 LLM" | Gemini 优先平台，Model Garden 提供第三方模型 |
| PTU | "专用容量" | 预置吞吐量单元——按小时计费的预留推理 GPU |
| 应用推理配置文件（Application Inference Profile） | "Bedrock 标签" | 带标签的按产品成本/用量配置文件，原生于 CloudWatch |
| Model Garden | "Vertex 目录" | Vertex AI 的第三方模型区，独立于 Gemini |
| 双提供商最低要求（Two-provider minimum） | "LLM 冗余" | 每条关键 LLM 路径跨 ≥2 家超大规模云厂商运行的策略 |
| BAA | "HIPAA 文件" | 商业伙伴协议；处理 PHI 的必要条件；三家均提供 |
| 滥用监控（Abuse monitoring） | "日志监视者" | 提供商侧的提示/输出安全扫描；企业版可关闭 |

## 延伸阅读

- [AWS Bedrock 定价](https://aws.amazon.com/bedrock/pricing/) — 权威费率卡和预置吞吐量定价
- [Azure OpenAI 服务定价](https://azure.microsoft.com/en-us/pricing/details/cognitive-services/openai-service/) — PTU 经济学和费率卡
- [Vertex AI 生成式 AI 定价](https://cloud.google.com/vertex-ai/generative-ai/pricing) — Gemini 层级和 Model Garden 附加费
- [Artificial Analysis LLM 排行榜](https://artificialanalysis.ai/) — 跨提供商的持续延迟和吞吐量基准测试
- [The AI Journal — AWS Bedrock vs Azure OpenAI CTO 指南 2026](https://theaijournal.co/2026/03/aws-bedrock-vs-azure-openai/) — 企业决策框架
- [Finout — Bedrock vs Vertex vs Azure FinOps](https://www.finout.io/blog/bedrock-vs.-vertex-vs.-azure-cognitive-a-finops-comparison-for-ai-spend) — 归因机制并排对比
