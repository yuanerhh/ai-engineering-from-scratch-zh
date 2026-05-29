# AI 网关——LiteLLM、Portkey、Kong AI Gateway、Bifrost

> 网关位于应用程序和模型提供商之间。核心功能包括提供商路由、故障转移、重试、速率限制、凭证引用、可观测性、护栏。2026 年市场格局：**LiteLLM** 是 MIT 开源，支持 100+ 提供商，兼容 OpenAI，但在约 2000 RPS 时开始崩溃（8GB 内存、已发布基准测试中出现级联故障）；最适合 Python、<500 RPS、开发/原型场景。**Portkey** 定位于控制平面（护栏、PII 脱敏、越狱检测、审计追踪），2026 年 3 月转为 Apache 2.0 开源，延迟开销 20-40ms，$49/月生产版。**Kong AI Gateway** 基于 Kong Gateway 构建——Kong 在同等 12 CPU 上的基准测试：比 Portkey 快 228%，比 LiteLLM 快 859%；$100/模型/月定价（Plus 版最多 5 个模型）；已在 Kong 上的企业客户首选。**Bifrost**（Maxim AI）——可配置退避策略的自动重试，OpenAI 429 时回退到 Anthropic 是典型配方。**Cloudflare / Vercel AI Gateway**——托管式、零运维、基础重试。数据合规要求驱动自托管决策；Portkey 和 Kong 以开源 + 可选托管处于中间位置。

**类型：** 学习
**编程语言：** Python（标准库，玩具网关路由模拟器）
**前置知识：** Phase 17 · 01（托管 LLM 平台）、Phase 17 · 16（模型路由）
**预计时间：** 约 60 分钟

## 学习目标

- 列举网关的六项核心功能（路由、故障转移、重试、速率限制、凭证管理、可观测性、护栏）。
- 将四个 2026 年网关（LiteLLM、Portkey、Kong AI、Bifrost）映射到其规模上限和使用场景。
- 引用 Kong 基准测试（比 Portkey 快 228%，比 LiteLLM 快 859%），并解释其对 >500 RPS 场景的意义。
- 根据数据合规要求和运维预算，选择自托管 vs 托管。

## 问题背景

你的产品调用 OpenAI、Anthropic 和自托管的 Llama。每个提供商有不同的 SDK、错误模型、速率限制和认证方式。你希望有故障转移（如果 OpenAI 返回 429，尝试 Anthropic）、统一的凭证存储、统一可观测性，以及按租户的速率限制。

在应用层重新实现这些功能会将每个服务耦合到每个提供商。网关层将其整合到一个进程中，提供一个 API（通常兼容 OpenAI），向提供商扇出。

## 核心概念

### 六项核心功能

1. **提供商路由** — 将 OpenAI、Anthropic、Gemini、自托管等统一在一个 API 后面。
2. **故障转移** — 在 429、5xx 或质量失败时重试其他提供商。
3. **重试** — 指数退避，有上限的尝试次数。
4. **速率限制** — 按租户、按密钥、按模型。
5. **凭证引用** — 运行时从保险库拉取凭证（永远不写入应用）。
6. **可观测性** — OTel + GenAI 属性（Phase 17 · 13）+ 成本归因。
7. **护栏** — PII 脱敏、越狱检测、允许话题过滤。

### LiteLLM——MIT 开源，Python

- 100+ 提供商，兼容 OpenAI，路由配置，故障转移，基础可观测性。
- Kong 基准测试中约 2000 RPS 时开始崩溃；8GB 内存占用，持续负载下出现级联故障。
- 最适合：Python 应用、<500 RPS、开发/预发布网关、实验性路由。
- 成本：开源版 $0；提供云免费层。

### Portkey——控制平面定位

- 2026 年 3 月起采用 Apache 2.0 开源。护栏、PII 脱敏、越狱检测、审计追踪。
- 每请求延迟开销 20-40ms。
- 生产版含保留策略 + SLA 的 $49/月。
- 最适合：需要捆绑护栏 + 可观测性的监管行业。

### Kong AI Gateway——规模方案

- 基于 Kong Gateway（成熟的 API 网关产品，Lua+OpenResty）构建。
- Kong 在 12 核等效机器上的自测基准：比 Portkey 快 228%，比 LiteLLM 快 859%。
- 定价：$100/模型/月，Plus 版最多 5 个。
- 最适合：已在 Kong 上；>1000 RPS；愿意付费授权。

### Bifrost（Maxim AI）

- 可配置退避的自动重试。
- OpenAI 429 时回退到 Anthropic 是标准配方。
- 较新的入局者；商业产品。

### Cloudflare AI Gateway / Vercel AI Gateway

- 托管式，零运维。基础重试和可观测性。
- 最适合：在 Cloudflare/Vercel 上的边缘服务 JavaScript 应用。
- 护栏和速率限制功能比 Kong/Portkey 有限。

### 自托管 vs 托管

数据合规是决定性因素。医疗和金融默认自托管（LiteLLM 或 Portkey OSS 或 Kong）。消费者产品默认托管（Cloudflare AI Gateway）或中间层（Portkey 托管）。混合：受监管租户自托管，其他租户托管。

### 延迟预算

- LiteLLM：典型 5-15ms 开销。
- Portkey：20-40ms 开销。
- Kong：3-8ms 开销。
- Cloudflare/Vercel：1-3ms 开销（边缘优势）。

网关延迟直接叠加到 TTFT 上。TTFT P99 < 100ms SLA 时，选 Kong 或 Cloudflare。P99 < 500ms 时，任何都可以。

### 速率限制语义很重要

简单令牌桶在中等规模下有效。多租户需要滑动窗口 + 突发允许 + 按租户分级。LiteLLM 提供令牌桶；Kong 提供滑动窗口；Portkey 提供分级。

### 网关 + 可观测性 + 路由的组合

Phase 17 · 13（可观测性）+ 16（模型路由）+ 19（网关）在生产中属于同一层。选择一个覆盖三者的工具，或谨慎地连接：2026 年大多数部署将 Helicone（可观测性）或 Portkey（护栏）与 Kong（规模）结合，各司其职。

### 需要记住的数字

- LiteLLM：约 2000 RPS 时崩溃，8GB 内存。
- Portkey：20-40ms 开销；2026 年 3 月起采用 Apache 2.0。
- Kong：比 Portkey 快 228%，比 LiteLLM 快 859%。
- Kong 定价：$100/模型/月，Plus 版最多 5 个。
- Cloudflare/Vercel：边缘 1-3ms 开销。

## 动手实践

`code/main.py` 在注入 429/5xx 的情况下，模拟跨 3 个提供商的带故障转移的网关路由。报告延迟、重试率和故障转移命中率。

## 产出技能

本课产出 `outputs/skill-gateway-picker.md`。给定规模、运维立场、合规要求、延迟预算，选择网关。

## 练习

1. 运行 `code/main.py`。配置 OpenAI→Anthropic→自托管的故障转移。5% 提供商错误率下预期命中率是多少？
2. 你的 SLA 是 300ms 基线上 TTFT P99 < 200ms。哪些网关在预算内？
3. 医疗客户要求自托管 + PII 脱敏 + 审计。选择 Portkey OSS 还是 Kong。
4. 对比 LiteLLM 和 Kong：团队应该在什么 RPS 上限时迁移？
5. 为多租户 SaaS 设计速率限制策略：免费层、试用层、付费层。令牌桶还是滑动窗口？

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| 网关（Gateway） | "API 经纪人" | 位于应用和提供商之间的进程 |
| LiteLLM | "MIT 那个" | Python OSS，100+ 提供商，2K RPS 时崩溃 |
| Portkey | "护栏网关" | 控制平面 + 可观测性，Apache 2.0 |
| Kong AI Gateway | "规模那个" | 基于 Kong Gateway 构建，基准测试领先 |
| Bifrost | "Maxim 的网关" | 重试 + Anthropic 故障转移配方 |
| Cloudflare AI Gateway | "边缘托管" | 边缘部署托管网关，零运维 |
| PII 脱敏（PII redaction） | "数据清洗" | 发送给模型前的正则 + NER 掩码 |
| 越狱检测（Jailbreak detection） | "提示词注入防护" | 对用户输入的分类器 |
| 审计追踪（Audit trail） | "合规日志" | 每次 LLM 调用的不可篡改记录 |
| 令牌桶（Token-bucket） | "简单速率限制" | 基于补充的速率限制器 |
| 滑动窗口（Sliding-window） | "精确速率限制" | 基于时间窗口的速率限制器；公平性更好 |

## 延伸阅读

- [Kong AI Gateway 基准测试](https://konghq.com/blog/engineering/ai-gateway-benchmark-kong-ai-gateway-portkey-litellm)
- [TrueFoundry——2026 年 AI 网关对比](https://www.truefoundry.com/blog/a-definitive-guide-to-ai-gateways-in-2026-competitive-landscape-comparison)
- [Techsy——2026 年顶级 LLM 网关工具](https://techsy.io/en/blog/best-llm-gateway-tools)
- [LiteLLM GitHub](https://github.com/BerriAI/litellm)
- [Portkey GitHub](https://github.com/Portkey-AI/gateway)
- [Kong AI Gateway 文档](https://docs.konghq.com/gateway/latest/ai-gateway/)
