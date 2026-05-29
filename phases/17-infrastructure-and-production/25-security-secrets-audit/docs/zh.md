# 安全——密钥管理、API 密钥轮换、审计日志、护栏

> 通过集中式保险库（HashiCorp Vault、AWS Secrets Manager、Azure Key Vault）消除密钥蔓延。永远不要将凭证存储在配置文件、VCS 中的 env 文件或电子表格中。使用 IAM 角色而非静态密钥；CI/CD 使用 OIDC。AI 网关模式是 2026 年的解决方案：应用 → 网关 → 模型提供商，网关在运行时从保险库拉取凭证。在保险库中轮换，所有应用几分钟内更新——无需重新部署，无需 Slack "谁有新密钥" 的消息。轮换策略 ≤90 天；每次提交用 TruffleHog / GitGuardian / Gitleaks 扫描。零信任：MFA、SSO、RBAC/ABAC、短期令牌、设备状态检查。PII 清洗使用实体识别，在转发前屏蔽 PHI/PII；一致性令牌化（Mesh 方法）将敏感值映射到稳定的占位符，使 LLM 保留代码/关系语义。网络出口：LLM 服务在专用 VPC/VNet 子网中，仅允许访问 `api.openai.com`、`api.anthropic.com` 等；阻止所有其他出站流量。2026 年事故驱动因素：Vercel 供应链攻击——被攻陷的 CI/CD 凭证从数千个客户部署中外泄了环境变量。

**类型：** 学习
**编程语言：** Python（标准库，玩具 PII 清洗器 + 审计日志写入器）
**前置知识：** Phase 17 · 19（AI 网关）、Phase 17 · 13（可观测性）
**预计时间：** 约 60 分钟

## 学习目标

- 列举四种密钥管理反模式（VCS 中的配置文件、硬编码环境变量、电子表格、静态密钥），并说出各自的替代方案。
- 解释 AI 网关从保险库拉取密钥模式作为 2026 年生产标准。
- 实现带一致性令牌化的 PII 清洗器（相同值 → 相同占位符），使语义得以保留。
- 描述 2026 年 Vercel 供应链事故以及它关于 CI/CD 凭证卫生的教训。

## 问题背景

一名实习生提交了带 API 密钥的 `.env` 文件。他们很快删除了它。密钥已经在 git 历史中——GitGuardian 扫描发现了它，你的轮换流程是"在 Slack 通知团队，更新 40 个配置文件，重新部署所有服务"。8 小时后，一半服务已上线，另一半在等待部署窗口。

另外，用户提示词包含"我的社会安全号码是 123-45-6789"。提示词被发送到 OpenAI。你有 BAA，但你的内部策略是在转发前屏蔽 PII。你没有这样做。

另外，你的 EKS 集群的 LLM Pod 可以访问任何互联网主机。有人通过 DNS 查询向攻击者控制的域名外泄数据。没有任何东西阻止了它。

LLM 服务的安全必须解决所有三个攻击向量：保险库支持的凭证、PII 清洗、网络出口过滤、审计日志。

## 核心概念

### 集中式保险库 + IAM 角色拉取

**保险库**：HashiCorp Vault、AWS Secrets Manager、Azure Key Vault、GCP Secret Manager。单一可信来源。

**IAM 角色**：应用/网关通过 IAM 身份进行认证，而非使用静态密钥。保险库返回令牌生命周期内的密钥。

**AI 网关模式**：网关在请求时从保险库拉取 `OPENAI_API_KEY`。在保险库中轮换；下一个请求获取新密钥。无需重新部署。

### 轮换策略 ≤90 天

所有 API 密钥、保险库根令牌、CI/CD 凭证。尽可能自动轮换。手动轮换需记录和追踪。

### 密钥扫描

- **TruffleHog** — 对提交进行正则 + 熵检测。
- **GitGuardian** — 商业，高准确率。
- **Gitleaks** — OSS，在 CI 中运行。

每次提交都运行。如果检测到新密钥，阻止 PR。

### 零信任态势

- 所有账户强制 MFA。
- 通过 SAML/OIDC 实现 SSO。
- RBAC（基于角色）或 ABAC（基于属性）实现细粒度访问控制。
- 短期令牌（小时级，不是天级）。
- 设备状态检查——只有带磁盘加密的企业设备。

### PII / PHI 清洗

在提示词离开你的基础设施之前：

1. 实体识别（spaCy NER、Presidio、商业方案）。
2. 屏蔽匹配实体：`"我的社会安全号码是 123-45-6789"` → `"我的社会安全号码是 [SSN_TOKEN_A3F]"`。
3. 一致性令牌化（Mesh 方法）：相同值映射到相同占位符，使 LLM 保留关系。
4. 可选的对 LLM 响应进行反向映射。

静态正则过滤器捕获基本模式；NER 捕获更多。两者都用。

### 输入 + 输出护栏

输入：阻止已知越狱攻击、禁止话题；按用户速率限制。

输出：正则清洗泄露的密钥（API 密钥模式、拒绝上下文中的邮件模式），策略违规分类器。

### 网络出口白名单

LLM 服务在专用子网中：
- 白名单：`api.openai.com`、`api.anthropic.com`、向量数据库端点、保险库端点。
- 其余：全部丢弃。
- 通过仅允许列表解析器进行 DNS（避免 DNS 隧道外泄）。

### 审计日志

每次 LLM 调用的不可变日志，包含：
- 时间戳。
- 用户/租户。
- 提示词哈希（不是原始提示词，保护隐私）。
- 模型 + 版本。
- Token 计数。
- 成本。
- 响应哈希。
- 任何护栏触发情况。

按监管要求保留（SOC 2 1 年，HIPAA 6 年）。

### 2026 年 Vercel 事故

供应链攻击：被攻陷的 CI/CD 凭证从数千个客户部署中外泄了环境变量。教训：CI/CD 凭证等同于生产凭证。存储在保险库中。窄范围授权。积极轮换。

### 需要记住的数字

- 轮换策略：≤90 天。
- 每次提交扫描：TruffleHog / GitGuardian / Gitleaks。
- Vercel 2026：CI/CD 凭证被攻陷 → 数千个客户环境变量泄露。
- 审计日志保留：SOC 2 = 1 年，HIPAA = 6 年。

## 动手实践

`code/main.py` 实现带一致性令牌化的玩具 PII 清洗器和仅追加审计日志。

## 产出技能

本课产出 `outputs/skill-llm-security-plan.md`。给定监管范围和当前状态，规划保险库迁移、清洗器、出口控制、审计日志。

## 练习

1. 运行 `code/main.py`。发送两个引用相同社会安全号码的提示词。确认两者获得相同的占位符。
2. 为调用 OpenAI + Anthropic + Weaviate 的 vLLM-on-EKS 部署设计网络出口策略。
3. 你发现 git 历史中有一个 2 年前的密钥。正确响应是什么——轮换密钥、清洗历史，还是两者？说明理由。
4. 你的审计日志每天增长 10GB。设计保留层（热层 30 天、温层 12 个月、冷层 6 年）。
5. 论证反向令牌化（将真实值替换回 LLM 响应中）是否值得增加复杂性，而不是保持占位符可见。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| 保险库（Vault） | "密钥存储" | 集中式凭证管理服务 |
| IAM 角色（IAM role） | "基于身份的认证" | 应用承担的角色；返回短期凭证 |
| CI/CD 的 OIDC | "云端签发的令牌" | CI 中无静态密钥——通过 OIDC 身份 |
| TruffleHog / GitGuardian / Gitleaks | "密钥扫描器" | 提交时的密钥检测 |
| RBAC / ABAC | "访问控制" | 基于角色 vs 基于属性 |
| PII 清洗（PII scrubbing） | "数据脱敏" | 删除或令牌化敏感实体 |
| 一致性令牌化（Consistent tokenization） | "稳定占位符" | 相同值 → 每次相同令牌 |
| Mesh 方法（Mesh approach） | "Mesh 令牌化" | 语义保留的令牌化模式 |
| 出口白名单（Egress whitelist） | "出站允许列表" | 只有允许的域名可以访问 |
| 审计日志（Audit log） | "不可变历史" | 用于合规的仅追加记录 |

## 延伸阅读

- [Doppler——高级 LLM 安全](https://www.doppler.com/blog/advanced-llm-security)
- [Portkey——用密钥引用管理 LLM API 密钥](https://portkey.ai/blog/secret-references-ai-api-key-management/)
- [Datadog——LLM 护栏最佳实践](https://www.datadoghq.com/blog/llm-guardrails-best-practices/)
- [JumpServer——2026 年密钥管理最佳实践](https://www.jumpserver.com/blog/secret-management-best-practices-2026)
- [Microsoft Presidio](https://github.com/microsoft/presidio) — PII 检测与匿名化
- [HashiCorp Vault 文档](https://developer.hashicorp.com/vault/docs)
