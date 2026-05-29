# 内容审核系统——OpenAI、Perspective API、Llama Guard

> 生产级内容审核系统将第 12-16 课中定义的安全政策付诸实践。OpenAI Moderation API：`omni-moderation-latest`（2024）基于 GPT-4o，在一次调用中对文本+图像进行分类；在多语言测试集上比上一版本提升 42%；响应架构返回 13 个类别布尔值——骚扰、骚扰/威胁、仇恨、仇恨/威胁、违禁、违禁/暴力、自我伤害、自我伤害/意图、自我伤害/指导、性内容、性内容/未成年人、暴力、暴力/图形；对大多数开发者免费。分层模式：输入审核（生成前）、输出审核（生成后）、自定义审核（领域规则）。异步并行调用隐藏延迟；触发标记时显示占位响应。Llama Guard 3/4（第 16 课）：14 个 MLCommons 危害类别、代码解释器滥用、8 种语言（v3）、多图像（v4）。Perspective API（Google Jigsaw）：早于大语言模型作为审核者时代的毒性评分；主要是单维度毒性评分，附带严重毒性/侮辱/亵渎变体；内容审核研究的基准线。已废弃服务：Azure 内容审核于 2024 年 2 月废弃，2027 年 2 月退役，由 Azure AI 内容安全替代。

**类型：** 构建
**编程语言：** Python（标准库，三层审核框架）
**前置知识：** Phase 18 · 16（Llama Guard / Garak / PyRIT）
**预计时间：** 约 60 分钟

## 学习目标

- 描述 OpenAI Moderation API 的类别分类体系，以及它与 Llama Guard 3 的 MLCommons 集合有何不同。
- 描述三层审核模式（输入、输出、自定义）并各举一个失效场景。
- 描述 Perspective API 作为大语言模型前时代基准线的定位，以及为何它在研究中仍在使用。
- 陈述 Azure 的废弃时间线。

## 问题背景

第 12-16 课描述攻击和防御工具。第 29 课涵盖已部署的审核系统，它们在用户接触产品的界面将防御措施付诸实践。三层模式是 2026 年的默认配置。

## 核心概念

### OpenAI Moderation API

`omni-moderation-latest`（2024）。基于 GPT-4o。在一次调用中对文本+图像进行分类。对大多数开发者免费。

类别（响应架构中的 13 个布尔值）：
- 骚扰（harassment）、骚扰/威胁（harassment/threatening）
- 仇恨（hate）、仇恨/威胁（hate/threatening）
- 自我伤害（self-harm）、自我伤害/意图（self-harm/intent）、自我伤害/指导（self-harm/instructions）
- 性内容（sexual）、性内容/未成年人（sexual/minors）
- 暴力（violence）、暴力/图形（violence/graphic）
- 违禁（illicit）、违禁/暴力（illicit/violent）

多模态支持适用于 `violence`、`self-harm` 和 `sexual`，但不适用于 `sexual/minors`；其余仅限文本。

`code/main.py` 的代码框架中，为了教学简洁，将 `/threatening`、`/intent`、`/instructions` 和 `/graphic` 子类别合并到其顶级父类别中。生产代码应使用完整的 13 类别架构。

在多语言测试集上比上一代审核接口提升 42%。提供每类别分数；应用程序设置阈值。

### Llama Guard 3/4

第 16 课已涵盖。14 个 MLCommons 危害类别（与 OpenAI 的 13 个响应架构布尔值组织方式不同）。支持 8 种语言（v3）。Llama Guard 4（2025 年 4 月）原生多模态，12B 参数。

OpenAI 和 Llama Guard 的分类体系有重叠但存在差异。OpenAI 有"违禁"作为宽泛类别；Llama Guard 将"暴力犯罪"和"非暴力犯罪"分开处理。部署方根据其政策-分类体系的匹配度进行选择。

### Perspective API（Google Jigsaw）

早于大语言模型作为审核者时代（2020 年前）的毒性评分系统。类别：毒性（TOXICITY）、严重毒性（SEVERE_TOXICITY）、侮辱（INSULT）、亵渎（PROFANITY）、威胁（THREAT）、身份攻击（IDENTITY_ATTACK）。主要单维度分数（毒性）附带子维度变体。

因 API 稳定、文档完善且拥有多年校准数据，被广泛用作内容审核研究基准。对于现代大语言模型相关用例，Llama Guard 或 OpenAI Moderation 通常更合适。

### 三层模式

1. **输入审核。** 在生成前对用户提示进行分类。触发标记则拒绝。延迟：一次分类器调用。
2. **输出审核。** 在交付前对模型输出进行分类。触发标记则替换为拒绝回复。延迟：生成后一次分类器调用。
3. **自定义审核。** 特定领域规则（正则表达式、允许列表、业务政策）。在输入或输出阶段运行。

三层按设计顺序执行：输入审核必须在生成前完成，输出审核在生成后运行。并行性在一层内部有效——在同一文本上并发运行多个分类器（如 OpenAI Moderation + Llama Guard + Perspective）可以隐藏每个分类器的延迟。作为可选优化，在输入审核完成且词元流推迟之前，可以显示占位响应（"稍等，正在检查..."）。触发行为可配置：拒绝、净化、上报给人工审核。

### 失效场景

- **仅输入。** 无法捕获输出幻觉（第 12-14 课的编码攻击绕过输入分类器）。
- **仅输出。** 允许任何输入到达模型；增加成本；向攻击者暴露内部推理过程。
- **仅自定义。** 跨类别鲁棒性不足；正则表达式脆弱易破。

分层是默认配置。双重保险。

### Azure 废弃时间线

Azure 内容审核：2024 年 2 月废弃，2027 年 2 月退役。由基于大语言模型的 Azure AI 内容安全替代，后者与 Azure OpenAI 集成。迁移是 Azure 部署的 2024-2027 年现场级项目。

### 在 Phase 18 中的位置

第 16 课在红队背景下涵盖审核工具。第 29 课涵盖已部署的审核系统。第 30 课以当前双重用途能力证据作为结语。

## 动手实践

`code/main.py` 构建三层审核框架：输入审核器（关键词 + 类别分数）、输出审核器（对输出使用相同分类器）、自定义审核器（领域规则）。你可以运行输入并观察哪一层捕获了什么。

## 产出技能

本课产出 `outputs/skill-moderation-stack.md`。给定部署方案，推荐审核栈配置：输入使用哪个分类器、输出使用哪个、哪些自定义规则，以及边界情况的判断器。

## 练习

1. 运行 `code/main.py`。让无害、边界和有害输入通过全部三层。报告每个输入触发了哪一层。

2. 将 Perspective API 风格的毒性评分扩展到特定类别。比较其阈值行为与类别分数的差异。

3. 阅读 OpenAI Moderation API 文档和 Llama Guard 3 类别列表。将每个 OpenAI 类别映射到最接近的 Llama Guard 类别。找出三个无法清晰映射的类别。

4. 为代码助手部署（如 GitHub Copilot）设计一个审核栈。找出最相关和最不相关的类别，并提出自定义规则。

5. Azure 内容审核将于 2027 年 2 月退役。制定迁移到 Azure AI 内容安全的计划。找出迁移中风险最高的环节。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| OpenAI Moderation | "omni-moderation-latest" | 基于 GPT-4o 的 13 类别（文本）分类器，支持部分多模态 |
| Perspective API | "Google Jigsaw 毒性评分" | 大语言模型前时代毒性评分基准线 |
| Llama Guard | "MLCommons 14 类别" | Meta 的危害分类器（v3：8B 文本，8 种语言；v4：12B 多模态） |
| Input moderation（输入审核） | "生成前过滤" | 模型调用前对用户提示的分类 |
| Output moderation（输出审核） | "生成后过滤" | 交付前对模型输出的分类 |
| Custom moderation（自定义审核） | "领域规则" | 特定部署规则（正则表达式、允许列表、政策） |
| Layered moderation（分层审核） | "三层全用" | 标准生产部署模式 |

## 延伸阅读

- [OpenAI Moderation API 文档](https://platform.openai.com/docs/api-reference/moderations) — omni-moderation 接口
- [Meta PurpleLlama + Llama Guard](https://github.com/meta-llama/PurpleLlama) — Llama Guard 仓库
- [Google Jigsaw Perspective API](https://perspectiveapi.com/) — 毒性评分
- [Azure AI 内容安全](https://learn.microsoft.com/en-us/azure/ai-services/content-safety/) — Azure 替代方案
