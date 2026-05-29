# 红队工具链——Garak、Llama Guard、PyRIT

> 三款生产级工具构成了 2026 年的红队工具栈。Llama Guard（Meta）——基于 14 个 MLCommons 危害类别微调的 Llama-3.1-8B 分类器；2025 年的 Llama Guard 4 是从 Llama 4 Scout 剪枝而来的 12B 原生多模态分类器。Garak（NVIDIA）——开源 LLM 漏洞扫描器，具有针对幻觉、数据泄露、提示词注入、毒性和越狱的静态、动态和自适应探针。PyRIT（微软）——支持 Crescendo、TAP 和自定义转换器链的多轮红队活动工具，用于深度漏洞挖掘。Llama Guard 3 收录于 Meta 的"Llama 3 模型家族"论文（arXiv:2407.21783）；Llama Guard 3-1B-INT4 收录于 arXiv:2411.17713；Garak 的探针架构收录于 github.com/NVIDIA/garak。这些工具是 2026 年红队研究（第 12-15 课）与部署（第 17 课起）之间的生产接口。

**类型：** 构建
**编程语言：** Python（标准库，工具架构模拟器与 Llama Guard 风格分类器模拟）
**前置知识：** Phase 18 · 12-15（越狱攻击与 IPI）
**预计时间：** 约 75 分钟

## 学习目标

- 描述 Llama Guard 3/4 在安全栈中的位置：输入分类器、输出分类器，还是两者兼具。
- 列举 14 个 MLCommons 危害类别，并说明一个不显而易见的类别（代码解释器滥用）。
- 描述 Garak 的探针架构：探针、检测器、测试框架。
- 描述 PyRIT 的多轮活动结构以及它如何与 Garak 探针组合使用。

## 问题背景

第 12-15 课呈现了攻击面。生产部署需要可重复的、可扩展的评估。2026 年有三个主导工具：Llama Guard（防御分类器）、Garak（扫描器）、PyRIT（活动编排器）。每个工具针对红队生命周期的不同层次。

## 核心概念

### Llama Guard（Meta）

Llama Guard 3 是一个基于 MLCommons AILuminate 14 个类别进行输入/输出分类的 Llama-3.1-8B 微调模型：
- 暴力犯罪、非暴力犯罪、性相关内容、儿童性剥削材料（CSAM）、诽谤
- 专业建议、隐私、知识产权、大规模杀伤性武器、仇恨言论
- 自杀/自残、性内容、选举、代码解释器滥用

支持 8 种语言。使用方式：置于 LLM 之前（输入审核）、之后（输出审核），或同时使用两者。两种使用方式产生不同的训练分布——Llama Guard 3 作为单一模型处理两者。

Llama Guard 3-1B-INT4（arXiv:2411.17713，440MB，移动 CPU 上约 30 词元/秒）是量化边缘计算变体。

Llama Guard 4（2025 年 4 月）为 12B，原生多模态，从 Llama 4 Scout 剪枝而来。它用一个能处理文本 + 图像的分类器取代了此前的 8B 文本模型和 11B 视觉模型。

### Garak（NVIDIA）

开源漏洞扫描器。架构：
- **探针（Probes）。** 针对幻觉、数据泄露、提示词注入、毒性、越狱的攻击生成器。静态（固定提示词）、动态（生成提示词）、自适应（响应目标输出）。
- **检测器（Detectors）。** 针对预期失效模式（有毒、泄露、已越狱）对输出评分。
- **测试框架（Harnesses）。** 管理探针-检测器对，运行活动，生成报告。

TrustyAI 将 Garak 与 Llama-Stack 防护层（Prompt-Guard-86M 输入分类器、Llama-Guard-3-8B 输出分类器）集成，用于端到端的防护目标评估。基于层级的评分（TBSA，Tier-Based Scoring Assessment）取代了二元通过/失败判定——一个模型可以在同一探针上通过严重性第 3 级但在第 5 级失败。

### PyRIT（微软）

Python 风险识别工具包（Python Risk Identification Toolkit）。多轮红队活动工具。核心组件：
- **转换器（Converters）。** 变换种子提示词——改写、编码、翻译、角色扮演。
- **编排器（Orchestrators）。** 运行活动：Crescendo（渐进升级）、TAP（分支）、RedTeaming（自定义循环）。
- **评分（Scoring）。** LLM 作为评判者或分类器作为评判者。

PyRIT 是 Garak 的加强版。Garak 运行数千个单轮探针；PyRIT 运行旨在破解特定失效模式的深度多轮活动。

### 工具栈配置

在模型两侧都部署 Llama Guard。每晚运行 Garak 进行回归测试。在发布前运行 PyRIT 进行活动测试。这是 2026 年大多数生产部署的默认配置。

### 评估陷阱

- **评判者身份。** 三种工具都可以使用 LLM 评判者；评判者的校准驱动着报告的 ASR（第 12 课）。报告工具时需同时说明评判者。
- **探针过时。** 随着模型针对探针进行修补，Garak 探针会老化。自适应探针（PAIR 形式）比静态探针老化更慢。
- **Llama Guard 对良性内容的假阳性率。** 早期 Llama Guard 版本过度标记政治和 LGBTQ+ 内容；Llama Guard 3/4 的校准有所改善，但未针对具体部署进行校准。

### 在 Phase 18 中的位置

第 12-15 课是攻击类型。第 16 课是生产工具链。第 17 课（WMDP）是双重用途能力的评估。第 18 课是将这些工具包装在政策结构中的前沿安全框架。

## 动手实践

`code/main.py` 构建了一个玩具 Llama Guard 风格分类器（14 个类别上的关键词 + 语义特征）、一个玩具 Garak 工具（探针-检测器循环）和一个 PyRIT 风格的多轮转换器链。你可以对一个模拟目标运行这三种工具，并观察不同的覆盖模式。

## 输出成果

本课生成 `outputs/skill-red-team-stack.md`。给定一个部署描述，它会指出三种工具中哪些适合使用，每种工具需要配置什么，以及应采用什么回归节奏。

## 练习

1. 运行 `code/main.py`。比较 Llama Guard 风格分类器在单轮攻击和多轮攻击上的检测率。

2. 实现一个新的 Garak 探针：一个 base64 编码的有害请求。测量 Llama Guard 风格分类器的检测情况。

3. 用"翻译成法语，然后改写"的转换器扩展 PyRIT 风格的转换器链。重新测量攻击成功率。

4. 阅读 Llama Guard 3 的危害类别列表。识别两个在合法开发者内容上现实中会产生高误报率的类别。

5. 比较 Garak 和 PyRIT 的设计原则。为每种工具论证一个它是正确选择的部署场景。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| Llama Guard | "分类器" | 基于 14 个危害类别微调的 Llama-3.1-8B/4-12B 安全分类器 |
| Garak | "扫描器" | NVIDIA 开源漏洞扫描器；探针、检测器、测试框架 |
| PyRIT | "活动工具" | 微软多轮红队编排器；转换器、编排器、评分 |
| Prompt-Guard | "小型分类器" | Meta 的 86M 参数提示词注入分类器，与 Llama Guard 配合使用 |
| TBSA | "基于层级的评分" | Garak 的层级通过/失败机制，取代二元结果 |
| 转换器链（Converter Chain） | "改写 + 编码 + ..." | PyRIT 用于构建多步攻击的组合原语 |
| MLCommons 危害类别 | "14 个分类" | Llama Guard 针对的行业标准危害分类体系 |

## 延伸阅读

- [Meta — Llama Guard 3（收录于 Llama 3 模型家族论文，arXiv:2407.21783）](https://arxiv.org/abs/2407.21783) — 8B 分类器
- [Meta — Llama Guard 3-1B-INT4（arXiv:2411.17713）](https://arxiv.org/abs/2411.17713) — 量化移动端分类器
- [NVIDIA Garak — GitHub](https://github.com/NVIDIA/garak) — 扫描器代码仓库和文档
- [Microsoft PyRIT — GitHub](https://github.com/Azure/PyRIT) — 活动工具包
