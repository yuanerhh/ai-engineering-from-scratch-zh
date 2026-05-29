# 对齐研究生态系统——MATS、Redwood、Apollo、METR

> 五个组织定义了 2026 年非实验室对齐研究层。MATS（ML 对齐与理论学者项目）：自 2021 年底以来 527+ 名研究人员，180+ 篇论文，1 万+ 引用，h 指数 47；2024 年夏季队列作为 501(c)(3) 注册，约 90 名学者和 40 名导师；80% 的 2025 年前校友从事安全/安保工作，200+ 人在 Anthropic、DeepMind、OpenAI、英国 AISI、RAND、Redwood、METR、Apollo 任职。Redwood Research：由 Buck Shlegeris 创立的应用对齐实验室；引入了 AI 控制（第 10 课）；与英国 AISI 合作开展控制安全案例研究。Apollo Research：针对前沿实验室的预部署谋算评估；撰写了《上下文谋算》（第 8 课）和《迈向 AI 谋算的安全案例》。METR（模型评估与威胁研究）：基于任务的能力评估，自主任务时间跨度研究；《前沿 AI 安全政策的共同要素》比较了各实验室框架。Eleos AI Research：模型福利预部署评估（第 19 课）；开展了 Claude Opus 4 福利评估。

**类型：** 学习
**编程语言：** 无
**前置知识：** Phase 18 · 01-27（Phase 18 前期课程）
**预计时间：** 约 45 分钟

## 学习目标

- 找出非实验室对齐研究生态系统的五个组织及其核心产出。
- 描述 MATS 的规模（学者、论文、h 指数）及其作为人才输送管道的作用。
- 描述 Redwood 的 AI 控制议程及其与英国 AISI 的合作关系。
- 描述 METR 的基于任务的评估方法论。

## 问题背景

前沿实验室（第 18 课）在内部产出安全评估并选择性地发布结果。实验室之外的生态系统是评估结果被验证、新失效模式被首次发现、以及人才被培养的地方。理解这个生态系统有助于判断哪些研究发现被谁信任。

## 核心概念

### MATS（ML 对齐与理论学者项目）

始于 2021 年底。研究导师制项目；学者用 10-12 周与资深研究人员一起研究特定对齐问题。

规模（2026 年）：
- 自成立以来 527+ 名研究人员。
- 发表 180+ 篇论文。
- 1 万+ 引用。
- h 指数 47。
- 2024 年夏季：90 名学者 + 40 名导师；注册为 501(c)(3)。

职业成果：约 80% 的 2025 年前校友从事安全/安保工作。200+ 人在 Anthropic、DeepMind、OpenAI、英国 AISI、RAND、Redwood、METR、Apollo 任职。

### Redwood Research

应用对齐实验室。由 Buck Shlegeris 创立。引入了 AI 控制议程（第 10 课）。与英国 AISI 合作开展控制安全案例研究。向 DeepMind 和 Anthropic 提供评估设计咨询。

代表性论文：Greenblatt、Shlegeris 等人，《AI 控制》（arXiv:2312.06942，ICML 2024）；《对齐伪装》（Greenblatt、Denison、Wright 等人，arXiv:2412.14093，与 Anthropic 联合发表）。

风格：具体威胁模型、最坏情况对手、可以压力测试的具体协议。

### Apollo Research

针对前沿实验室的预部署谋算评估。撰写了《上下文谋算》（第 8 课，arXiv:2412.04984）。与 2025 年 OpenAI 反谋算训练合作的合作伙伴。产出《迈向 AI 谋算的安全案例》（2024）。

风格：智能体场景评估（欺骗行为可能涌现的场景）；三支柱分解（不对齐、目标导向性、情境意识）。

### METR（模型评估与威胁研究）

基于任务的能力评估。自主任务完成时间跨度研究。《前沿 AI 安全政策的共同要素》（metr.org/common-elements，2025）比较了各实验室框架。

与 Apollo 共同撰写 AI 谋算安全案例草图。

风格：长时间跨度任务评估、实证能力测量、框架综合。

### Eleos AI Research

模型福利预部署评估。开展了 Claude Opus 4 福利评估，记录于系统卡第 5.3 节。为第 19 课福利相关声明提供外部方法论核查。

### 流程

MATS 培训研究人员。毕业生前往 Anthropic、DeepMind、OpenAI（实验室安全团队）或 Redwood、Apollo、METR、Eleos（外部评估）。外部评估者与实验室及英国 AISI / CAISI 合作。发表的论文反馈给生态系统，供下一届 MATS 使用。

### 为何这一层至关重要

单一来源评估是不可靠的：实验室评估自己的模型存在结构性利益冲突。外部评估者可以提出并验证实验室可能少报的失效模式。2024 年《睡眠代理》论文（第 7 课）是 Anthropic + Redwood；《对齐伪装》是 Anthropic + Redwood；《上下文谋算》是 Apollo；《反谋算》是 Apollo + OpenAI。多组织结构是质量控制机制。

### 在 Phase 18 中的位置

第 7-11 课引用了 Redwood 和 Apollo 的工作；第 18 课引用了 METR 的框架比较；第 19 课引用了 Eleos。第 28 课是整个 Phase 依赖的生态系统的明确组织地图。

## 动手实践

无代码。阅读 METR 的《前沿 AI 安全政策的共同要素》，作为外部综合如何为实验室内部政策工作增添价值的示例。

## 产出技能

本课产出 `outputs/skill-ecosystem-map.md`。给定对齐声明或评估，找出所属组织、发表场所和方法论风格，并与已知的对应组织进行交叉核查。

## 练习

1. 从第 7-15 课中选择一篇论文，找出涉及的组织。将作者与 MATS 校友和当前生态系统归属进行交叉核查。

2. 阅读 METR 的《前沿 AI 安全政策的共同要素》。找出他们强调的三个跨实验室共识点和两个最大的分歧点。

3. MATS 的职业成果约 80% 是安全/安保方向。论述这种选择压力是适应性的（培训该领域）还是有偏见的（过滤掉异质性立场）。

4. Redwood 和 Apollo 都从事控制/谋算工作，但风格不同。选择一种失效模式，描述每个组织会如何调查它。

5. Eleos AI 是唯一纯粹的模型福利组织。设计一个专注于不同福利相关问题（认知自由、机器人具身等）的假想第二组织，并阐明其方法论。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| MATS | "导师制项目" | ML 对齐与理论学者项目；自 2021 年以来 527+ 名研究人员 |
| Redwood Research | "控制实验室" | 应用对齐；AI 控制作者；英国 AISI 合作伙伴 |
| Apollo Research | "谋算评估" | 针对前沿实验室的预部署谋算评估 |
| METR | "任务时间跨度评估" | 基于任务的能力评估；框架综合 |
| Eleos AI | "福利实验室" | 模型福利预部署评估 |
| Talent pipeline（人才输送管道） | "MATS -> 实验室" | MATS 毕业生流向 Anthropic、DM、OpenAI、Redwood、Apollo、METR |
| External evaluation（外部评估） | "非实验室核查" | 非由模型生产者进行的评估；增加可信度 |

## 延伸阅读

- [MATS（ML 对齐与理论学者项目）](https://www.matsprogram.org/) — 导师制项目
- [Redwood Research](https://www.redwoodresearch.org/) — AI 控制论文
- [Apollo Research](https://www.apolloresearch.ai/) — 谋算评估
- [METR——前沿 AI 安全政策的共同要素](https://metr.org/blog/2025-03-26-common-elements-of-frontier-ai-safety-policies/) — 框架比较
- [Eleos AI Research](https://www.eleosai.org/research) — 模型福利方法论
