# LLM 功能 A/B 测试——GrowthBook、Statsig 与"感觉不错"问题

> 传统 A/B 测试并非为非确定性 LLM 设计。关键区别：评估（eval）回答"模型能完成工作吗？"A/B 测试回答"用户在乎吗？"两者都必需；凭感觉上线的时代已过去。2026 年值得测试的内容：提示词工程（措辞）、模型选择（GPT-4 vs GPT-3.5 vs OSS；准确率 vs 成本 vs 延迟）、生成参数（temperature、top-p）。真实案例：聊天机器人奖励模型变体带来 +70% 对话长度和 +30% 留存率；Nextdoor AI 主题行实验在奖励函数优化后实现 +1% CTR；可汗学院 Khanmigo 在延迟 vs 数学准确率轴上持续迭代。平台分化：**Statsig**（2025 年 9 月被 OpenAI 以 11 亿美元收购）——序贯测试、CUPED、一体化。**GrowthBook**——开源、仓库原生、贝叶斯 + 频率主义 + 序贯引擎、CUPED、SRM 检查、Benjamini-Hochberg + Bonferroni 修正。选择依据是仓库 SQL 偏好以及"被 OpenAI 收购"对你的组织是否重要。

**类型：** 学习
**编程语言：** Python（标准库，玩具序贯测试模拟器）
**前置知识：** Phase 17 · 13（可观测性）、Phase 17 · 20（渐进式部署）
**预计时间：** 约 60 分钟

## 学习目标

- 区分评估（"模型能完成工作"）和 A/B 测试（"用户在乎吗"）。
- 列举三个可测试维度（提示词、模型、参数），并为每个维度选择指标。
- 解释 CUPED、序贯测试和 Benjamini-Hochberg 多重比较修正。
- 根据仓库 SQL 立场和企业收购态度，选择 Statsig 或 GrowthBook。

## 问题背景

你手动调优了系统提示词。感觉更好了。你上线了。转化率变化在噪声范围内。你怪指标不对。或者你上线了新模型，转化率没有变化——是模型退化了还是变化太小检测不到？你不知道，因为你没有做 A/B 测试就上线了。

评估回答模型能否在标注集上完成任务。它无法回答用户是否更喜欢输出结果。只有受控在线实验才能回答这个问题，而且前提是实验有足够的统计功效、控制了非确定性，并且修正了多重比较。

## 核心概念

### 评估 vs A/B 测试

**评估（Eval）** — 离线，标注集，裁判（评分标准、LLM 作为裁判或人工）。回答："在这个固定分布上，输出是否正确/有帮助/安全？"

**A/B 测试** — 在线，真实用户，随机化。回答："新变体是否改变了用户层面的关键指标？"

两者都必需。评估在暴露前发现回归；A/B 测试在暴露后确认产品影响。

### 测试什么

1. **提示词工程** — 措辞、系统提示词结构、示例。指标：任务成功率、用户留存率、每请求成本。
2. **模型选择** — GPT-4 vs GPT-3.5-Turbo vs Llama-OSS。指标：准确率（任务）+ 每请求成本 + 延迟 P99。多目标。
3. **生成参数** — temperature、top-p、max_tokens。指标：任务专属（输出多样性 vs 确定性）。

### CUPED——方差缩减

使用实验前数据的受控实验（Controlled-experiments Using Pre-Experiment Data）。在比较实验后期之前，对实验前期方差做回归消除。典型方差缩减：30-70%。有效样本量免费提升。

实现：Statsig 和 GrowthBook 都实现了。

### 序贯测试

经典 A/B 假设固定样本量。序贯测试（"peek-and-decide"）在重复查看下控制假阳性率。始终有效的序贯方法（mSPRT、Howard 置信序列）允许在明确赢家出现时提前停止。

### 多重比较修正

在 95% 置信度下运行 20 个 A/B 测试，平均会有一个假阳性。Bonferroni 修正按测试数量收紧 α；Benjamini-Hochberg 控制错误发现率。GrowthBook 都实现了。

### SRM——样本比例不匹配

分配哈希将用户随机化到变体。如果 50/50 分流实际得到 47/53，说明出了问题——SRM 检查会标记出来。两个平台都实现了。

### Statsig vs GrowthBook

**Statsig**：
- 2025 年 9 月被 OpenAI 以 11 亿美元收购。托管 SaaS。
- 序贯测试、CUPED、留存人群。
- 一体化：功能标志 + 实验 + 可观测性。
- 最适合：团队已经想要捆绑产品，不在意 OpenAI 所有权。

**GrowthBook**：
- 开源（MIT）；仓库原生（直接读取 Snowflake/BigQuery/Redshift）。
- 多引擎：贝叶斯、频率主义、序贯。
- CUPED、SRM、Bonferroni、BH 修正。
- 自托管或托管云。
- 最适合：仓库 SQL 团队，数据团队控制指标层，偏好 OSS。

### 非确定性使功效计算复杂化

相同提示词产生不同输出。传统功效计算假设 IID 观测。在 LLM 非确定性下，有效样本量低于名义样本量。将所需样本量乘以约 1.3-1.5 倍作为安全边际。

### 真实案例结果

- 聊天机器人奖励模型变体：+70% 对话长度，+30% 留存率。
- Nextdoor 主题行：奖励函数优化后 +1% CTR。
- 可汗学院 Khanmigo：延迟 vs 数学准确率权衡的迭代优化。

### 反模式：凭感觉上线

每位资深工程师都能说出一个"感觉更好"就上线、没做 A/B 测试的功能。大多数都让团队几个月没注意到的产品指标回退了。A/B 测试是强制执行机制。

### 需要记住的数字

- Statsig 被 OpenAI 收购：11 亿美元，2025 年 9 月。
- GrowthBook：MIT 开源；贝叶斯 + 频率主义 + 序贯。
- CUPED 方差缩减：30-70%。
- LLM 非确定性 → 样本量缓冲 +30-50%。

## 动手实践

`code/main.py` 用固定和序贯边界模拟序贯 A/B 测试。展示序贯测试如何允许提前停止。

## 产出技能

本课产出 `outputs/skill-ab-plan.md`。给定功能变更、工作负载、基线，选择平台、卡口和样本量。

## 练习

1. 运行 `code/main.py`。对于基线转化率 3%、期望提升 5% 的情况，80% 统计功效需要多大样本量？
2. 为医疗监管的本地部署客户选择 Statsig 还是 GrowthBook。
3. 设计一个测试 GPT-4 vs GPT-3.5 在每张已解决工单成本上的 A/B 测试。主指标、保护指标和次要指标分别是什么？
4. 金丝雀通过但 A/B 显示 -1.2% 转化率。该上线吗？写出升级标准。
5. 将 CUPED 应用于实验前期方差为实验后期 60% 的情况。计算有效样本量提升。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| 评估（Eval） | "离线测试" | 模型能力的标注集评估 |
| A/B 测试（A/B test） | "实验" | 用户上的实时随机对比 |
| CUPED | "方差缩减" | 用实验前期做回归以减少方差 |
| 序贯测试（Sequential test） | "可中途查看的测试" | 允许提前停止的始终有效方法 |
| 多重比较（Multiple comparison） | "家族误差" | 运行多个测试会使假阳性膨胀 |
| Bonferroni | "严格修正" | 将 α 除以测试数量 |
| Benjamini-Hochberg | "BH FDR" | 错误发现率控制，不那么保守 |
| SRM | "坏分流" | 样本比例不匹配；分配缺陷 |
| Statsig | "OpenAI 旗下" | 商业一体化平台，2025 年被收购 |
| GrowthBook | "OSS 那个" | MIT 仓库原生平台 |
| mSPRT | "序贯概率比检验" | 经典序贯方法 |

## 延伸阅读

- [GrowthBook——如何对 AI 做 A/B 测试](https://blog.growthbook.io/how-to-a-b-test-ai-a-practical-guide/)
- [Statsig——超越提示词：数据驱动的 LLM 优化](https://www.statsig.com/blog/llm-optimization-online-experimentation)
- [Statsig vs GrowthBook 对比](https://www.statsig.com/perspectives/ab-testing-feature-flags-comparison-tools)
- [Deng 等——CUPED 论文](https://www.exp-platform.com/Documents/2013-02-CUPED-ImprovingSensitivityOfControlledExperiments.pdf)
- [Howard——置信序列](https://arxiv.org/abs/1810.08240)
