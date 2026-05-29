# 多智能体原语模型

> 2026 年发布的每个多智能体框架——AutoGen、LangGraph、CrewAI、OpenAI Agents SDK、Microsoft Agent Framework——都是一个四维设计空间中的一个点。四个原语，不多不少：智能体、交接、共享状态、编排者。本课从零构建它们，在所有四个上运行一个玩具系统，然后将每个主要框架映射到相同的轴上，这样你就可以用一段话读懂任何新版本。

**类型：** 学习
**编程语言：** Python（标准库）
**前置知识：** Phase 14（智能体工程）、Phase 16 · 01（为什么需要多智能体）
**预计时间：** 约 60 分钟

## 问题背景

每六个月就有一个新的多智能体框架发布。2023 年的 AutoGen。2024 年的 CrewAI。2024 年的 LangGraph 和 OpenAI Swarm。2025 年 4 月的 Google ADK。2026 年 2 月的 Microsoft Agent Framework RC。每个新闻稿都声称是"正确的抽象"。

如果你逐一学习它们，你会精疲力竭。API 看起来不同。文档对"智能体"是什么意见不一。一个框架称其共享内存为"黑板"，另一个称之为"消息池"，第三个称之为"StateGraph"。你开始怀疑该领域只是在翻腾。

不是的。在营销之下，四个原语是稳定的。学一次，用一段话读懂每个新框架。

## 核心概念

### 四个原语

1. **智能体（Agent）** — 系统提示词加工具列表。无状态；每次运行都从其系统提示词和当前消息历史开始。
2. **交接（Handoff）** — 控制权从一个智能体到另一个的结构化转移。机械上是一个返回新智能体的工具调用，或遵循条件的图边。
3. **共享状态（Shared state）** — 多个智能体可以读取（有时写入）的任何数据结构。消息池、黑板、键值存储、向量内存。
4. **编排者（Orchestrator）** — 决定谁下一个发言。选项：显式图（确定性）、LLM 发言者选择器（软性）、最后一个发言者的交接调用（OpenAI Swarm）、或队列上的调度器（群体架构）。

这就是整个设计空间。每个框架为每个轴选择默认值；其余的是表面语法。

### 每个 2026 年框架如何映射到它

| 框架 | 智能体 | 交接 | 共享状态 | 编排者 |
|------|--------|------|---------|--------|
| OpenAI Swarm / Agents SDK | `Agent(instructions, tools)` | 工具返回智能体 | 调用者的问题 | LLM 的下一个交接调用 |
| AutoGen v0.4 / AG2 | `ConversableAgent` | GroupChat 上的发言者选择器 | 消息池 | 选择器函数（LLM 或轮询） |
| CrewAI | `Agent(role, goal, backstory)` | `Process.Sequential / Hierarchical` | 任务输出链式传递 | 管理者 LLM 或静态顺序 |
| LangGraph | 节点函数 | 图边 + 条件 | `StateGraph` 减速器 | 图，确定性 |
| Microsoft Agent Framework | 智能体 + 编排模式 | 模式特定 | 线程/上下文 | 模式特定 |
| Google ADK | 智能体 + A2A 卡片 | A2A 任务 | A2A 工件 | 主机决定 |

表面差异看起来很大。底层：相同的四个旋钮。

### 为什么这很重要

一旦你看到了原语，框架比较就变成了一个简短的检查清单：

- 编排者是否信任 LLM 来路由（Swarm）还是将路由固定在代码中（LangGraph）？
- 共享状态是全历史（GroupChat）还是投影的（StateGraph 减速器）？
- 智能体可以修改彼此的提示词（CrewAI 管理者）还是只能交接（Swarm）？

这三个问题回答了 80% 的关于哪个框架适合给定问题。你停止购物"最好的多智能体框架"，开始为你实际关心的轴进行设计。

### 无状态洞察

除共享状态之外的每个原语都是无状态的。智能体是（提示词、工具）的函数。交接是函数调用。编排者是调度器。**系统中唯一有状态的东西是共享状态。** 所有有趣的错误都在那里：内存中毒（第 15 课）、消息排序、版本控制、写入竞争。

隐藏共享状态的框架（Swarm）将问题推给调用者。集中化它的框架（LangGraph 检查点、AutoGen 池）使其可检查，但将协调成本转移到共享状态实现上。

### 单个原语的解剖

#### 智能体

```
Agent = (system_prompt, tools, model, optional_name)
```

没有内存。没有状态。具有相同系统提示词和工具的两个智能体是可互换的。看起来像每智能体状态的所有东西实际上都在共享状态或交接协议中。

#### 交接

```
Handoff = (from_agent, to_agent, reason, payload)
```

三种主要实现：

- **函数返回** — 工具返回下一个智能体。这是 OpenAI Swarm 模式。智能体在其工具架构中携带路由。
- **图边** — LangGraph。边是声明性的。LLM 产生一个值；条件选择下一个节点。
- **发言者选择** — AutoGen GroupChat。选择器函数（有时本身是 LLM 调用）读取池并选择下一个发言者。

#### 共享状态

```
SharedState = { messages: [], artifacts: {}, context: {} }
```

最少是消息列表。通常更多：结构化工件（CrewAI 任务输出）、类型化上下文（LangGraph 减速器）、外部内存（MCP、向量数据库）。

两种拓扑：**完整池**（每个智能体看到每条消息）和**投影**（智能体看到角色范围的视图）。完整池简单且扩展性差。投影池可扩展但需要前期模式设计。

#### 编排者

```
Orchestrator = ({state, last_speaker}) -> next_agent
```

四种类型：

- **静态** — 图在构建时固定（LangGraph 确定性、CrewAI 顺序）。
- **LLM 选择** — LLM 读取池并选择下一个发言者（AutoGen、CrewAI 分层）。
- **交接驱动** — 当前智能体通过调用交接工具来决定（Swarm）。
- **队列驱动** — 工作者从共享队列中拉取；没有显式的下一个发言者（群体架构、Matrix）。

### 框架之间有什么变化

一旦原语固定，剩余的设计决策是：

- **内存策略** — 短暂的与持久检查点（LangGraph checkpointer）。
- **安全边界** — 谁可以批准交接（人机协作）。
- **成本核算** — 每智能体 token 预算。
- **可观察性** — 追踪交接，持久化状态以供重放。

所有这些都可以在原语之上实现。它们都不是新原语。

## 动手实践

`code/main.py` 用约 150 行标准库 Python 实现四个原语。没有真实 LLM——每个智能体是一个脚本化策略，这样焦点保持在协调结构上。

文件导出：

- `Agent` — 名称、系统提示词、工具、策略函数的数据类。
- `Handoff` — 返回新智能体的函数。
- `SharedState` — 线程安全的消息池。
- `Orchestrator` — 三个变体：`StaticOrchestrator`、`HandoffOrchestrator`、`LLMSelectorOrchestrator`（模拟）。

演示通过所有三种编排者类型运行相同的三智能体管道（研究 → 写作 → 审查），并在最后打印消息池。你可以看到输出只在*谁选择下一个*上有所不同；智能体和共享状态在运行中是相同的。

运行：

```
python3 code/main.py
```

预期输出：三个编排者运行，每个模式一个。每个打印最终消息池。如果研究者决定提前完成，交接驱动的运行会到达更少的智能体——这是 LLM 路由权衡的缩影。

## 产出技能

`outputs/skill-primitive-mapper.md` 是一个读取任何多智能体代码库或框架文档并返回四原语映射的技能。对新框架版本运行它，在深入阅读文档之前获得一段话的理解。

## 输出

在采用新框架之前，为其写出原语映射。如果你做不到，文档不完整或框架在发明第五个原语（罕见——检查你没见过的共享状态类型）。

将映射固定在你的架构文档中。当新团队成员加入时，在 API 文档之前发送映射给他们。当框架版本改变时，差异化映射，而不是变更日志。

## 练习

1. 用不同的智能体策略运行 `code/main.py` 三次。观察编排者选择如何改变哪些智能体运行。
2. 实现第四种编排者类型：队列驱动的，智能体轮询共享状态寻找工作。可能发生什么死锁，如何检测它？
3. 以四个原语重写 LangGraph 快速入门（https://docs.langchain.com/oss/python/langgraph/workflows-agents）。LangGraph 的哪些抽象 1:1 映射，哪些是便利包装器？
4. 阅读 OpenAI Swarm cookbook（https://developers.openai.com/cookbook/examples/orchestrating_agents）。找出 Swarm 使哪个原语最符合人体工程学，哪个推给调用者。
5. 找出这个表中完全隐藏共享状态的一个框架。解释当智能体需要在交接之间协调而不重新阅读历史时会发生什么。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| 智能体（Agent） | "带工具的 LLM" | `(system_prompt, tools, model)` 三元组。无状态。 |
| 交接（Handoff） | "控制权转移" | 命名下一个智能体和可选有效负载的结构化调用。三种实现：函数返回、图边、发言者选择。 |
| 共享状态（Shared state） | "内存"/"上下文" | 多智能体系统中唯一有状态的部分。消息池或黑板。 |
| 编排者（Orchestrator） | "协调者" | 决定谁下一个运行。静态图、LLM 选择器、交接驱动或队列驱动。 |
| 原语（Primitive） | "抽象" | 每个框架参数化的四个轴之一。不是框架特性。 |
| 消息池（Message pool） | "共享聊天历史" | 全历史共享状态。容易推理，扩展性差。 |
| 投影状态（Projected state） | "范围视图" | 角色特定的共享状态视图。可扩展，需要模式设计。 |
| 发言者选择（Speaker selection） | "谁下一个说话" | 编排者模式，函数（通常是 LLM）从组中选择下一个智能体。 |

## 延伸阅读

- [OpenAI cookbook: 编排智能体——例程和交接](https://developers.openai.com/cookbook/examples/orchestrating_agents) — 交接驱动编排最清晰的阐述
- [AutoGen 稳定文档](https://microsoft.github.io/autogen/stable/) — GroupChat + 发言者选择是 LLM 选择编排的参考
- [LangGraph 工作流和智能体](https://docs.langchain.com/oss/python/langgraph/workflows-agents) — 图边编排和基于减速器的共享状态
- [CrewAI 介绍](https://docs.crewai.com/en/introduction) — 角色-目标-背景智能体，顺序/分层流程
- [AG2（社区 AutoGen 延续）](https://github.com/ag2ai/ag2) — Microsoft 将 v0.4 移入维护模式后的活跃 AutoGen v0.2 分支
