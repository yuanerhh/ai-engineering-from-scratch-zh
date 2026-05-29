# 群聊与发言者选择

> AutoGen GroupChat 和 AG2 GroupChat 在 N 个智能体之间共享一个对话；选择器函数（LLM、轮询或自定义）选择谁下一个发言。这是涌现多智能体对话的原型——智能体不知道自己在静态图中的角色，它们只是对共享池做出反应。AutoGen v0.2 的 GroupChat 语义在 AG2 分叉中得到保留；AutoGen v0.4 将其改写为事件驱动的 actor 模型。Microsoft 在 2026 年 2 月将 AutoGen 置于维护模式，并将其与 Semantic Kernel 合并为 Microsoft Agent Framework（RC 2026 年 2 月）。GroupChat 原语在 AG2 和 Microsoft Agent Framework 中都存在——学一次，处处使用。

**类型：** 学习 + 构建
**编程语言：** Python（标准库）
**前置知识：** Phase 16 · 04（原语模型）
**预计时间：** 约 60 分钟

## 问题背景

静态图（LangGraph）在工作流已知时很好。真实对话不是静态的：有时编码者问审查者，有时问研究者，有时问撰写者。硬编码每个可能的交接会产生边爆炸。你想要*智能体对共享池做出反应*，有某个函数决定谁下一个说话。

这正是 AutoGen GroupChat 所做的。

## 核心概念

### 形状

```
              ┌─── shared pool ────┐
              │   m1  m2  m3  ...  │
              └─────────┬──────────┘
                        │ (everyone reads all)
      ┌───────┬─────────┼─────────┬───────┐
      ▼       ▼         ▼         ▼       ▼
    Agent A  Agent B  Agent C  Agent D  Selector
                                           │
                                           ▼
                                  "next speaker = C"
```

每个智能体看到每条消息。每次轮次调用选择器函数来选择谁下一个发言。

### 三种选择器类型

**轮询（Round-robin）。** 固定循环。确定性。在 N 上线性扩展，但忽略上下文——即使话题是法律审查，编码者也会得到轮次。

**LLM 选择。** 调用 LLM 读取最近的池并返回最佳下一个发言者。上下文感知但缓慢：每次轮次增加一次 LLM 调用。AutoGen 的默认值。

**自定义。** 带你想要的任何逻辑的 Python 函数。典型：带回退规则的 LLM 选择（例如，"编码者之后总是给验证者轮次"）。

### ConversableAgent API

```python
agent = ConversableAgent(
    name="coder",
    system_message="You write Python.",
    llm_config={...},
)
chat = GroupChat(agents=[coder, reviewer, tester], messages=[])
manager = GroupChatManager(groupchat=chat, llm_config={...})
```

`GroupChatManager` 持有选择器。当智能体完成轮次时，管理者调用选择器，返回下一个智能体。循环继续直到终止条件。

### 终止

三种常见模式：

- **最大轮数。** 总轮次的硬限制。
- **"TERMINATE"令牌。** 智能体可以发出哨兵消息；管理者在出现时停止。
- **目标达成检查。** 轻量级验证者每轮运行，完成时停止聊天。

### AutoGen → AG2 分叉与 Microsoft Agent Framework 合并

2025 年初，Microsoft 开始了 AutoGen（v0.4）围绕事件驱动 actor 模型的重大重写。社区将 AutoGen v0.2 的 GroupChat 语义分叉为 AG2，保留了早期采用者已集成的 API。

2026 年 2 月，Microsoft 宣布 AutoGen 将进入维护模式，事件驱动 actor 模型合并到 **Microsoft Agent Framework**（RC 2026 年 2 月，现在与 Semantic Kernel 合并）。GroupChat 概念在两个轨道中都存在；实现细节不同。AG2 是 v0.2 兼容代码的首选上游。

### GroupChat 适合的情况

- **涌现对话。** 你不想预先连接每个可能的下一个发言者。
- **角色混合任务。** 编码者问研究者，研究者问档案管理员，档案管理员问回编码者。流程不是 DAG。
- **探索性问题解决。** 想想"头脑风暴会议"，不是"流水线"。

### 失败情况

- **严格确定性。** LLM 选择器可能不一致。相同提示词，不同运行，不同的下一个发言者。
- **顺从级联。** 智能体服从最自信的说话者。明确地反提示词。
- **上下文膨胀。** 每个智能体读取每条消息；10 轮后上下文很大。使用投影（第 15 课）来限定视图范围。
- **热门发言者。** 一个智能体主导对话，因为选择器偏向其专业。将发言者平衡引入为选择器特性。

### 群聊与监督者

相同的原语，不同的默认值：

- 监督者：一个智能体规划，其他的执行。选择器是"询问规划者做什么"。
- 群聊：所有智能体都是对等体；选择器是共享池上的函数。

两者都使用第 04 课的四个原语。群聊默认为 LLM 选择的编排和全池共享状态。

## 动手实践

`code/main.py` 从标准库从头实现 GroupChat。三个智能体（编码者、审查者、管理者），轮询和 LLM 选择变体，以及在 `TERMINATE` 令牌上终止。

演示打印对话记录以及两种变体的选择器决策追踪。

运行：

```
python3 code/main.py
```

## 产出技能

`outputs/skill-groupchat-selector.md` 为给定任务配置 GroupChat 选择器——轮询与 LLM 选择与自定义，以及使用哪些选择器输入（最近消息、智能体专业、轮次计数）。

## 输出

检查清单：

- **最大轮数限制。** 始终。典型任务 10-20 轮。
- **发言者平衡指标。** 追踪每个智能体的轮次；当不平衡超过阈值时发出警报。
- **终止令牌。** `TERMINATE` 或专用验证者智能体。
- **投影或范围内存。** 约 10 条消息后，考虑给每个智能体只有一个范围视图以防止上下文膨胀。
- **选择器日志。** 对于 LLM 选择变体，记录选择器的输入和其选择。否则调试是不可能的。

## 练习

1. 运行 `code/main.py`。比较轮询与 LLM 选择下的对话。每种情况下哪个智能体主导？
2. 在选择器中添加"每智能体最大发言次数"规则。它如何影响记录？
3. 实现目标达成终止：当审查者返回"已批准"时停止。它在轮数限制之前触发多少次？
4. 阅读 AutoGen 稳定文档中的 GroupChat（https://microsoft.github.io/autogen/stable/user-guide/core-user-guide/design-patterns/group-chat.html）。找出 `GroupChatManager` 使用的默认选择器。
5. 阅读 AG2 仓库（https://github.com/ag2ai/ag2）并比较其 v0.2 GroupChat 与 v0.4 事件驱动版本。v0.4 添加了什么具体属性（吞吐量、容错、可组合性）？

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| GroupChat | "智能体在一个聊天室" | 共享消息池 + 选择器函数。AutoGen/AG2 原语。 |
| 发言者选择（Speaker selection） | "谁下一个说话" | 选择下一个智能体的函数。轮询、LLM 选择或自定义。 |
| GroupChatManager | "会议主持人" | 持有选择器并循环轮次的 AutoGen 组件。 |
| ConversableAgent | "基础智能体" | AutoGen 基类；可以发送和接收消息的智能体。 |
| 终止令牌（Termination token） | "'停止'词" | 结束聊天的哨兵字符串（通常是 `TERMINATE`）。 |
| 热门发言者（Hot speaker） | "一个智能体主导" | 选择器不断选择同一智能体的失败模式。 |
| 上下文膨胀（Context bloat） | "池无限增长" | 每个智能体读取每条先前消息；上下文随轮次增长。 |
| 投影（Projection） | "范围视图" | 角色特定的共享池视图，以防止上下文膨胀。 |

## 延伸阅读

- [AutoGen 群聊文档](https://microsoft.github.io/autogen/stable/user-guide/core-user-guide/design-patterns/group-chat.html) — 参考实现
- [AG2 仓库](https://github.com/ag2ai/ag2) — 社区 AutoGen v0.2 延续
- [Microsoft Agent Framework 文档](https://microsoft.github.io/agent-framework/) — 合并的后继者，RC 2026 年 2 月
- [AutoGen v0.4 发布说明](https://microsoft.github.io/autogen/stable/) — 事件驱动 actor 模型重写详情
