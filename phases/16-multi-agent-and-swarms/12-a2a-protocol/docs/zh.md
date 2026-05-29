# A2A——智能体间协议

> Google 于 2025 年 4 月宣布 A2A；到 2026 年 4 月，规范位于 https://a2a-protocol.org/latest/specification/，150 多个组织支持它。A2A 是 MCP（第 13 课）的水平补充：MCP 是垂直的（智能体 ↔ 工具），A2A 是点对点的（智能体 ↔ 智能体）。它定义了智能体名片（发现）、带工件的任务（文本、结构化数据、视频）、不透明的任务生命周期和认证。生产系统越来越多地将 MCP 与 A2A 配对使用。Google Cloud 在 2025-2026 年间将 A2A 支持集成到了 Vertex AI Agent Builder。

**类型：** 学习 + 构建
**编程语言：** Python（标准库，`http.server`，`json`）
**前置知识：** Phase 16 · 04（原语模型）
**预计时间：** 约 75 分钟

## 问题背景

你的智能体需要调用另一个系统上的另一个智能体。怎么做？你可以暴露一个 HTTP 端点，定义一个专用 JSON 模式，并希望对方能理解它。每对智能体都变成了一个自定义集成。

A2A 是该调用的通用线协议。标准发现、标准任务模型、标准传输、标准工件。就像 HTTP+REST 但以智能体为一等公民。

## 核心概念

### 四个要素

**智能体名片（Agent Card）。** 位于 `/.well-known/agent.json` 的 JSON 文档，描述智能体：名称、技能、端点、支持的模态、认证要求。通过读取名片完成发现。

```
GET https://agent.example.com/.well-known/agent.json
→ {
    "name": "code-review-agent",
    "skills": ["review-python", "review-typescript"],
    "endpoints": {
      "tasks": "https://agent.example.com/tasks"
    },
    "auth": {"type": "bearer"},
    "modalities": ["text", "structured"]
  }
```

**任务（Task）。** 工作单元。一个异步、有状态的对象，具有生命周期：`submitted（已提交）→ working（处理中）→ completed（已完成）/ failed（已失败）/ canceled（已取消）`。客户端发送任务，轮询或订阅更新。

**工件（Artifact）。** 任务产生的结果类型。文本、结构化 JSON、图像、视频、音频。工件有类型，使不同模态成为一等公民。

**不透明生命周期（Opaque lifecycle）。** A2A 不规定远程智能体*如何*解决任务。客户端看到状态转换和工件；实现可以自由使用任何框架。

### MCP/A2A 分工

- **MCP**（第 13 课）：智能体 ↔ 工具。智能体通过 JSON-RPC 向工具服务器读/写。默认无状态。
- **A2A**：智能体 ↔ 智能体。对等协议；双方都是有自己推理能力的智能体。

生产多智能体系统同时使用两者。A2A 对等体在自己一侧调用 MCP 工具。这种分工使两个关注点保持清晰。

### 发现流程

```
客户端                     智能体服务器
  ├──GET /.well-known/agent.json──>
  <──智能体名片 JSON─────────────
  ├──POST /tasks {skill, input}──>
  <──201 task_id, state=submitted
  ├──GET /tasks/{id}──────────────>
  <──state=working, 42% done──────
  ├──GET /tasks/{id}──────────────>
  <──state=completed, artifacts──
```

或使用流式传输：通过 SSE 订阅 `/tasks/{id}/events` 以获取推送更新。

### 认证

A2A 支持三种常见模式：

- **Bearer 令牌** — OAuth2 或不透明令牌。
- **mTLS** — 双向 TLS；组织之间互相证明身份。
- **签名请求** — 对有效负载进行 HMAC 签名。

认证在智能体名片中声明；客户端发现并遵守。

### 到 2026 年 4 月的 150 多个组织

企业采用推动了 A2A 的规模扩张。标志性成果：A2A 成为企业智能体系统跨越信任边界的方式。Google Cloud 发布了 Vertex AI Agent Builder 的 A2A 支持；Microsoft Agent Framework 支持它；大多数主流框架（LangGraph、CrewAI、AutoGen）都提供了 A2A 适配器。

### A2A 的优势场景

- **跨组织调用。** A 公司的智能体调用 B 公司的智能体。没有 A2A，每对都是专用合同。
- **异构框架。** LangGraph 智能体调用 CrewAI 智能体调用自定义 Python 智能体。A2A 标准化了接口。
- **类型化工件。** 视频结果、结构化 JSON、音频——全部是一等公民。
- **长时间运行的任务。** 不透明生命周期 + 轮询使数小时的任务变得简单。

### A2A 的劣势场景

- **延迟敏感的微调用。** A2A 的生命周期是异步的。亚毫秒级的智能体间调用不适合；使用直接 RPC。
- **紧耦合的进程内智能体。** 如果两个智能体运行在同一个 Python 进程中，A2A 的 HTTP 往返是过度设计。
- **小团队。** 规范开销是真实存在的；仅内部使用的智能体可能不需要这种正式性。

### A2A vs ACP、ANP、NLIP

2024-2026 年间出现了几个相关规范：

- **ACP**（IBM/Linux Foundation）— A2A 的前身，范围更窄。
- **ANP**（智能体网络协议）— 重点在对等发现，去中心化优先。
- **NLIP**（Ecma 自然语言交互协议，2025 年 12 月标准化）— 自然语言内容类型。

截至 2026 年 4 月，A2A 是采用最广泛的对等协议。参见 arXiv:2505.02279（Liu 等人，"智能体互操作协议调查"）进行比较。

## 动手实践

`code/main.py` 使用 `http.server` 和 JSON 实现了一个最小化 A2A 服务器和客户端。服务器：

- 暴露 `/.well-known/agent.json`，
- 接受 `POST /tasks`，
- 管理任务状态，
- 在 `GET /tasks/{id}` 时返回工件。

客户端：

- 获取智能体名片，
- 提交任务，
- 轮询直到完成，
- 读取工件。

运行：

```
python3 code/main.py
```

脚本在后台线程中启动服务器，然后运行客户端与其交互。你会看到完整流程：发现、提交、轮询、工件。

## 产出技能

`outputs/skill-a2a-integrator.md` 设计 A2A 集成：智能体名片内容、任务模式、认证选择、流式传输与轮询。

## 输出

检查清单：

- **固定规范版本。** A2A 仍在演进；智能体名片应声明协议版本。
- **幂等任务创建。** 重复提交（网络重试）应只产生一个任务。
- **工件模式。** 声明智能体返回的形状；消费者应验证。
- **速率限制 + 认证。** A2A 面向公众；应用标准 Web 安全措施。
- **失败任务的死信队列。** 随时间检查模式以识别反复出现的失败类型。

## 练习

1. 运行 `code/main.py`。确认客户端发现服务器并收到正确的工件。
2. 向服务器添加第二个技能（例如，"summarize"）。更新智能体名片。编写一个根据任务类型选择技能的客户端。
3. 实现 SSE 流式端点：`/tasks/{id}/events`，发出状态变化。客户端需要做什么不同的事情？
4. 阅读 A2A 规范（https://a2a-protocol.org/latest/specification/）。找出规范要求但此演示未实现的三件事。
5. 比较 A2A（智能体名片发现）与 MCP（通过 `listTools` 进行服务器端能力列举）。自描述智能体与能力探测之间的权衡是什么？

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| A2A | "智能体间协议" | 智能体跨系统调用其他智能体的对等协议。Google 2025。 |
| 智能体名片（Agent Card） | "智能体的名片" | `/.well-known/agent.json` 处的 JSON，描述技能、端点、认证。 |
| 任务（Task） | "工作单元" | 具有生命周期的异步有状态对象；完成时产生工件。 |
| 工件（Artifact） | "结果" | 类型化输出：文本、结构化 JSON、图像、视频、音频。一等媒体。 |
| 不透明生命周期（Opaque lifecycle） | "如何解决是智能体的事" | 客户端看到状态转换；服务器可以自由选择框架/工具。 |
| 发现（Discovery） | "找到智能体" | `GET /.well-known/agent.json` 返回名片。 |
| MCP vs A2A | "工具 vs 对等体" | MCP：垂直 智能体 ↔ 工具。A2A：水平 智能体 ↔ 智能体。 |
| ACP / ANP / NLIP | "兄弟协议" | 相邻规范；A2A 是 2026 年采用最广泛的。 |

## 延伸阅读

- [A2A 规范](https://a2a-protocol.org/latest/specification/) — 规范文档
- [Google 开发者博客 — A2A 公告](https://developers.googleblog.com/en/a2a-a-new-era-of-agent-interoperability/) — 2025 年 4 月发布帖
- [A2A GitHub 仓库](https://github.com/a2aproject/A2A) — 参考实现和 SDK
- [Liu 等人 — 智能体互操作协议调查](https://arxiv.org/html/2505.02279v1) — MCP、ACP、A2A、ANP 比较
