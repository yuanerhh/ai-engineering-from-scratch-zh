# 为什么需要多智能体？

> 单个智能体碰壁了。明智之举不是更大的智能体——而是更多的智能体。

**类型：** 学习
**编程语言：** TypeScript
**前置知识：** Phase 14（智能体工程）
**预计时间：** 约 60 分钟

## 学习目标

- 识别单智能体上限（上下文溢出、混合专业知识、顺序瓶颈），并解释何时拆分为多个智能体是正确的选择
- 比较编排模式（管道、并行扇出、监督者、分层），并为给定任务结构选择正确的模式
- 设计具有清晰角色边界、共享状态和通信合约的多智能体系统
- 分析多智能体复杂性（延迟、成本、调试难度）与单智能体简单性的权衡

## 问题背景

你在 Phase 14 构建了一个单智能体。它有效。它可以读取文件、运行命令、调用 API，并推理结果。然后你把它指向一个真实的代码库：200 个文件，三种语言，依赖基础设施的测试，以及在编写代码之前研究外部 API 的要求。

智能体卡住了。不是因为 LLM 不够聪明，而是因为任务超出了一个智能体循环能处理的范围。上下文窗口被文件内容填满。智能体忘记了 40 次工具调用前读到的内容。它试图同时成为研究员、编码员和审查员，结果三样都做得很差。

这就是单智能体上限。每当任务需要以下情况时你就会碰到它：

- **比一个窗口能容纳的更多上下文** - 读取 50 个文件会超过 200k 个 token
- **不同阶段需要不同专业知识** - 研究需要与代码生成不同的提示词
- **可以并行完成的工作** - 为什么要顺序读取三个文件，而可以同时读取？

## 核心概念

### 单智能体上限

单智能体是一个循环、一个上下文窗口、一个系统提示词。想象一下：

```
┌─────────────────────────────────────────┐
│            SINGLE AGENT                 │
│                                         │
│  ┌───────────────────────────────────┐  │
│  │         Context Window            │  │
│  │                                   │  │
│  │  research notes                   │  │
│  │  + code files                     │  │
│  │  + test output                    │  │
│  │  + review feedback                │  │
│  │  + API docs                       │  │
│  │  + ...                            │  │
│  │                                   │  │
│  │  ██████████████████████ FULL ███  │  │
│  └───────────────────────────────────┘  │
│                                         │
│  One system prompt tries to cover       │
│  research + coding + review + testing   │
│                                         │
│  Result: mediocre at everything         │
└─────────────────────────────────────────┘
```

三件事会出问题：

1. **上下文饱和** - 工具结果堆积。到第 30 轮时，智能体已消耗了 150k 个 token 的文件内容、命令输出和先前的推理。第 5 轮的关键细节丢失了。

2. **角色混乱** - 说"你是研究员、编码员、审查员和测试员"的系统提示词产生一个半研究、半编码、从不完成审查的智能体。

3. **顺序瓶颈** - 智能体读取文件 A，然后文件 B，然后文件 C。三次串行 LLM 调用。三次串行工具执行。没有并行性。

### 多智能体解决方案

分割工作。给每个智能体一个工作、一个上下文窗口和一个针对该工作调整的系统提示词：

```
┌──────────────────────────────────────────────────────────┐
│                    ORCHESTRATOR                          │
│                                                          │
│  "Build a REST API for user management"                  │
│                                                          │
│         ┌──────────┬──────────┬──────────┐               │
│         │          │          │          │               │
│         ▼          ▼          ▼          ▼               │
│   ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐  │
│   │RESEARCHER│ │  CODER   │ │ REVIEWER │ │  TESTER  │  │
│   │          │ │          │ │          │ │          │  │
│   │ Reads    │ │ Writes   │ │ Checks   │ │ Runs     │  │
│   │ docs,    │ │ code     │ │ code     │ │ tests,   │  │
│   │ finds    │ │ based on │ │ quality, │ │ reports  │  │
│   │ patterns │ │ research │ │ finds    │ │ results  │  │
│   │          │ │ + spec   │ │ bugs     │ │          │  │
│   └─────┬────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘  │
│         │           │            │             │         │
│         └───────────┴────────────┴─────────────┘         │
│                          │                               │
│                     Merge results                        │
└──────────────────────────────────────────────────────────┘
```

每个智能体有：
- 专注的系统提示词（"你是代码审查员。你唯一的工作是发现错误。"）
- 自己的上下文窗口（不被其他智能体的工作污染）
- 清晰的输入/输出合约（接收研究笔记，输出代码）

### 真实系统中的实践

**Claude Code 子智能体** - 当 Claude Code 用 `Task` 生成子智能体时，它创建一个有范围任务的子智能体。父智能体保持上下文干净。子智能体进行专注工作并返回摘要。

**Devin** - 运行规划智能体、编码智能体和浏览器智能体。规划者将工作分解为步骤。编码者编写代码。浏览器研究文档。每个都有独立的上下文。

**多智能体编码团队（SWE-bench）** - SWE-bench 上表现最好的系统使用读取代码库的研究者、设计修复的规划者和实现它的编码者。单智能体系统得分更低。

**ChatGPT 深度研究** - 并行生成多个搜索智能体，每个探索不同角度，然后综合结果。

### 谱系

多智能体不是二元的。它是一个谱系：

```
SIMPLE ──────────────────────────────────────────── COMPLEX

 Single        Sub-         Pipeline      Team         Swarm
 Agent         agents

 ┌───┐       ┌───┐        ┌───┐───┐    ┌───┐───┐    ┌─┐┌─┐┌─┐
 │ A │       │ A │        │ A │ B │    │ A │ B │    │ ││ ││ │
 └───┘       └─┬─┘        └───┘─┬─┘    └─┬─┘─┬─┘    └┬┘└┬┘└┬┘
               │                │        │   │       ┌┴──┴──┴┐
             ┌─┴─┐          ┌───┘───┐    │   │       │shared │
             │ a │          │ C │ D │  ┌─┴───┴─┐    │ state │
             └───┘          └───┘───┘  │  msg   │    └───────┘
                                       │  bus   │
 1 loop      Parent +      Stage by    │       │    N peers,
 1 context   child tasks   stage       └───────┘    emergent
                                       Explicit      behavior
                                       roles
```

**单智能体** - 一个循环，一个提示词。适合简单任务。

**子智能体** - 父智能体为专注子任务生成子智能体。父智能体维护计划。子智能体报告回来。这就是 Claude Code 所做的。

**管道** - 智能体按顺序运行。智能体 A 的输出成为智能体 B 的输入。适合分阶段工作流：研究 -> 代码 -> 审查 -> 测试。

**团队** - 智能体并行运行，共享消息总线。每个都有角色。编排者协调。当同时需要不同技能时很好。

**群体** - 许多相同或近似相同的智能体，共享状态。没有固定编排者。智能体从队列中接取工作。适合高吞吐量并行任务。

### 四种多智能体模式

#### 模式 1：管道

```
Input ──▶ Agent A ──▶ Agent B ──▶ Agent C ──▶ Output
          (research)  (code)      (review)
```

每个智能体转换数据并传递前进。易于推理。一个阶段的失败会阻塞其余阶段。

#### 模式 2：扇出/扇入

```
                ┌──▶ Agent A ──┐
                │              │
Input ──▶ Split ├──▶ Agent B ──├──▶ Merge ──▶ Output
                │              │
                └──▶ Agent C ──┘
```

将工作分散到并行智能体，然后合并结果。适合分解为独立子任务的任务。

#### 模式 3：编排者-工作者

```
                    ┌──────────┐
                    │  Orch.   │
                    └──┬───┬───┘
                  task │   │ task
                 ┌─────┘   └─────┐
                 ▼               ▼
           ┌──────────┐   ┌──────────┐
           │ Worker A │   │ Worker B │
           └──────────┘   └──────────┘
```

智能编排者决定做什么，委托给工作者，并综合结果。编排者本身是一个带有生成工作者工具的智能体。

#### 模式 4：对等群体

```
         ┌───┐ ◄──── msg ────▶ ┌───┐
         │ A │                  │ B │
         └─┬─┘                  └─┬─┘
           │                      │
      msg  │    ┌───────────┐     │ msg
           └───▶│  Shared   │◄────┘
                │  State    │
           ┌───▶│  / Queue  │◄────┐
           │    └───────────┘     │
      msg  │                      │ msg
         ┌─┴─┐                  ┌─┴─┐
         │ C │ ◄──── msg ────▶ │ D │
         └───┘                  └───┘
```

没有中央编排者。智能体点对点通信。决策从交互中涌现。更难调试，但可扩展到许多智能体。

### 何时不使用多智能体

多智能体增加了复杂性。智能体之间的每条消息都是潜在的故障点。调试从"读一个对话"变成"追踪五个智能体的消息"。

**坚持单智能体的情况：**
- 任务适合一个上下文窗口（工作数据不到约 100k token）
- 你不需要为不同阶段使用不同的系统提示词
- 顺序执行足够快
- 任务足够简单，拆分它增加的开销多于价值

**复杂性成本：**
- 每个智能体边界都是一个有损压缩步骤：智能体 A 的完整上下文被压缩为智能体 B 的消息
- 协调逻辑（谁做什么，何时，以什么顺序）本身是错误的来源
- 延迟增加：N 个智能体意味着最少 N 次串行 LLM 调用，如果它们需要来回通信则更多
- 成本倍增：每个智能体独立消耗 token

经验法则：如果任务需要不到 20 次工具调用且适合 100k token，保持单智能体。

## 动手实践

### 第 1 步：过载的单智能体

下面是一个试图做所有事情的单智能体。它有一个巨大的系统提示词和一个保存研究、代码和审查的上下文窗口：

```typescript
type AgentResult = {
  content: string;
  tokensUsed: number;
  toolCalls: number;
};

async function singleAgentApproach(task: string): Promise<AgentResult> {
  const systemPrompt = `You are a full-stack developer. You must:
1. Research the requirements
2. Write the code
3. Review the code for bugs
4. Write tests
Do ALL of these in a single conversation.`;

  const contextWindow: string[] = [];
  let totalTokens = 0;
  let totalToolCalls = 0;

  const research = await fakeLLMCall(systemPrompt, `Research: ${task}`);
  contextWindow.push(research.output);
  totalTokens += research.tokens;
  totalToolCalls += research.calls;

  const code = await fakeLLMCall(
    systemPrompt,
    `Given this research:\n${contextWindow.join("\n")}\n\nNow write code for: ${task}`
  );
  contextWindow.push(code.output);
  totalTokens += code.tokens;
  totalToolCalls += code.calls;

  const review = await fakeLLMCall(
    systemPrompt,
    `Given all previous context:\n${contextWindow.join("\n")}\n\nReview the code.`
  );
  contextWindow.push(review.output);
  totalTokens += review.tokens;
  totalToolCalls += review.calls;

  return {
    content: contextWindow.join("\n---\n"),
    tokensUsed: totalTokens,
    toolCalls: totalToolCalls,
  };
}
```

这种方法的问题：
- 上下文窗口随每个阶段增长。到审查步骤时，它包含研究笔记 AND 代码 AND 先前推理。
- 系统提示词是通用的。无法为每个阶段调整。
- 没有任何并行运行。

### 第 2 步：专家智能体

现在拆分它。每个智能体得到一个工作：

```typescript
type SpecialistAgent = {
  name: string;
  systemPrompt: string;
  run: (input: string) => Promise<AgentResult>;
};

function createSpecialist(name: string, systemPrompt: string): SpecialistAgent {
  return {
    name,
    systemPrompt,
    run: async (input: string) => {
      const result = await fakeLLMCall(systemPrompt, input);
      return {
        content: result.output,
        tokensUsed: result.tokens,
        toolCalls: result.calls,
      };
    },
  };
}

const researcher = createSpecialist(
  "researcher",
  "You are a technical researcher. Read documentation, find patterns, and summarize findings. Output only the facts needed for implementation."
);

const coder = createSpecialist(
  "coder",
  "You are a senior TypeScript developer. Given requirements and research notes, write clean, tested code. Nothing else."
);

const reviewer = createSpecialist(
  "reviewer",
  "You are a code reviewer. Find bugs, security issues, and logic errors. Be specific. Cite line numbers."
);
```

每个专家都有专注的提示词。每个都得到一个干净的上下文窗口，只包含它需要的输入。

### 第 3 步：通过消息协调

用显式消息传递将专家连接起来：

```typescript
type AgentMessage = {
  from: string;
  to: string;
  content: string;
  timestamp: number;
};

async function multiAgentApproach(task: string): Promise<AgentResult> {
  const messages: AgentMessage[] = [];
  let totalTokens = 0;
  let totalToolCalls = 0;

  const researchResult = await researcher.run(task);
  messages.push({
    from: "researcher",
    to: "coder",
    content: researchResult.content,
    timestamp: Date.now(),
  });
  totalTokens += researchResult.tokensUsed;
  totalToolCalls += researchResult.toolCalls;

  const coderInput = messages
    .filter((m) => m.to === "coder")
    .map((m) => `[From ${m.from}]: ${m.content}`)
    .join("\n");

  const codeResult = await coder.run(coderInput);
  messages.push({
    from: "coder",
    to: "reviewer",
    content: codeResult.content,
    timestamp: Date.now(),
  });
  totalTokens += codeResult.tokensUsed;
  totalToolCalls += codeResult.toolCalls;

  const reviewerInput = messages
    .filter((m) => m.to === "reviewer")
    .map((m) => `[From ${m.from}]: ${m.content}`)
    .join("\n");

  const reviewResult = await reviewer.run(reviewerInput);
  messages.push({
    from: "reviewer",
    to: "orchestrator",
    content: reviewResult.content,
    timestamp: Date.now(),
  });
  totalTokens += reviewResult.tokensUsed;
  totalToolCalls += reviewResult.toolCalls;

  return {
    content: messages.map((m) => `[${m.from} -> ${m.to}]: ${m.content}`).join("\n\n"),
    tokensUsed: totalTokens,
    toolCalls: totalToolCalls,
  };
}
```

每个智能体只接收发给它的消息。没有上下文污染。研究者的 50k token 文档阅读从未进入审查者的上下文。

### 第 4 步：对比

```typescript
async function compare() {
  const task = "Build a rate limiter middleware for an Express.js API";

  console.log("=== Single Agent ===");
  const single = await singleAgentApproach(task);
  console.log(`Tokens: ${single.tokensUsed}`);
  console.log(`Tool calls: ${single.toolCalls}`);

  console.log("\n=== Multi-Agent ===");
  const multi = await multiAgentApproach(task);
  console.log(`Tokens: ${multi.tokensUsed}`);
  console.log(`Tool calls: ${multi.toolCalls}`);
}
```

多智能体版本使用更多总 token（三个智能体，三次独立 LLM 调用），但每个智能体的上下文保持干净。每个阶段的质量提高，因为系统提示词是专门化的。

## 产出技能

本课产生一个可重用的提示词，用于决定何时使用多智能体。参见 `outputs/prompt-multi-agent-decision.md`。

## 练习

1. 添加第四个专家："测试者"智能体，接收来自编码者的代码和来自审查者的审查反馈，然后编写测试
2. 修改管道，使审查者可以将反馈发回给编码者进行修订循环（最多 2 轮）
3. 将顺序管道转换为扇出：并行运行研究者和"需求分析器"智能体，然后在传递给编码者之前合并它们的输出

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| 群体（Swarm） | "AI 智能体的集体思维" | 一组具有共享状态且没有固定领导者的对等智能体。行为从局部交互中涌现。 |
| 编排者（Orchestrator） | "老板智能体" | 其工具包括生成和管理其他智能体的智能体。它计划和委托，但可能不做实际工作。 |
| 协调者（Coordinator） | "交通警察" | 基于规则在智能体之间路由消息的非智能体组件（通常只是代码，不是 LLM）。 |
| 共识（Consensus） | "智能体同意" | 多个智能体必须达成一致才能继续的协议。当需要解决冲突输出时使用。 |
| 涌现行为（Emergent behavior） | "智能体自己想出来的" | 从智能体交互中产生但没有明确编程的系统级模式。可以是有用的或有害的。 |
| 扇出/扇入（Fan-out / fan-in） | "智能体的 Map-Reduce" | 将任务分散到并行智能体（扇出），然后合并结果（扇入）。 |
| 消息传递（Message passing） | "智能体互相通信" | 智能体之间的通信机制：从一个智能体发送到另一个智能体的结构化数据，替代共享上下文窗口。 |

## 延伸阅读

- [新兴 AI 智能体架构全景](https://arxiv.org/abs/2409.02977) — 多智能体模式调查
- [AutoGen: 使能下一代 LLM 应用](https://arxiv.org/abs/2308.08155) — 微软的多智能体对话框架
- [Claude Code 子智能体文档](https://docs.anthropic.com/en/docs/claude-code) — Claude Code 如何用 Task 委托
- [CrewAI 文档](https://docs.crewai.com/) — 基于角色的多智能体框架
