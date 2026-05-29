# 自托管服务选型——llama.cpp、Ollama、TGI、vLLM、SGLang

> 2026 年四大引擎主导自托管推理。根据硬件、规模和生态系统进行选择。**llama.cpp** 在 CPU 上最快——最广泛的模型支持，对量化和线程的完整控制。**Ollama** 是开发者笔记本的一键安装工具，比 llama.cpp 慢约 15-30%（Go + CGo + HTTP 序列化），生产级负载下吞吐量差距 3 倍。**TGI 于 2025 年 12 月 11 日进入维护模式**——仅修复 Bug，原始吞吐量比 vLLM 历史上慢约 10%，但可观测性和 HF 生态系统集成一流。该维护状态使其成为长期赌注的高风险选择——对于新项目，SGLang 或 vLLM 是更安全的默认选项。**vLLM** 是通用生产默认引擎——v0.15.1（2026 年 2 月）增加了 PyTorch 2.10、RTX Blackwell SM120、H200 优化。**SGLang** 是智能体多轮/前缀密集型专家——生产中 400,000+ GPU（xAI、LinkedIn、Cursor、Oracle、GCP、Azure、AWS）。硬件约束：仅 CPU → 只能用 llama.cpp。AMD/非 NVIDIA → 只能用 vLLM（TRT-LLM 是 NVIDIA 独占的）。2026 年管道模式：开发 = Ollama，预发布 = llama.cpp，生产 = vLLM 或 SGLang。整个流程使用相同的 GGUF/HF 权重。

**类型：** 学习
**编程语言：** Python（标准库，引擎决策树遍历器）
**前置知识：** Phase 17 所有引擎相关课程（04、06、07、09、18）
**预计时间：** 约 45 分钟

## 学习目标

- 根据硬件（CPU / AMD / NVIDIA Hopper / Blackwell）、规模（1 用户 / 100 / 10,000）和工作负载（通用聊天 / 智能体 / 长上下文）选择引擎。
- 了解 2026 年 TGI 维护模式状态（2025 年 12 月 11 日），以及为何这使新项目倾向于选择 vLLM 或 SGLang。
- 描述在整个开发/预发布/生产流程中使用相同 GGUF 或 HF 权重的管道模式。
- 解释为什么"仅 CPU"强制使用 llama.cpp，以及"AMD"排除了 TRT-LLM。

## 问题背景

你的团队启动一个新的自托管 LLM 项目。一名工程师说 Ollama，另一名说 vLLM，第三名说"TGI 不是开箱即用吗？"三人在不同上下文中都是对的。没有一个适合所有情况。

2026 年，选择树很重要：硬件优先，规模其次，工作负载第三。还有一个特定的 2025 年事件——TGI 于 12 月 11 日进入维护模式——改变了新项目的默认选择。

## 核心概念

### 五大引擎

| 引擎 | 最适合 | 备注 |
|------|-------|------|
| **llama.cpp** | CPU / 边缘 / 最少依赖 / 最广泛模型支持 | CPU 最快，完全控制 |
| **Ollama** | 开发笔记本，单用户，一键安装 | 比 llama.cpp 慢 15-30%；生产吞吐量差距 3 倍 |
| **TGI** | HF 生态系统，受监管行业 | **2025 年 12 月 11 日进入维护模式** |
| **vLLM** | 通用生产，100+ 用户 | 广泛生产默认；2026 年 2 月 v0.15.1 |
| **SGLang** | 智能体多轮，前缀密集型工作负载 | 生产中 400,000+ GPU |

### 硬件优先决策

**仅 CPU** → llama.cpp。Ollama 也能用但更慢。其他引擎在 CPU 上都没有竞争力。

**AMD GPU** → vLLM（AMD ROCm 支持）。SGLang 也可以。TRT-LLM 是 NVIDIA 独占的，排除。

**NVIDIA Hopper（H100/H200）** → vLLM 或 SGLang 或 TRT-LLM。三者都是顶级。

**NVIDIA Blackwell（B200/GB200）** → TRT-LLM 是吞吐量领先者（Phase 17 · 07）。vLLM 和 SGLang 紧随其后。

**Apple Silicon（M 系列）** → llama.cpp（Metal）。Ollama 封装了这个。

### 规模其次决策

**1 用户 / 本地开发** → Ollama。一个命令，秒级首 token。

**10-100 用户 / 小团队** → vLLM 单 GPU。

**100-10k 用户 / 生产** → vLLM 生产技术栈（Phase 17 · 18）或 SGLang。

**10k+ 用户 / 企业** → vLLM 生产技术栈 + 分离式（Phase 17 · 17）+ LMCache（Phase 17 · 18）。

### 工作负载第三决策

**通用聊天 / 问答** → vLLM 在广泛默认场景中胜出。

**智能体多轮（工具、规划、记忆）** → SGLang 的 RadixAttention（Phase 17 · 06）占主导。

**前缀复用密集型 RAG** → SGLang。

**代码生成** → vLLM 可以；SGLang 在缓存上略优。

**长上下文（128K+）** → vLLM + 分块预填充；SGLang + 分层 KV。

### TGI 维护模式陷阱

Hugging Face TGI 于 2025 年 12 月 11 日进入维护模式——此后只修复 Bug。历史上：顶级可观测性、最佳 HF 生态系统集成（模型卡、安全工具），原始吞吐量略落后于 vLLM。

2026 年的新项目：默认远离 TGI。现有 TGI 部署可以继续，但最终应该迁移。SGLang 和 vLLM 是更安全的默认选择。

### 管道模式

开发（Ollama）→ 预发布（llama.cpp）→ 生产（vLLM）。整个流程使用相同的 GGUF 或 HF 权重。工程师在笔记本上快速迭代；预发布环境镜像生产量化；生产是服务目标。

### Ollama 注意事项

Ollama 非常适合开发。对于共享生产环境则不然：Go HTTP 序列化增加开销，并发管理比 vLLM 更简单，OpenTelemetry 支持滞后。在 Ollama 擅长的场景使用它——单用户、单命令——共享时切换到 vLLM。

### 自托管 vs 托管是独立决策

Phase 17 · 01（托管超大规模云）、· 02（推理平台）涵盖了托管选项。本课假设你已经决定自托管。自托管的理由：数据合规要求、自定义微调、规模化的总体拥有成本、托管平台上不可用的领域模型。

### 需要记住的数字

- TGI 维护模式：2025 年 12 月 11 日。
- vLLM v0.15.1：2026 年 2 月；PyTorch 2.10；Blackwell SM120 支持。
- SGLang 生产规模：400,000+ GPU。
- Ollama 与 llama.cpp 的吞吐量差距：慢 15-30%；生产负载下 3 倍。

## 动手实践

`code/main.py` 是一个决策树遍历器：给定硬件 + 规模 + 工作负载，选择引擎并说明原因。

## 产出技能

本课产出 `outputs/skill-engine-picker.md`。给定约束条件，选择引擎并编写迁移方案。

## 练习

1. 用你的硬件/规模/工作负载运行 `code/main.py`。输出与你的直觉一致吗？
2. 你的基础设施是 12 台 H100 和 8 台 MI300X AMD。选什么引擎？为什么 TRT-LLM 不在考虑之列？
3. 一个团队想在 2026 年继续使用 TGI，因为"这是我们熟悉的"。论证迁移的理由。
4. 从 Ollama 开发切换到 vLLM 生产：量化、配置和可观测性有什么变化？
5. P99 前缀长度 8K、跨租户高度复用的 RAG 产品。选择引擎并结合 Phase 17 · 11 + 18 进行技术栈设计。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| llama.cpp | "CPU 那个" | 最广泛模型支持，CPU 最快 |
| Ollama | "笔记本那个" | 一键安装，开发级吞吐量 |
| TGI | "HF 的服务" | 2025 年 12 月起维护模式 |
| vLLM | "默认选项" | 2026 年广泛生产基线 |
| SGLang | "智能体那个" | 前缀密集型，RadixAttention |
| TRT-LLM | "NVIDIA 独占" | Blackwell 吞吐量领先者，仅限 NVIDIA |
| GGUF | "llama.cpp 格式" | 捆绑 K-量化变体 |
| 生产技术栈（Production-stack） | "vLLM K8s" | Phase 17 · 18 参考部署 |
| 管道模式（Pipeline pattern） | "开发→预发布→生产" | 使用相同权重的 Ollama → llama.cpp → vLLM |

## 延伸阅读

- [AI Made Tools——2026 年 vLLM vs Ollama vs llama.cpp vs TGI](https://www.aimadetools.com/blog/vllm-vs-ollama-vs-llamacpp-vs-tgi/)
- [Morph——2026 年 llama.cpp vs Ollama](https://www.morphllm.com/comparisons/llama-cpp-vs-ollama)
- [n1n.ai——LLM 推理引擎全面对比](https://explore.n1n.ai/blog/llm-inference-engine-comparison-vllm-tgi-tensorrt-sglang-2026-03-13)
- [PremAI——2026 年 10 个最佳 vLLM 替代品](https://blog.premai.io/10-best-vllm-alternatives-for-llm-inference-in-production-2026/)
- [TGI 维护公告](https://github.com/huggingface/text-generation-inference) — 发布说明
- [vLLM v0.15.1 发布说明](https://github.com/vllm-project/vllm/releases)
