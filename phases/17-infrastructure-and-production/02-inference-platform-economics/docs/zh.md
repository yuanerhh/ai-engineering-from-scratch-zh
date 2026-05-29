# 推理平台经济学——Fireworks、Together、Baseten、Modal、Replicate、Anyscale

> 2026 年的推理市场不再只是 GPU 时间租赁。它分化为三个细分：定制芯片（Groq、Cerebras、SambaNova）、GPU 平台（Baseten、Together、Fireworks、Modal）和 API 优先市场（Replicate、DeepInfra）。Fireworks 于 2026 年 5 月 1 日上调 GPU 价格 1 美元/小时，40 亿美元估值和每日处理 10 万亿+ token 的体量说明量驱动模式是有效的。Baseten 于 2026 年 1 月以 50 亿美元估值完成 3 亿美元 E 轮融资。竞争定位规则很简单：Fireworks 优化延迟，Together 优化目录广度，Baseten 优化企业级打磨，Modal 优化 Python 原生开发体验，Replicate 优化多模态覆盖，Anyscale 优化分布式 Python。本课为你提供一张可以直接交给创始人的对比矩阵。

**类型：** 学习
**编程语言：** Python（标准库，玩具每次调用经济学比较器）
**前置知识：** Phase 17 · 01（托管 LLM 平台）、Phase 17 · 04（vLLM 服务内部原理）
**预计时间：** 约 60 分钟

## 学习目标

- 命名三个市场细分（定制芯片、GPU 平台、API 优先），并将每个供应商映射到相应细分。
- 解释为什么"按 token"API 定价模型向服务引擎的成本曲线收敛，而非硬件成本曲线。
- 计算至少三家供应商的每请求有效成本，并解释何时按分钟计费（Baseten、Modal）优于按 token 计费。
- 识别哪个平台是给定工作负载的正确默认选择（无服务器突发、稳定高吞吐、微调变体、多模态）。

## 问题背景

你评估了托管超大规模云平台后，决定需要一个更专精、更快的提供商——Fireworks 用于延迟，Together 用于广度，Baseten 用于微调的自定义模型。现在你有六个真实选择，但定价页面对不上。Fireworks 显示每百万 token 费用；Baseten 显示每分钟费用；Modal 显示每秒费用；Replicate 显示每次预测费用。不对工作负载建模就无法直接比较。

更糟的是，每个定价页面背后的商业模式不同。Fireworks 在共享 GPU 上运行自己的定制引擎（FireAttention）；按 token 费率反映其利用率曲线。Baseten 提供 Truss + 专用 GPU；按分钟计费反映独占性。Modal 是真正的 Python 无服务器——按秒计费，冷启动低于 1 秒。输出相同（一个 LLM 响应），但有三种不同的成本函数。

本课对这六家建模，并告诉你何时每家胜出。

## 核心概念

### 三个细分市场

**定制芯片** — Groq（LPU）、Cerebras（WSE）、SambaNova（RDU）。同等模型下，解码速度通常比基于 GPU 的集群快 5-10 倍。按 token 价格更高（Groq 在 2025 年末 Llama-70B 上约为每百万 token 0.99 美元），但延迟敏感型用例无可替代。Groq 是语音智能体和实时翻译的生产首选。

**GPU 平台** — Baseten、Together、Fireworks、Modal、Anyscale。运行在 NVIDIA（2026 年为 H100、H200、B200）或有时 AMD 上。介于"裸 GPU 租赁"（RunPod、Lambda）和"超大规模云托管服务"（Bedrock）之间的经济层。

**API 优先市场** — Replicate、DeepInfra、OpenRouter、Fal。广泛目录，按次预测或按秒计费，强调首次调用时间。

### Fireworks — 延迟优化的 GPU 平台

- FireAttention 引擎（定制）；宣传在等效配置下比 vLLM 延迟低 4 倍。
- 批处理层价格约为无服务器层的 50%，适用于非交互式工作负载。
- 微调模型以与基础模型相同的费率提供服务——这是相较于为你的 LoRA 收取溢价的其他提供商的真实差异化点。
- 2026 年中：2026 年 5 月 1 日起按需 GPU 租赁价格上调 1 美元/小时。规模化后可协商批量定价。
- 财务信号：40 亿美元估值，每日处理 10 万亿+ token。

### Together — 广度优化

- 200+ 个模型，包括上游发布后数日内的开源新版本。
- 在等效 LLM 模型上比 Replicate 便宜 50-70%——"AI 原生云"定位依托于量和目录。
- 推理 + 微调 + 训练集成于一个 API。

### Baseten — 企业级打磨优化

- Truss 框架：将依赖项、密钥和服务配置集成在一个清单中的模型打包工具。
- GPU 范围从 T4 到 B200。按分钟计费，配有合理的冷启动缓解机制。
- SOC 2 Type II，HIPAA 就绪。金融科技和医疗行业的常见选择。
- 50 亿美元估值，2026 年 1 月 E 轮融资（CapitalG、IVP、NVIDIA 出资 3 亿美元）。

### Modal — Python 原生优化

- 纯 Python 基础设施即代码。用 `@modal.function(gpu="A100")` 装饰函数，一条命令部署。
- 按秒计费。预热后冷启动 2-4 秒；小型模型 < 1 秒。
- 8700 万美元 B 轮融资，11 亿美元估值（2025 年）。独立调查中开发者体验评分最高。

### Replicate — 多模态广度

- 按次预测计费。图像、视频和音频模型的默认平台。
- 集成生态系统（Zapier、Vercel、CMS 插件）。
- LLM 按 token 费率竞争力较弱，但在多模态品种上胜出。

### Anyscale — Ray 原生

- 基于 Ray 构建；RayTurbo 是 Anyscale 的专有推理引擎（与 vLLM 竞争）。
- 最适合分布式 Python 工作负载，其中推理步骤是更大计算图中的一个节点。
- 托管 Ray 集群；与 Ray AIR 和 Ray Serve 紧密集成。

### 按 token vs 按分钟——何时各自胜出

当工作负载对延迟不敏感且呈突发性时，按 token 计费更合理——只为实际使用付费。当利用率高且可预测时，按分钟计费更合理——一旦 GPU 饱和，就能胜过按 token。

粗略规则：对于专用 GPU 持续利用率高于约 30% 的工作负载，按分钟（Baseten、Modal）开始优于按 token（Fireworks、Together）。低于此阈值时，按 token 胜出，因为你避免了为空闲付费。

### 定制引擎是真正的护城河

上述每个平台都声称有高于 vLLM 和 SGLang 的定制引擎——FireAttention、RayTurbo、Baseten 的推理栈。定制引擎的声明带有营销色彩——诚实的说法是 vLLM + SGLang 约占生产开源推理的 80%，平台层的差异化因素是开发体验、归因和 SLA。

### 需要记住的数字

- Fireworks GPU 租赁：2026 年 5 月 1 日起上调 1 美元/小时。
- Fireworks 声明：在等效配置下比 vLLM 延迟低 4 倍。
- Together：在 LLM 上比 Replicate 便宜 50-70%。
- Baseten 估值：50 亿美元（E 轮，2026 年 1 月，3 亿美元融资轮）。
- Modal 估值：11 亿美元（B 轮，2025 年）。
- 持续利用率高于约 30% 时，按分钟优于按 token。

## 动手实践

`code/main.py` 在合成工作负载上跨定价模型比较六家供应商。报告每日费用和有效每百万 token 费用。运行它以找到按 token 和按分钟之间的盈亏平衡点。

## 产出技能

本课产出 `outputs/skill-inference-platform-picker.md`。给定工作负载画像、SLA 和预算，选择主要推理平台并指定次优选项。

## 练习

1. 运行 `code/main.py`。对于一台 H100 上的 70B 模型，在多大持续利用率下 Baseten（按分钟）优于 Fireworks（按 token）？自行推导交叉点并与经验法则对比。
2. 你的产品同时提供图像生成、聊天和语音转文字。为每种模态选择平台，并说明统一它们的网关模式。
3. Fireworks 将你主要模型的价格上调 1 美元/小时。如果 40% 的流量转移到批处理层（5 折），对综合成本影响建模。
4. 受监管客户要求 SOC 2 Type II + HIPAA + 专用 GPU。哪三个平台可行，哪个在 FinOps 方面胜出？
5. 比较 Fireworks 无服务器、Together 按需、Baseten 专用和 Replicate API 上 Llama 3.1 70B 每 1000 次预测的成本。10 次/天最便宜的是哪个？10000 次/天呢？

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| 定制芯片（Custom silicon） | "非 GPU 芯片" | Groq LPU、Cerebras WSE、SambaNova RDU——专为解码优化 |
| FireAttention | "Fireworks 引擎" | 定制注意力内核；宣传比 vLLM 延迟低 4 倍 |
| Truss | "Baseten 的格式" | 模型打包清单；依赖项 + 密钥 + 服务配置 |
| 按 token（Per-token） | "API 定价" | 按消耗 token 收费；空闲不付费 |
| 按分钟（Per-minute） | "专用定价" | 按 GPU 挂钟时间收费；高利用率时胜出 |
| 按次预测（Per-prediction） | "Replicate 定价" | 按模型调用次数收费；图像/视频常见 |
| RayTurbo | "Anyscale 引擎" | Ray 上的专有推理；与 vLLM 竞争于 Ray 集群 |
| 批处理层（Batch tier） | "5 折优惠" | 以降低费率排队的非交互式队列；Fireworks、OpenAI 常见 |
| 微调按基础费率（Fine-tuned at base rate） | "Fireworks LoRA" | LoRA 服务请求按基础模型费率收费（差异化点） |

## 延伸阅读

- [Fireworks 定价](https://fireworks.ai/pricing) — 按 token 费率、批处理层、GPU 租赁
- [Baseten 定价](https://www.baseten.co/pricing/) — 按分钟费率、承诺容量、企业层级
- [Modal 定价](https://modal.com/pricing) — 按秒 GPU 费率和免费层
- [Together AI 定价](https://www.together.ai/pricing) — 模型目录和按 token 费率
- [Anyscale 定价](https://www.anyscale.com/pricing) — RayTurbo 和托管 Ray 定价
- [Northflank — Fireworks AI 替代方案](https://northflank.com/blog/7-best-fireworks-ai-alternatives-for-inference) — 对比评估
- [Infrabase — AI 推理 API 提供商 2026](https://infrabase.ai/blog/ai-inference-api-providers-compared) — 供应商格局
