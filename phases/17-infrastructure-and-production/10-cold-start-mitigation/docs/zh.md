# 无服务器 LLM 的冷启动缓解

> 20GB 的模型镜像从冷态到服务需要 5-10 分钟（7B）到 20 分钟以上（70B）。在真正的无服务器世界中，这不是预热——这是停机。缓解措施在五层上运作：预置节点镜像（AWS 的 Bottlerocket，双卷架构）、模型流式加载（NVIDIA Run:ai Model Streamer，vLLM 原生支持）、GPU 内存快照（Modal 检查点，重启速度提升最高 10 倍）、温热池（`min_workers=1`）、分级加载（ServerlessLLM 的 NVMe→DRAM→HBM 管道，延迟降低 10-200 倍），以及传输输入 token（KB）而非 KV 缓存（GB）的实时迁移。Modal 发布的冷启动底线为 2-4 秒；Baseten 默认 5-10 秒，预热后低于 1 秒。本课教你测量、规划和叠加这五层。

**类型：** 学习
**编程语言：** Python（标准库，玩具冷启动路径模拟器）
**前置知识：** Phase 17 · 02（推理平台经济学）、Phase 17 · 03（GPU 自动扩缩容）
**预计时间：** 约 60 分钟

## 学习目标

- 枚举冷启动缓解的五层，并在每层命名一个工具或模式。
- 将 70B 模型的总冷启动时间计算为（节点供给）+（权重下载）+（权重加载到 HBM）+（引擎初始化）之和。
- 解释为什么实时迁移传输输入 token（KB）而非 KV 缓存（GB），以及代价是什么（重新计算）。
- 命名温热池的权衡（为空闲 GPU 付费 vs 接受冷启动尾部延迟），以及 `min_workers > 0` 变为必须的 SLA 阈值。

## 问题背景

你的无服务器 LLM 端点在夜间缩放到零。早上 8 点流量激增。第一个请求在等待：

1. Karpenter 供给 GPU 节点：45-60 秒。
2. 容器拉取带权重的 30GB 镜像：120-300 秒。
3. 引擎将权重加载到 HBM：45-120 秒，取决于模型大小和存储速度。
4. vLLM 或 TRT-LLM 初始化 CUDA 图、KV 缓存池、分词器：10-30 秒。

总计：220-510 秒（约 3-8 分钟）才有第一个 token 返回。你的 SLA 是 2 秒。你配置温热池（`min_workers=1`），问题似乎消失了——但现在你 24x7 为一个空闲 GPU 付费。如果你的服务有 5 个产品，每个有一个温热副本，那就是每月 5 × 24 × 30 = 3600 GPU 小时，不管有没有用户调用。

冷启动缓解是在保持无服务器经济性的同时近似始终在线延迟的方法。

## 核心概念

### 第一层——预置节点镜像（Bottlerocket）

在 AWS 上，Bottlerocket 的双卷架构将操作系统与数据分离。用预拉取的容器镜像对数据卷做快照；在 `EC2NodeClass` 中引用快照 ID。新节点启动时权重已在本地 NVMe 上——步骤 2 和部分步骤 3 消失。与 Karpenter 原生配合。大型模型每次冷启动典型节省：2-4 分钟。

GCP 等效方案：预置容器层的自定义 VM 镜像。Azure：相同模式的托管磁盘快照。

### 第二层——模型流式加载（Run:ai Model Streamer）

不是在回答第一个请求之前加载整个文件，而是逐层将权重流式加载到 GPU 内存，并在第一个 Transformer 块就绪后立即开始处理。NVIDIA Run:ai Model Streamer 在 2026 年的 vLLM 中原生支持。支持 S3、GCS 和本地 NVMe。通过将 I/O 与计算设置重叠，将大型模型的权重加载时间大约缩短一半。

### 第三层——GPU 内存快照（Modal）

Modal 在首次加载后对 GPU 状态（权重、CUDA 图、KV 缓存区域）做检查点。后续重启直接将其反序列化到 HBM——比重新初始化快 10 倍。这是最接近"2 秒内启动温热 GPU"的方案。权衡：快照是针对特定 GPU 拓扑的，所以如果 Karpenter 将你迁移到不同的 SKU，需要重新做检查点。

### 第四层——温热池（min_workers=1）

最简单的缓解：始终保持一个副本就绪。成本是一块 GPU 24x7 的每小时费率。对小型模型来说算术很残酷（花 0.85-1.50 美元/小时避免 30 秒冷启动），对大型模型则合理（花 4 美元/小时避免 5 分钟冷启动）。温热池变为必须的 SLA 阈值：通常是 70B+ 模型的 P99 TTFT < 60 秒。

### 第五层——分级加载（ServerlessLLM）

ServerlessLLM 将存储视为层次结构：NVMe（快但大）、DRAM（中等但分层）、HBM（小但即时）。权重预加载到 DRAM；按需加载到 HBM。论文报告相比朴素磁盘到 HBM，冷加载延迟降低 10-200 倍。生产采用尚处早期，但与 vLLM 的集成已存在。

### 第六层——实时迁移（额外模式）

当节点变得不可用（spot 节点被回收、节点驱逐）时，传统模式是冷启动另一个副本并排空请求队列。实时迁移将输入 token（千字节）移动到已加载模型的目标节点，并在目标上重新计算 KV 缓存。重新计算比通过网络传输 GB 级 KV 缓存更便宜。适用于分离式部署。

### 温热池数学

对于 P99 TTFT SLA 为 2 秒的服务，问题不是"要不要温热池"，而是"多少温热副本，以及哪些路径使用"。

- 高价值交互路径（实时聊天、语音智能体）：`min_workers=1-2`。
- 后台批处理路径（夜间分类）：可接受缩放到零，5-10 分钟冷启动可容忍。
- 高级层：每租户 `min_workers` 加专用容量。

### 优化前先测量

70B 模型在全新节点上的冷启动解析（示意）：

| 阶段 | 时间 | 缓解措施 |
|------|------|---------|
| 节点供给 | 50 秒 | Bottlerocket + 预置镜像，温热池 |
| 镜像拉取 | 180 秒 | 预置数据卷（消除） |
| 权重加载到 HBM | 75 秒 | 模型流式（减半）；GPU 快照（消除） |
| 引擎初始化 | 20 秒 | 持久化 CUDA 图缓存 |
| 首次前向传递 | 3 秒 | 固有最低延迟 |
| **冷启动总计** | **328 秒** | |
| **带缓解措施总计** | **约 15 秒** | 22 倍减少 |

### 需要记住的数字

- Modal 冷启动：2-4 秒（带 GPU 快照）。
- Baseten 默认冷启动：5-10 秒；预热后低于 1 秒。
- 原始 70B 冷启动：3-8 分钟。
- Run:ai Model Streamer：约 2 倍权重加载加速。
- ServerlessLLM 分级加载：10-200 倍延迟降低（论文数字）。

## 动手实践

`code/main.py` 对带/不带每种缓解措施的冷启动路径建模。报告总冷启动时间、温热池成本，以及温热池开始盈利的盈亏平衡请求速率。

## 产出技能

本课产出 `outputs/skill-cold-start-planner.md`。给定 SLA、模型大小和流量形状，选择要叠加的缓解措施。

## 练习

1. 运行 `code/main.py`。计算温热副本比通过额外请求丢失缴纳冷启动税更便宜的盈亏平衡请求速率。
2. 你部署一个 P99 TTFT SLA 为 3 秒的 13B 模型。选择达成目标的最小缓解措施堆叠（最少层数）。
3. Bottlerocket 预置消除了镜像拉取，但权重仍需从快照加载到 HBM。如果快照支持的 NVMe 读取速度为 7 GB/s，计算 70B 模型的挂钟时间。
4. 你的无服务器提供商提供 GPU 快照（Modal），你的团队拒绝，因为"快照会泄露 PII"。从两个角度论证——现实风险是什么，以及缓解措施是什么（临时快照、加密、命名空间隔离）？
5. 设计分层温热池策略：付费用户、试用用户和批处理工作负载各需多少温热副本？展示计算过程。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| 冷启动（Cold start） | "大停顿" | 从请求到全新副本首 token 的时间 |
| 温热池（Warm pool） | "始终在线最低值" | `min_workers >= 1` 以保持至少一个副本就绪 |
| 预置镜像（Pre-seeded image） | "烘焙 AMI" | 容器权重预置在节点镜像中 |
| Bottlerocket | "AWS 节点操作系统" | AWS 容器优化操作系统，带双卷快照支持 |
| 模型流式加载（Model streamer） | "流式加载" | 将权重 I/O 与计算设置重叠 |
| GPU 快照（GPU snapshot） | "HBM 检查点" | 序列化加载后的 GPU 状态；重启时反序列化 |
| 分级加载（Tiered loading） | "NVMe + DRAM + HBM" | 存储层次结构；按需加载 |
| 实时迁移（Live migration） | "移动 token" | 传输输入（KB），在目标上重新计算 KV |
| `min_workers` | "温热副本" | 无服务器最低保活数量 |
| 缩放到零（Scale-to-zero） | "完全无服务器" | 空闲时零成本；接受完整冷启动税 |

## 延伸阅读

- [Modal——冷启动性能](https://modal.com/docs/guide/cold-start) — Modal 发布的基准测试和检查点架构
- [AWS Bottlerocket](https://github.com/bottlerocket-os/bottlerocket) — 预置数据卷快照模式
- [NVIDIA Run:ai Model Streamer](https://github.com/run-ai/runai-model-streamer) — 将权重加载与计算设置重叠
- [Baseten——冷启动缓解](https://www.baseten.co/blog/cold-start-mitigation/) — 预热手册
- [ServerlessLLM 论文（USENIX OSDI'24）](https://www.usenix.org/conference/osdi24/presentation/fu) — 分级加载设计
- [NVIDIA——Kubernetes 上的分离式 LLM 推理](https://developer.nvidia.com/blog/deploying-disaggregated-llm-inference-workloads-on-kubernetes/) — 分离式部署的实时迁移
