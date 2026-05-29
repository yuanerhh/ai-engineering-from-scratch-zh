# Kubernetes 上的 GPU 自动扩缩容——Karpenter、KAI 调度器、Gang 调度

> 三层，而非一层。Karpenter 动态供给节点（不到一分钟，比 Cluster Autoscaler 快 40%）。KAI 调度器负责 gang 调度、拓扑感知和分层队列——它能防止"7/8 部分分配"陷阱：七个节点等待并白白消耗资源，只因少了一块 GPU。应用级自动扩缩容器（NVIDIA Dynamo Planner、llm-d 工作负载变体自动扩缩容器）根据推理专用信号——队列深度、KV 缓存利用率——进行扩缩容，而非 CPU/DCGM 占空比。经典 HPA 陷阱是：`DCGM_FI_DEV_GPU_UTIL` 是占空比测量：100% 可能代表 10 个请求或 100 个。vLLM 预分配 KV 缓存内存，所以内存永远不会触发缩容。本课教你组合这三层，并避免默认 Karpenter `WhenEmptyOrUnderutilized` 策略——它会在推理过程中终止正在运行的 GPU 作业。

**类型：** 学习
**编程语言：** Python（标准库，玩具队列深度自动扩缩容模拟器）
**前置知识：** Phase 17 · 02（推理平台经济学）、Phase 17 · 04（vLLM 服务内部原理）
**预计时间：** 约 75 分钟

## 学习目标

- 绘制三层自动扩缩容架构图（节点供给、gang 调度、应用级），并命名每层使用的工具。
- 解释为什么 `DCGM_FI_DEV_GPU_UTIL` 对 vLLM 来说是错误的 HPA 信号，并命名两个替代信号（队列深度、KV 缓存利用率）。
- 描述 gang 调度以及 KAI 调度器防止的部分分配失败模式（8 块 GPU 中 7 块空闲）。
- 命名会终止正在运行的 GPU 作业的 Karpenter 合并策略（`WhenEmptyOrUnderutilized`），并说明 2026 年的安全替代方案。

## 问题背景

你的团队在 Kubernetes 上部署 LLM 服务。你用 `DCGM_FI_DEV_GPU_UTIL` 作为信号设置了 HPA。服务在工作时间利用率固定在 100%。HPA 从不扩容——它已经认为你满了。你手动添加一个副本；首 token 延迟下降。HPA 仍然不扩容。信号在欺骗你。

另外，你用 Cluster Autoscaler 管理节点。凌晨 2 点来了一个 100 万 token 的提示词；集群花了 3 分钟供给节点，请求超时。

再另外，你部署一个需要跨 2 个节点使用 8 块 GPU 的 70B 模型。集群有 7 块空闲 GPU，1 块分散在 3 个节点上。Cluster Autoscaler 为那块缺少的 GPU 供给一个节点。七个节点等待 4 分钟烧钱，等待 Kubernetes 启动最后一块 GPU。

三层，三种不同的失败模式。2026 年 GPU 感知自动扩缩容不是"打开 HPA"，而是组合节点供给、gang 调度和应用信号扩缩容。

## 核心概念

### 第一层——节点供给（Karpenter）

Karpenter 监视 pending 的 Pod 并在约 45-60 秒内供给节点（Cluster Autoscaler 对 GPU 节点通常需要 90-120 秒）。它根据 `NodePool` 约束动态选择实例类型——如果你的 Pod 需要 8 块 H100 而集群没有匹配的节点，Karpenter 直接供给一个，而不是扩展现有节点组。

**合并陷阱**：Karpenter 默认的 `consolidationPolicy: WhenEmptyOrUnderutilized` 对 GPU 池来说很危险。它会终止正在运行的 GPU 节点，将 Pod 迁移到更便宜的合适规格实例。对于推理工作负载，这意味着驱逐正在运行的请求，并在新节点上重新加载 70B 模型。损失是数分钟的容量加上请求失败。

GPU 池的安全设置：

```yaml
disruption:
  consolidationPolicy: WhenEmpty
  consolidateAfter: 1h
```

允许 Karpenter 在一小时后合并真正空的节点，但永远不驱逐正在运行的作业。

### 第二层——Gang 调度（KAI 调度器）

KAI 调度器（项目曾命名为"Karp"，后改名）处理默认 kube-scheduler 不处理的问题：

**Gang 调度** — 全有或全无调度。需要 8 块 GPU 的分布式推理 Pod，要么 8 个全部一起启动，要么一个都不启动。没有这个机制，就会出现部分分配陷阱：8 个 Pod 中的 7 个启动，无限等待，烧钱。

**拓扑感知** — 了解哪些 GPU 共享 NVLink，哪些在同一机架上，哪些之间有 InfiniBand。相应地放置 Pod。DeepSeek-V3 67B 张量并行工作负载必须保持在一个 NVLink 域内；KAI 调度器遵从这一点。

**分层队列** — 多个团队以优先级和配额竞争同一个 GPU 池。团队 A 的生产紧急任务只有在优先级规则允许时，才会被团队 B 的训练作业抢占。

KAI 作为辅助调度器与 kube-scheduler 并排部署；你为工作负载添加注解来使用它。Ray 和 vLLM 生产栈都已集成。

### 第三层——应用级信号

**HPA 陷阱**：`DCGM_FI_DEV_GPU_UTIL` 是占空比指标——它测量 GPU 在每个采样间隔是否在工作。100% 利用率可能意味着 10 个并发请求或 100 个；GPU 在两种情况下都很忙。基于占空比扩缩容是盲目扩缩容。

更糟的是，vLLM 和类似引擎预分配 KV 缓存内存（最高达 `--gpu-memory-utilization`）。即使只有一个请求，内存使用量也接近 90%。基于内存的 HPA 永远不会缩容。

**2026 年替代信号**：

- 队列深度（等待预填充的请求数）。
- KV 缓存利用率（分配给活跃序列的块比例）。
- 每副本 P99 首 token 延迟（你的 SLA 信号）。
- Goodput（每秒满足所有 SLO 的请求数）。

NVIDIA Dynamo Planner 和 llm-d 工作负载变体自动扩缩容器消费这些信号并扩缩副本。它们完全替代 LLM 服务的 HPA。

### 何时使用什么

| 扩缩容决策 | 工具 |
|-----------|------|
| 添加/移除节点 | Karpenter |
| 调度多 GPU 作业 | KAI 调度器 |
| 添加/移除副本 | Dynamo Planner / llm-d WVA（或基于队列深度的自定义 HPA） |
| 选择 GPU 类型 | Karpenter NodePool |
| 抢占低优先级 | KAI 调度器队列 |

### 分离预填充/解码让一切更复杂

如果你运行分离式预填充/解码（Phase 17 · 17），你有两类 Pod，触发扩缩容的信号不同：预填充 Pod 基于队列深度扩缩容，解码 Pod 基于 KV 缓存压力扩缩容。llm-d 将这些暴露为独立的 `Service`，每个角色各自对应 HPA。不要尝试用一个 HPA 放在两者前面。

### 冷启动在这里同样重要

冷启动缓解（Phase 17 · 10）是节点供给时间变为用户可见的环节。Karpenter 45-60 秒预热加上 20GB 模型加载加上引擎初始化，意味着从零开始的请求需要 2-5 分钟。对 SLO 关键路径保持温热池（`min_workers=1`），或在应用层使用 Modal 风格的检查点。

### 需要记住的数字

- Karpenter 节点供给：约 45-60 秒，vs Cluster Autoscaler 约 90-120 秒（GPU 节点）。
- KAI 调度器防止部分分配浪费——7/8 陷阱。
- `DCGM_FI_DEV_GPU_UTIL` 作为 HPA 信号：已损坏；使用队列深度或 KV 利用率。
- Karpenter `WhenEmptyOrUnderutilized`：终止正在运行的 GPU 作业。推理时使用 `WhenEmpty + consolidateAfter: 1h`。

## 动手实践

`code/main.py` 在突发 GPU 工作负载上模拟三层自动扩缩容器。比较朴素 HPA（占空比）、队列深度 HPA 和 KAI Gang 调度扩缩容。报告未满足请求数、空闲 GPU 分钟数和综合评分。

## 产出技能

本课产出 `outputs/skill-gpu-autoscaler-plan.md`。给定集群拓扑、工作负载形状和 SLO，设计三层自动扩缩容方案。

## 练习

1. 运行 `code/main.py`。在突发工作负载下，朴素占空比 HPA 丢弃了多少请求是队列深度 HPA 能接住的？差异来自哪里？
2. 为在 H100 SXM5 上服务 Llama 3.3 70B FP8 的集群设计 Karpenter NodePool。指定 `capacity-type`、`disruption.consolidationPolicy`、`consolidateAfter`，以及防止非 GPU 工作负载进入这些节点的污点。
3. 你的团队报告部署卡在 Pending，因为"GPU 可用但 Pod 不调度"。诊断——是 Karpenter、kube-scheduler 还是 KAI 调度器的问题？哪些指标可以确认？
4. 为分离式预填充 Pod 选择一个扩缩容信号，为解码 Pod 选择另一个不同的信号。各自说明理由。
5. 计算 `WhenEmptyOrUnderutilized` 合并陷阱在一个每天平均 60 次 P99 首 token 延迟 > 10 秒请求丢弃事件的 24x7 生产服务上的成本。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| Karpenter | "节点供给器" | Kubernetes 节点自动扩缩容器；亚分钟级供给 |
| Cluster Autoscaler | "老版扩缩容器" | Kubernetes 节点自动扩缩容器前身；更慢，基于节点组 |
| KAI 调度器（KAI Scheduler） | "GPU 调度器" | 用于 gang + 拓扑 + 队列的辅助调度器 |
| Gang 调度（Gang scheduling） | "全有或全无" | 原子性调度 N 个 Pod，否则全部推迟 |
| 拓扑感知（Topology awareness） | "机架感知" | 根据 NVLink/IB/机架位置放置 Pod |
| `DCGM_FI_DEV_GPU_UTIL` | "GPU 利用率" | 占空比指标；不适合作为 LLM 的扩缩容信号 |
| 队列深度（Queue depth） | "等待请求数" | 预填充受限扩缩容的正确 HPA 信号 |
| KV 缓存利用率（KV cache utilization） | "内存压力" | 解码受限扩缩容的正确 HPA 信号 |
| 合并（Consolidation） | "Karpenter 合并" | 将节点迁移到更便宜的实例类型 |
| `WhenEmpty + 1h` | "安全合并" | 不驱逐正在运行 GPU 作业的策略 |

## 延伸阅读

- [KAI 调度器 GitHub](https://github.com/kai-scheduler/KAI-Scheduler) — 设计文档和配置示例
- [Karpenter 中断控制](https://karpenter.sh/docs/concepts/disruption/) — 合并策略语义和 GPU 安全默认值
- [NVIDIA — Kubernetes 上的分离式 LLM 推理](https://developer.nvidia.com/blog/deploying-disaggregated-llm-inference-workloads-on-kubernetes/) — Dynamo Planner 扩缩容信号
- [Ray 文档——RayCluster 的 KAI 调度器](https://docs.ray.io/en/latest/cluster/kubernetes/k8s-ecosystem/kai-scheduler.html) — Ray 集成模式
- [AWS EKS 计算和自动扩缩容最佳实践](https://docs.aws.amazon.com/eks/latest/best-practices/aiml-compute.html) — 托管 Kubernetes 专项指南
- [llm-d GitHub](https://github.com/llm-d/llm-d) — 工作负载变体自动扩缩容器设计
