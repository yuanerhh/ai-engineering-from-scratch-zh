# 分离式预填充/解码——NVIDIA Dynamo 与 llm-d

> 预填充是计算密集型的；解码是内存带宽受限的。在同一块 GPU 上运行两者会浪费其中一种资源。分离式架构将它们拆分到独立的 GPU 池，并通过 NIXL（RDMA/InfiniBand 或 TCP 降级）在池间传输 KV 缓存。NVIDIA Dynamo（GTC 2025 发布，1.0 GA）架设在 vLLM/SGLang/TRT-LLM 之上——其 Planner Profiler + SLA Planner 自动调整预填充:解码比例以满足 SLO。NVIDIA 发布了该范围内的吞吐量增益——developer.nvidia.com（2025-06）显示 GB200 NVL72 + Dynamo 上 DeepSeek-R1 MoE 在中等延迟区间约有 6 倍改进，Dynamo 产品页面（developer.nvidia.com，日期未注明）宣传 GB300 NVL72 + Dynamo 相比 Hopper 的 MoE 吞吐量提升最高 50 倍。"30 倍"数字是跨完整技术栈 Blackwell + Dynamo + DeepSeek-R1 报告的社区汇总；我们没有找到明确声明 30 倍的单一主要来源，因此将其视为方向性说法。llm-d（Red Hat + AWS）是 Kubernetes 原生的：预填充/解码/路由器作为独立 Service，每个角色有独立 HPA。llm-d 0.5 新增了分层 KV 卸载、缓存感知 LoRA 路由、UCCL 网络、缩放到零。经济效益：多个客户披露的内部汇总显示，在恒定 SLA 下从共置服务切换到 Dynamo 分离式时，200 万美元级推理支出可节省 30-40%（即每年 60-80 万美元）；这一具体数字是内部综合值，并非单一已发布案例研究——用作数量级参考，而非引用来源。短提示词（<512 token，短输出）不足以证明传输成本合理。

**类型：** 学习
**编程语言：** Python（标准库，玩具分离式 vs 共置模拟器）
**前置知识：** Phase 17 · 04（vLLM 服务内部原理）、Phase 17 · 08（推理指标）
**预计时间：** 约 75 分钟

## 学习目标

- 解释为什么预填充和解码有不同的最优 GPU 分配，并量化共置下的浪费。
- 绘制分离式架构图：预填充池、解码池、通过 NIXL 的 KV 传输、路由器。
- 命名分离式架构不划算的条件（短提示词，短输出）。
- 区分 NVIDIA Dynamo（架设在上层）与 llm-d（Kubernetes 原生），并将每种匹配到适合的运营场景。

## 问题背景

你在 8 台 H100 上运行 Llama 3.3 70B。在混合工作负载（长提示词 + 短输出）下，GPU 在解码期间空闲，因为大部分计算花在了预填充上。在不同工作负载（短提示词 + 长输出）下情况相反。共置预填充 + 解码意味着你对两者都过度供给。

预算影响：20-40% 的 GPU 时间浪费在错误的资源上。你在购买 H100 计算能力来运行内存带宽受限的解码，或者购买 H100 HBM 带宽来运行计算密集型预填充。两者都是昂贵的浪费。

分离式架构将预填充和解码拆分到按各自瓶颈规格设置的独立池。KV 缓存通过高带宽互连从预填充池传输到解码池。

## 核心概念

### 瓶颈为何不同

**预填充** — 在一次前向传递中对整个输入提示词运行 Transformer。矩阵乘法占主导；计算密集型。H100 FP8 提供约 2000 TFLOPS 的有效吞吐量。批次效率好——一次前向处理许多 token。

**解码** — 每次迭代生成一个 token，读取完整权重。内存带宽受限。HBM3 提供约 3 TB/s。批次效率仅在高并发下才好——权重读取在批次中摊销。

共置两者：你购买对两者都优化的 GPU。H100 两者都好，但成本相同。在规模上，你希望预填充池使用 H100/计算密集型；解码池使用 H200/内存密集型，或搭配激进量化。

### 架构

```
              ┌──────────────┐
  请求 →      │    路由器    │ ───────────────────────┐
              └──────┬───────┘                        │
                     │                                │
                     ▼ (仅提示词)                     │
              ┌──────────────┐    KV 缓存    ┌────────▼─────┐
              │  预填充池    │ ─── NIXL ────►│   解码池     │
              │   (计算)    │                │   (内存)    │
              └──────────────┘                └──────┬───────┘
                                                     │ token
                                                     ▼
                                                   客户端
```

NIXL 是 NVIDIA 的节点间传输层。有 RDMA/InfiniBand 时使用，否则降级为 TCP。传输延迟是实际存在的——70B FP8 上 4K token 提示词的 KV 缓存通常 20-80ms。这就是为什么短提示词不足以证明分离式架构合理：传输成本超过节省。

### Dynamo vs llm-d

**NVIDIA Dynamo**（GTC 2025 发布，1.0 GA）：
- 作为编排器架设在 vLLM、SGLang、TRT-LLM 之上。
- Planner Profiler 测量工作负载，SLA Planner 自动配置预填充:解码比例。
- Rust 内核，Python 可扩展性。
- 吞吐量增益：NVIDIA 报告 GB200 NVL72 + Dynamo 上 DeepSeek-R1 MoE 在中等延迟区间 6 倍（developer.nvidia.com，2025-06）；完整 Blackwell + Dynamo + DeepSeek-R1 技术栈"高达 30 倍"的社区报告缺少单一主要来源，应视为方向性说法。
- GB300 NVL72 + Dynamo：相比 Hopper 最高 50 倍 MoE 吞吐量（Dynamo 产品页，日期未注明）。

**llm-d**（Red Hat + AWS，Kubernetes 原生）：
- 预填充/解码/路由器作为独立 Kubernetes Service。
- 每角色 HPA，使用队列深度（预填充）/KV 利用率（解码）信号。
- `topologyConstraint packDomain: rack` 将预填充+解码集群打包在同一机架以获得高带宽 KV 传输。
- llm-d 0.5（2026 年）：分层 KV 卸载、缓存感知 LoRA 路由、UCCL 网络、缩放到零。

如果你想要托管的上层编排器，使用 Dynamo。如果你想要 Kubernetes 原生原语并承诺 CNCF 生态系统，使用 llm-d。

### 经济效益

内部综合数字（非单一已发布案例研究——数量级参考）：

- 共置服务每年推理支出 200 万美元。
- 切换到带 Dynamo 的分离式。
- 相同请求量，相同 P99 延迟 SLA。
- 报告节省：每年 60-80 万美元（降低 30-40%）。
- 无需新硬件。

我们从多个客户披露而非单一可引用案例研究中综合了这一数字；最接近的已发布数据点是 Baseten 使用 Dynamo KV 路由实现首 token 延迟快 2 倍/吞吐量提升 61%（baseten.co，2025-10），以及 VAST + CoreWeave 预测在 40-60% KV 命中率下每美元 token 数提升 60-130%（vastdata.com，2025-12）。节省来自对每个池的合理规格设置；预填充密集型工作负载（带 8K+ 前缀的 RAG）受益更多。

### 何时不该分离

- 提示词 < 512 token 且输出 < 200 token：传输成本超过收益。
- 小型集群（< 4 台 GPU）：池多样性不足。
- 团队无法操作两个带按角色扩缩容的 GPU 池：Dynamo 有帮助，但不是轻而易举。
- 无 RDMA 网络：TCP 传输成本更重。

### 路由器与 Phase 17 · 11 集成

分离式路由器具有 KV 缓存感知（Phase 17 · 11）。请求落在持有其前缀的解码池上——如果没有匹配，流经预填充 → 解码。命中率和分离式架构相互叠加——缓存感知路由器决定是否甚至需要新的预填充。

### Blackwell 上的 MoE 才是真实数字的来源

GB300 NVL72 + Dynamo 显示相比 Hopper 基线 50 倍 MoE 吞吐量。MoE 专家路由在预填充上计算密集，但在解码上内存密集（专家缓存），所以分离式是双重胜出。2026 年前沿模型服务以 MoE 为主（DeepSeek-V3、未来 GPT-5 变体）。

### 需要记住的数字

基准数字在变化——NVIDIA 和推理技术栈每季度发布更新结果。引用前重新核实。

- GB200 NVL72 + Dynamo 上 DeepSeek-R1：中等延迟区间约 6 倍吞吐量（developer.nvidia.com，2025-06）；完整 Blackwell + Dynamo 技术栈上"高达 30 倍"的社区说法是没有单一主要来源的方向性汇总。
- GB300 NVL72 + Dynamo：相比 Hopper 最高 50 倍 MoE 吞吐量（developer.nvidia.com，日期未注明）。
- 节省参考（内部综合，非单一案例研究）：恒定 SLA 下每年 200 万美元支出节省 60-80 万美元。
- 分离式阈值：提示词 > 512 token + 输出 > 200 token。
- 通过 NIXL 的 KV 传输：70B FP8 上 4K 提示词 KV 为 20-80ms。

## 动手实践

`code/main.py` 模拟共置 vs 分离式服务。报告吞吐量、每请求成本和提示词长度交叉点。

## 产出技能

本课产出 `outputs/skill-disaggregation-decider.md`。给定工作负载和集群，决定是否分离。

## 练习

1. 运行 `code/main.py`。在多长提示词时，分离式优于共置？
2. 为 P99 前缀长度 8K、输出 300 的 RAG 服务设计预填充池和解码池。
3. Dynamo vs llm-d：为纯 Kubernetes 团队（无 Python 运行时偏好）选择一个。
4. 计算 KV 传输成本：70B FP8 上 4K 预填充 = 约 500MB KV。RDMA 100 GB/s 时，传输 = 5ms。TCP 10 GB/s = 50ms。哪个对你的 SLA 重要？
5. MoE 专家路由改变了 KV 访问模式。分离式架构如何处理每个 token 激活不同专家的 MoE？

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| 分离式服务（Disaggregated serving） | "拆分预填充/解码" | 每个阶段独立的 GPU 池 |
| NIXL | "NVIDIA 传输层" | Dynamo 的节点间 KV 传输（RDMA/TCP） |
| NVIDIA Dynamo | "编排器" | vLLM/SGLang/TRT-LLM 之上的技术栈协调器 |
| llm-d | "Kubernetes 原生" | Red Hat + AWS K8s 分离式技术栈 |
| Planner Profiler | "Dynamo 自动配置" | 测量工作负载，配置池比例 |
| SLA Planner | "Dynamo 策略" | 自动调整预填充:解码比例以满足 SLO |
| `packDomain: rack` | "llm-d 拓扑" | 将预填充+解码打包在同一机架以获得快速 KV |
| UCCL | "统一集体" | llm-d 0.5 用于缩放到零的网络层 |
| MoE 专家路由（MoE expert routing） | "每 token 专家" | DeepSeek-V3 模式；分离式有帮助 |

## 延伸阅读

- [NVIDIA——介绍 Dynamo](https://developer.nvidia.com/blog/introducing-nvidia-dynamo-a-low-latency-distributed-inference-framework-for-scaling-reasoning-ai-models/)
- [NVIDIA——Kubernetes 上的分离式 LLM 推理](https://developer.nvidia.com/blog/deploying-disaggregated-llm-inference-workloads-on-kubernetes/)
- [TensorRT-LLM 分离式服务博客](https://nvidia.github.io/TensorRT-LLM/blogs/tech_blog/blog5_Disaggregated_Serving_in_TensorRT-LLM.html)
- [llm-d GitHub](https://github.com/llm-d/llm-d)
- [llm-d 0.5 发布说明](https://github.com/llm-d/llm-d/releases)
