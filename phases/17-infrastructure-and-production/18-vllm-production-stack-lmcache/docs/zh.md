# vLLM 生产技术栈与 LMCache KV 卸载

> vLLM 生产技术栈是 Kubernetes 的参考部署——路由器、引擎和可观测性组件连接在一起。LMCache 是 KV 卸载层，将 KV 缓存从 GPU 显存中提取出来，并在查询和引擎之间复用（CPU DRAM，再到磁盘/Ceph）。vLLM 0.11.0 KV 卸载连接器（2026 年 1 月）通过 Connector API（v0.9.0+）使其支持异步操作且可插拔。卸载延迟对用户不可见。即使没有共享前缀，LMCache 也有价值——当 GPU 的 KV 槽耗尽时，被抢占的请求可以从 CPU 恢复，而无需重新计算预填充。已发布的 16x H100（80GB HBM）跨 4 台 a3-highgpu-4g 基准测试：当 KV 缓存超过 HBM 时，原生 CPU 卸载和 LMCache 都能大幅提升吞吐量；在低 KV 占用时，所有配置与基线相当，仅有较小的额外开销。

**类型：** 学习
**编程语言：** Python（标准库，玩具 KV 溢出模拟器）
**前置知识：** Phase 17 · 04（vLLM 服务内部原理）、Phase 17 · 06（SGLang/RadixAttention）
**预计时间：** 约 60 分钟

## 学习目标

- 绘制 vLLM 生产技术栈层次图：路由器、引擎、KV 卸载、可观测性。
- 解释 KV 卸载 Connector API（v0.9.0+）以及 0.11.0 异步路径如何隐藏卸载延迟。
- 量化 LMCache CPU-DRAM 何时有帮助（KV > HBM）vs 何时增加开销（KV 足以放入 HBM）。
- 根据部署约束，在原生 vLLM CPU 卸载和 LMCache 连接器之间作出选择。

## 问题背景

你的 vLLM 服务在并发上升时 GPU HBM 达到 100%，并频繁出现抢占事件。请求被驱逐、重新排队，同一个 2K token 提示词在一分钟内被重新预填充四次。GPU 算力浪费在冗余预填充上；实际吞吐量（goodput）远低于原始吞吐量。

增加 GPU 成本线性增长。HBM 无法扩展。但 CPU DRAM 便宜——一个插槽有 512GB+ 内存，延迟比 HBM 高几个数量级，但对"临时热" KV 缓存来说完全够用。

LMCache 将 KV 缓存提取到 CPU DRAM，使被抢占的请求快速恢复，并让跨引擎的重复前缀共享缓存，无需每个引擎单独重新预填充。

## 核心概念

### vLLM 生产技术栈

`github.com/vllm-project/production-stack` 是 Kubernetes 的参考部署：

- **路由器** — 缓存感知（Phase 17 · 11）。消费 KV 事件。
- **引擎** — vLLM 工作进程。每 GPU 一个或每 TP/PP 组一个。
- **KV 缓存卸载** — LMCache 部署或原生连接器。
- **可观测性** — Prometheus 采集、Grafana 仪表板、OTel 追踪。
- **控制平面** — 服务发现、配置、滚动更新。

以 Helm chart + operator 形式发布。

### KV 卸载 Connector API（v0.9.0+）

vLLM 0.9.0 引入了可插拔 KV 缓存后端的 Connector API。引擎将块卸载到连接器；连接器存储它们（RAM、磁盘、对象存储、LMCache）。请求需要某个块时，连接器将其加载回来。

vLLM 0.11.0（2026 年 1 月）增加了异步卸载路径——卸载可以在后台进行，引擎在常规情况下不会阻塞。端到端延迟和吞吐量仍取决于工作负载形状、KV 缓存命中率和系统压力；vLLM 自己的说明指出，自定义内核卸载在低命中率时可能降低吞吐量，且异步调度与投机性解码之间存在已知的交互问题。

### 原生 CPU 卸载 vs LMCache

**原生 vLLM CPU 卸载**：引擎本地。将 KV 块存储在主机 RAM 中。实现简单，零网络跳转。不跨引擎。

**LMCache 连接器**：集群规模。将块存储在共享 LMCache 服务器中（CPU DRAM + Ceph/S3 层）。任何引擎都可以访问这些块。已发布 16x H100 基准测试。

当单个引擎有 HBM 压力时选择原生。当多个引擎共享前缀时选择 LMCache（带通用系统提示词的 RAG、带共享模板的多租户）。

### 基准行为

16x H100（80GB HBM）跨 4 台 a3-highgpu-4g 的测试结果：

- 低 KV 占用（短提示词、低并发）：所有配置与基线相当，LMCache 增加约 3-5% 开销。
- 中等占用：LMCache 开始在跨引擎前缀复用上发挥作用。
- KV 超过 HBM：原生 CPU 卸载和 LMCache 都能大幅提升吞吐量；LMCache 提升更大，因为有跨引擎共享。

### LMCache 发挥决定性作用的场景

- 系统提示词跨租户共享的多租户服务。
- 文档块跨查询重复出现的 RAG。
- 同一基础模型上的微调变体（LoRA）：基础模型 KV 复用减少冗余工作。
- 抢占频繁的工作负载：从 CPU 恢复比重新预填充便宜。

### 不该启用的场景

- HBM 压力小——付出开销却得不到收益。
- 短上下文（<1K token）——传输时间 > 重新预填充时间。
- 单租户单提示词工作负载——没有可复用的内容。

### 与分离式服务的集成

Phase 17 · 17 的分离式服务 + LMCache 叠加效果：KV 从预填充池传输到解码池，若未被使用则落入 LMCache；后续查询从 LMCache 拉取。Phase 17 · 11 的缓存感知路由器可以路由到本地缓存或 LMCache 共享缓存匹配的引擎。

### 需要记住的数字

- vLLM 0.9.0：Connector API 发布。
- vLLM 0.11.0（2026 年 1 月）：异步卸载路径；端到端延迟影响取决于工作负载、KV 命中率和系统压力（不是绝对保证）。
- 16x H100 基准：KV 占用超过 HBM 时，LMCache 有帮助。
- 低 HBM 压力：增加 3-5% 开销而无收益。

## 动手实践

`code/main.py` 模拟有无 LMCache 的抢占密集型工作负载。报告避免的重新预填充次数、吞吐量增益和盈亏平衡 HBM 利用率。

## 产出技能

本课产出 `outputs/skill-vllm-stack-decider.md`。给定工作负载形状和 vLLM 部署，决定使用原生卸载、LMCache 还是不启用。

## 练习

1. 运行 `code/main.py`。在什么 HBM 利用率下，LMCache 开始有回报？
2. 一个租户在 200 个查询/小时中共享一个 6K token 系统提示词。计算每个租户的预期 LMCache 节省。
3. LMCache 服务器是单点故障。设计高可用策略（副本、回退到原生）。
4. LMCache 存储到机械硬盘上的 Ceph。对于 70B FP8 下的 4K token KV（500MB），读取时间 vs 重新预填充时间是多少？
5. 论证 vLLM 0.11.0 异步路径是否"免费"——开销隐藏在哪里？

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| 生产技术栈（Production-stack） | "参考部署" | vLLM 的 Kubernetes Helm chart + operator |
| Connector API | "KV 后端接口" | vLLM 0.9.0+ 可插拔 KV 存储接口 |
| 原生 CPU 卸载（Native CPU offload） | "引擎本地溢出" | 在同一引擎主机 RAM 中存储 KV |
| LMCache | "集群 KV 缓存" | CPU DRAM + 磁盘上的跨引擎 KV 缓存服务器 |
| 0.11.0 异步（0.11.0 async） | "非阻塞卸载" | 卸载隐藏在引擎流程后台 |
| 抢占（Preemption） | "驱逐腾空间" | HBM 满时的 KV 缓存置换 |
| 前缀复用（Prefix reuse） | "相同系统提示词" | 多个查询共享开头部分；缓存命中 |
| Ceph 层（Ceph tier） | "磁盘层" | 缓存层次结构中 DRAM 下方的持久化存储 |

## 延伸阅读

- [vLLM 博客——KV 卸载连接器（2026 年 1 月）](https://blog.vllm.ai/2026/01/08/kv-offloading-connector.html)
- [vLLM 生产技术栈 GitHub](https://github.com/vllm-project/production-stack) — Helm chart + operator
- [面向企业级 LLM 推理的 LMCache（arXiv:2510.09665）](https://arxiv.org/html/2510.09665v2)
- [LMCache GitHub](https://github.com/LMCache/LMCache) — 连接器实现
- [vLLM 0.11.0 发布说明](https://github.com/vllm-project/vllm/releases) — 异步路径详情
