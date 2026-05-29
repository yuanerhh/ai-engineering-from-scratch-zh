# 边缘推理——Apple 神经引擎、高通 Hexagon、WebGPU/WebLLM、Jetson

> 边缘端的核心约束是内存带宽，而非计算。移动 DRAM 约 50-90 GB/s；数据中心 HBM3 超过 2-3 TB/s——差距 30-50 倍。解码是内存带宽受限的，因此这一差距具有决定性。2026 年的格局分为四个方向。Apple M4/A18 神经引擎峰值 38 TOPS，采用统一内存（无 CPU↔NPU 复制）。高通骁龙 X Elite/8 Gen 4 Hexagon 达到 45 TOPS。WebGPU + WebLLM 在 M3 Max 上以约 41 token/秒运行 Llama 3.1 8B（Q4）（约为原生速度的 70-80%）；17.6k GitHub 星标，OpenAI 兼容 API，约 70-75% 移动端覆盖。NVIDIA Jetson Orin Nano Super（8GB）适合 Llama 3.2 3B/Phi-3；AGX Orin 通过 vLLM 以约 40 token/秒运行 gpt-oss-20b；Jetson T4000（JetPack 7.1）是 AGX Orin 的 2 倍。TensorRT Edge-LLM 支持 EAGLE-3、NVFP4、分块预填充——博世、ThunderSoft、联发科在 CES 2026 上展示。

**类型：** 学习
**编程语言：** Python（标准库，玩具带宽受限解码模拟器）
**前置知识：** Phase 17 · 04（vLLM 服务内部原理）、Phase 17 · 09（生产量化）
**预计时间：** 约 60 分钟

## 学习目标

- 解释为什么移动 LLM 推理是内存带宽受限的，计算是次要的。
- 枚举四个边缘目标（Apple ANE、高通 Hexagon、WebGPU/WebLLM、NVIDIA Jetson），并将每个与使用场景对应。
- 命名 2026 年 WebGPU 覆盖差距（Firefox Android 追赶中）和 Safari iOS 26 落地情况。
- 为每个目标选择量化格式（ANE 的 Core ML INT4 + FP16，Hexagon 的 QNN INT8/INT4，浏览器的 WebGPU Q4，Jetson Thor 的 NVFP4）。

## 问题背景

客户想要设备端聊天机器人：语音优先、默认隐私、离线工作。在 MacBook Pro M3 Max 上，Llama 3.1 8B Q4 运行约 55 token/秒——可以。在 iPhone 16 Pro 上，同款模型运行 3 token/秒——不行。在带骁龙 8 Gen 3 的中端安卓设备上，7 token/秒。在 Chrome Android v121+ 的浏览器中通过 WebGPU，根据设备不同 4-8 token/秒。

吞吐量差异不是移植问题，而是带宽差距乘以量化格式乘以 NPU 是否可从用户空间访问。2026 年的边缘推理是四个不同的问题，有四个不同的解决方案。

## 核心概念

### 带宽是真正的上限

解码为每个 token 读取完整的权重集。一个 Q4 格式的 7B 模型是 3.5GB。以 50 GB/s 读取 3.5GB 需要 70ms——理论上限约 14 token/秒。以 90 GB/s（高端移动 DRAM）上限移至约 25 token/秒。在这个数字以下，再多计算也没用。

数据中心 HBM3 以 3 TB/s 在 1.2ms 内读完同样的 3.5GB——上限是 830 token/秒。同款模型，同款权重，不同的内存子系统。

### Apple 神经引擎（M4 / A18）

- 最高 38 TOPS。统一内存（CPU 和 ANE 共享同一内存池）——无复制开销。
- 通过 Core ML + `.mlmodel` 编译模型访问，或通过 PyTorch 的 Metal Performance Shaders（MPS）访问。
- llama.cpp Metal 后端使用 MPS，而非直接使用 ANE；原生 ANE 需要 Core ML 转换。
- 2026 年 iOS 应用最佳实践路径：带 INT4 权重 + FP16 激活的 Core ML。

### 高通 Hexagon（骁龙 X Elite / 8 Gen 4）

- 最高 45 TOPS。与 CPU 和 GPU 集成在 SoC 中，但内存域独立。
- QNN（高通神经网络）SDK 和 AI Hub 提供从 PyTorch/ONNX 的转换。
- 聊天模板、Llama 3.2、Phi-3 均作为 AI Hub 的一等制品发布。

### Intel / AMD NPU（Lunar Lake、Ryzen AI 300）

- 40-50 TOPS。软件落后于 Apple/高通；OpenVINO 在改进中，但仍是小众。
- 最适合 Windows ARM Copilot 应用；AMD/Intel 桌面端的本地优先应用。

### WebGPU + WebLLM

- 通过 WebGPU 计算着色器在浏览器中运行模型；无需安装。
- M3 Max 上 Llama 3.1 8B Q4 约 41 token/秒——通过同一后端约为原生速度的 70-80%。
- WebLLM 17.6k GitHub 星标；OpenAI 兼容 JS API；Apache 2.0 开源。
- 2026 年覆盖：Chrome Android v121+，Safari iOS 26 GA，Firefox Android 仍在追赶。整体约 70-75% 移动端覆盖。

### NVIDIA Jetson 系列

- Orin Nano Super（8GB）：以良好 token/秒速度运行 Llama 3.2 3B、Phi-3。
- AGX Orin：通过 vLLM 以约 40 token/秒运行 gpt-oss-20b。
- Thor / T4000（JetPack 7.1）：AGX Orin 性能的 2 倍，支持 EAGLE-3 和 NVFP4。
- TensorRT Edge-LLM（2026 年）支持 EAGLE-3 投机解码、NVFP4 权重、分块预填充——数据中心优化移植到边缘端。

### 各目标的量化选择

| 目标 | 格式 | 说明 |
|------|------|------|
| Apple ANE | INT4 权重 + FP16 激活 | Core ML 转换路径 |
| 高通 Hexagon | QNN INT8 / INT4 | AI Hub 转换器 |
| WebGPU / WebLLM | Q4 MLC（q4f16_1） | 使用 `mlc_llm convert_weight` + 编译的 `.wasm`；不支持 GGUF |
| Jetson Orin Nano | Q4 GGUF 或 TRT-LLM INT4 | 内存带宽受限 |
| Jetson AGX / Thor | NVFP4 + FP8 KV | Edge-LLM 路径 |

### 边缘端的长上下文陷阱

Llama 3.1 的 128K 上下文是数据中心特性。在有 8GB RAM 的手机上，4GB 模型 + 32K token 的 2GB KV 缓存 + 操作系统开销 = OOM。边缘端部署将上下文保持在 4K-8K，除非接受激进的 KV 量化（Q4 KV）。

### 语音是杀手级应用

语音智能体对延迟敏感（首 token < 500ms）。本地推理完全消除了网络延迟。结合语音转文字（Whisper Turbo 变体可在边缘端运行），边缘推理成为生产质量的语音循环。

### 需要记住的数字

- Apple M4 / A18 ANE：38 TOPS。
- 高通 Hexagon 骁龙 X Elite：45 TOPS。
- WebLLM M3 Max：Llama 3.1 8B Q4 约 41 token/秒。
- AGX Orin：通过 vLLM 约 40 token/秒（gpt-oss-20b）。
- 数据中心-边缘带宽差距：30-50 倍。
- WebGPU 移动端覆盖：约 70-75%（Firefox Android 落后）。

## 动手实践

`code/main.py` 从带宽受限数学计算各边缘目标的理论解码吞吐量上限。与观察到的基准测试对比，突出带宽（而非计算）在哪里是瓶颈。

## 产出技能

本课产出 `outputs/skill-edge-target-picker.md`。给定平台（iOS/Android/浏览器/Jetson）、模型和延迟/内存预算，选择量化格式和转换管道。

## 练习

1. 运行 `code/main.py`。对于骁龙 8 Gen 3（约 77 GB/s 带宽）上 Q4 的 7B 模型，计算解码上限。与观察到的 6-8 token/秒相比——运行时效率如何？
2. Android 上的 WebGPU 需要 Chrome v121+。为旧版浏览器设计降级方案——通过相同 OpenAI 兼容 API 进行服务器端处理。
3. 你的 iOS 应用需要 4K 上下文流式传输。哪种模型/格式组合让你在 iPhone 16 上保持活跃内存低于 4GB？
4. Jetson AGX Orin 以 40 token/秒运行 gpt-oss-20b。Jetson Nano 只能运行 3B 模型。如果你的产品同时针对两者，如何统一推理栈？
5. 论证"WebLLM 在 2026 年是否生产就绪"。引用覆盖率、性能和 Firefox Android 差距。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| ANE | "Apple 神经引擎" | M 系列和 A 系列中的设备端 NPU；统一内存 |
| Hexagon | "高通 NPU" | 骁龙 NPU；通过 QNN SDK 访问 |
| WebGPU | "浏览器 GPU" | W3C 标准化浏览器 GPU API；2026 年 Chrome/Safari |
| WebLLM | "浏览器 LLM 运行时" | MLC-LLM 项目；Apache 2.0；OpenAI 兼容 JS |
| Jetson | "NVIDIA 边缘端" | Orin Nano / AGX / Thor / T4000 系列 |
| TRT Edge-LLM | "边缘端 TensorRT" | 2026 年 TensorRT-LLM 的边缘端移植；EAGLE-3 + NVFP4 |
| 统一内存（Unified memory） | "共享内存池" | CPU 和 NPU 见到相同 RAM；无复制开销 |
| 带宽受限（Bandwidth-bound） | "内存受限" | 解码由读取权重的字节/秒限制 |
| Core ML | "Apple 转换" | Apple 用于 ANE 原生模型的框架 |
| QNN | "高通技术栈" | 高通神经网络 SDK |

## 延伸阅读

- [设备端 LLM 现状 2026](https://v-chandra.github.io/on-device-llms/) — 格局和基准测试
- [NVIDIA Jetson 边缘 AI](https://developer.nvidia.com/blog/getting-started-with-edge-ai-on-nvidia-jetson-llms-vlms-and-foundation-models-for-robotics/) — Orin / AGX / Thor
- [NVIDIA TensorRT Edge-LLM](https://developer.nvidia.com/blog/accelerating-llm-and-vlm-inference-for-automotive-and-robotics-with-nvidia-tensorrt-edge-llm/) — 2026 年边缘端移植公告
- [WebLLM（arXiv:2412.15803）](https://arxiv.org/html/2412.15803v2) — 设计和基准测试
- [Apple Core ML](https://developer.apple.com/documentation/coreml) — ANE 原生转换
- [高通 AI Hub](https://aihub.qualcomm.com/) — Hexagon 预转换模型
