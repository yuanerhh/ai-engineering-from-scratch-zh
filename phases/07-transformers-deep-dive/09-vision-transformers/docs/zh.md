# 视觉 Transformer（ViT）

> 图像是图块的网格。句子是 token 的网格。同一个 Transformer 能吃掉两者。

**类型：** 构建
**语言：** Python
**前置条件：** Phase 7 · 05（完整 Transformer）、Phase 4 · 03（CNN）、Phase 4 · 14（视觉 Transformer 简介）
**时长：** 约 45 分钟

## 问题背景

2020 年之前，计算机视觉意味着卷积。ImageNet、COCO 和检测基准上的每个最先进方案都使用 CNN 骨干。Transformer 是用于语言的。

Dosovitskiy et al.（2020）——"一张图像值 16×16 个词"——表明你可以完全去掉卷积。将图像切成固定大小的图块，对每个图块进行线性投影成嵌入，将序列输入到普通 Transformer 编码器。在足够的规模下（ImageNet-21k 预训练或更大），ViT 能匹配或超越基于 ResNet 的模型。

ViT 开启了 2026 年更广泛的模式：一种架构，多种模态。Whisper 对音频分词。ViT 对图像分词。机器人学的动作 token。视频的像素 token。Transformer 不在乎——给它一个序列，它就能学习。

到 2026 年，ViT 及其后代（DeiT、Swin、DINOv2、ViT-22B、SAM 3）主导了大部分视觉任务。CNN 在边缘设备和延迟敏感任务上仍然胜出。其他所有任务的栈中都有 ViT。

## 核心概念

![图像 → 图块 → token → Transformer](../assets/vit.svg)

### 步骤一——图块化（Patchify）

将 `H × W × C` 图像分割成 `N × (P·P·C)` 的扁平图块序列。典型设置：`224 × 224` 图像，`16 × 16` 图块 → 196 个图块，每个 768 维。

```
图像 (224, 224, 3) → 14 × 14 的 16x16x3 图块网格 → 196 个长度为 768 的向量
```

图块大小是调节杠杆。图块越小 = token 越多、分辨率越高、注意力代价呈二次方增长。图块越大 = 越粗糙、越便宜。

### 步骤二——线性嵌入

单个可学习矩阵将每个扁平图块投影到 `d_model`。等价于核大小为 `P`、步长为 `P` 的卷积。在 PyTorch 中字面上就是 `nn.Conv2d(C, d_model, kernel_size=P, stride=P)`——两行实现。

### 步骤三——添加 `[CLS]` token，加位置嵌入

- 前置一个可学习的 `[CLS]` token。其最终隐藏状态是用于分类的图像表示。
- 添加可学习位置嵌入（ViT 原始版）或正弦二维（后续变体）。
- 2024 年以来，RoPE 扩展到二维位置，有时不使用显式嵌入。

### 步骤四——标准 Transformer 编码器

堆叠 L 个 `LayerNorm → 自注意力 → + → LayerNorm → MLP → +` 块。与 BERT 完全相同。无视觉专用层。这是论文的教学亮点。

### 步骤五——头

分类：取 `[CLS]` 隐藏状态 → 线性 → softmax。对于 DINOv2 或 SAM，丢弃 `[CLS]`，直接使用图块嵌入。

### 重要变体

| 模型 | 年份 | 变化 |
|------|------|------|
| ViT | 2020 | 原始版本。固定图块大小，完整全局注意力。 |
| DeiT | 2021 | 蒸馏；仅在 ImageNet-1k 上可训练。 |
| Swin | 2021 | 带移位窗口的分层结构。固定亚二次方代价。 |
| DINOv2 | 2023 | 自监督（无标注）。最佳通用视觉特征。 |
| ViT-22B | 2023 | 220 亿参数；扩展律适用。 |
| SigLIP | 2023 | ViT + 语言配对，sigmoid 对比损失。 |
| SAM 3 | 2025 | 分割一切；ViT-Large + 可提示掩码解码器。 |

### 为什么需要一段时间

ViT 需要*大量*数据才能匹配 CNN，因为它没有 CNN 的归纳偏置（平移不变性、局部性）。没有 1 亿以上的标注图像或强大的自监督预训练，在相同算力下 CNN 仍然胜出。DeiT 在 2021 年通过蒸馏技巧修复了这一问题；DINOv2 在 2023 年通过自监督永久解决了它。

## 动手实现

见 `code/main.py`。纯标准库的图块化 + 线性嵌入 + 健全性检查。无需训练——任何实际规模的 ViT 都需要 PyTorch 和数小时 GPU 时间。

### 步骤一：假图像

24 × 24 RGB 图像，表示为 `(R, G, B)` 元组行的列表。我们使用 6×6 图块 → 16 个图块，每个 108 维嵌入向量。

### 步骤二：图块化

```python
def patchify(image, P):
    H = len(image)
    W = len(image[0])
    patches = []
    for i in range(0, H, P):
        for j in range(0, W, P):
            patch = []
            for di in range(P):
                for dj in range(P):
                    patch.extend(image[i + di][j + dj])
            patches.append(patch)
    return patches
```

光栅顺序：跨网格的行主序。每个 ViT 都使用这种排序。

### 步骤三：线性嵌入

将每个扁平图块乘以随机的 `(patch_flat_size, d_model)` 矩阵。在前置 `[CLS]` 后验证输出形状为 `(N_patches + 1, d_model)`。

### 步骤四：统计实际 ViT 的参数量

打印 ViT-Base 的参数量：12 层、12 头、d=768、patch=16。与 ResNet-50（约 2500 万）对比。ViT-Base 约 8600 万。ViT-Large 约 3.07 亿。ViT-Huge 约 6.32 亿。

## 生产使用

```python
from transformers import ViTImageProcessor, ViTModel
import torch
from PIL import Image

processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")

img = Image.open("cat.jpg")
inputs = processor(img, return_tensors="pt")
out = model(**inputs).last_hidden_state   # (1, 197, 768)：[CLS] + 196 个图块
cls_emb = out[:, 0]                       # 图像表示
```

**DINOv2 嵌入是 2026 年图像特征的默认方案。** 冻结骨干，训练微型头。适用于分类、检索、检测、字幕生成。Meta 的 DINOv2 检查点在每个非文本视觉任务上都优于 CLIP。

**图块大小选择。** 小模型使用 16×16（ViT-B/16）。密集预测（分割）使用 8×8 或 14×14（SAM、DINOv2）。超大模型使用 14×14。

## 上手实践

见 `outputs/skill-vit-configurator.md`。该技能根据数据集大小、分辨率和算力预算，为新视觉任务选择 ViT 变体和图块大小。

## 练习

1. **简单。** 运行 `code/main.py`。验证图块数量等于 `(H/P) * (W/P)`，扁平图块维度等于 `P*P*C`。
2. **中等。** 实现二维正弦位置嵌入——每个图块的 `row` 和 `col` 各一个独立的正弦编码，拼接在一起。在微型 PyTorch ViT 中使用它们，在 CIFAR-10 上与可学习位置嵌入比较准确率。
3. **困难。** 构建 3 层 ViT（PyTorch），在 1000 张 MNIST 图像上用 4×4 图块训练。测量测试准确率。现在在同样的 1000 张图像上添加 DINOv2 预训练（简化版：只训练编码器预测被遮蔽图块的嵌入）。准确率有提升吗？

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| 图块（Patch） | "视觉 Transformer 的 token" | 图像 `P × P × C` 区域的像素值扁平向量。 |
| 图块化（Patchify） | "切分 + 展平" | 将图像切成不重叠的图块，将每个展平为向量。 |
| `[CLS]` token | "图像摘要" | 前置的可学习 token；其最终嵌入是图像表示。 |
| 归纳偏置（Inductive bias） | "模型的假设" | ViT 比 CNN 的先验更少；需要更多数据来弥补差距。 |
| DINOv2 | "自监督 ViT" | 使用图像增强 + 动量教师无标注训练。2026 年最佳通用图像特征。 |
| SigLIP | "CLIP 的继任者" | 用 sigmoid 对比损失训练的 ViT + 文本编码器；在相同算力下优于 CLIP。 |
| Swin | "窗口化 ViT" | 带局部注意力 + 移位窗口的分层 ViT；亚二次方复杂度。 |
| 寄存器 token（Register tokens） | "2023 技巧" | 几个额外的可学习 token，吸收注意力汇聚（attention sink）；改善 DINOv2 特征。 |

## 延伸阅读

- [Dosovitskiy et al. (2020). An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929) — ViT 论文
- [Touvron et al. (2021). Training data-efficient image transformers & distillation through attention](https://arxiv.org/abs/2012.12877) — DeiT
- [Liu et al. (2021). Swin Transformer: Hierarchical Vision Transformer using Shifted Windows](https://arxiv.org/abs/2103.14030) — Swin
- [Oquab et al. (2023). DINOv2: Learning Robust Visual Features without Supervision](https://arxiv.org/abs/2304.07193) — DINOv2
- [Darcet et al. (2023). Vision Transformers Need Registers](https://arxiv.org/abs/2309.16588) — DINOv2 的寄存器 token 修复
