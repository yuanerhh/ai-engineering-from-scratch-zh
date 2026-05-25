# T5、BART——编码器-解码器模型

> 编码器理解。解码器生成。把它们重新组合，你就得到了一个专为输入 → 输出任务构建的模型：翻译、摘要、改写、转录。

**类型：** 学习
**语言：** Python
**前置条件：** Phase 7 · 05（完整 Transformer）、Phase 7 · 06（BERT）、Phase 7 · 07（GPT）
**时长：** 约 45 分钟

## 问题背景

仅解码器的 GPT 和仅编码器的 BERT 各自针对不同目标精简了 2017 年的架构。但许多任务天然是输入-输出形式的：

- 翻译：英语 → 法语。
- 摘要：5000 token 文章 → 200 token 摘要。
- 语音识别：音频 token → 文本 token。
- 结构化提取：散文 → JSON。

对于这些任务，编码器-解码器是最自然的架构。编码器生成源序列的密集表示。解码器生成输出，每一步都交叉关注该表示。训练是输出侧的移位一位。与 GPT 相同的损失，只是以编码器输出为条件。

两篇论文定义了现代范式：

1. **T5**（Raffel et al. 2019）。"文本到文本迁移 Transformer"。每个 NLP 任务重新定义为文本输入、文本输出。单一架构、单一词汇表、单一损失。在掩码跨度预测上预训练（在输入中破坏跨度，在输出中解码它们）。
2. **BART**（Lewis et al. 2019）。"双向自回归 Transformer"。去噪自编码器：以多种方式破坏输入（打乱、掩码、删除、旋转），让解码器重建原始内容。

2026 年，编码器-解码器格式在输入结构重要的地方继续存在：

- Whisper（语音 → 文本）。
- Google 的翻译栈。
- 一些具有明确上下文和编辑结构的代码补全/修复模型。
- Flan-T5 及其变体，用于结构化推理任务。

仅解码器赢得了聚光灯，但编码器-解码器从未消失。

## 核心概念

![带交叉注意力的编码器-解码器](../assets/encoder-decoder.svg)

### 前向循环

```
源 token ─▶ 编码器 ─▶ (N_src, d_model)  ──┐
                                           │
目标 token ─▶ 解码器块                      │
              ├─▶ 掩码自注意力              │
              ├─▶ 交叉注意力 ◀─────────────┘
              └─▶ FFN
             ↓
           下一个 token logit
```

关键在于，编码器对每个输入只运行一次。解码器自回归运行，但每一步交叉关注*相同的*编码器输出。缓存编码器输出对长输入是免费的加速。

### T5 预训练——跨度破坏

随机选择输入的跨度（平均长度 3 个 token，共 15%）。将每个跨度替换为唯一的哨兵 token：`<extra_id_0>`、`<extra_id_1>` 等。解码器只输出带哨兵前缀的破坏跨度：

```
源：   The quick <extra_id_0> fox jumps <extra_id_1> dog
目标： <extra_id_0> brown <extra_id_1> over the lazy
```

比预测整个序列更便宜的信号。在 T5 论文的消融研究中与 MLM（BERT）和前缀-LM（UniLM）相比具有竞争力。

### BART 预训练——多噪声去噪

BART 尝试了五种噪声函数：

1. Token 掩码。
2. Token 删除。
3. 文本填充（掩码一个跨度，解码器插入正确长度）。
4. 句子排列。
5. 文档旋转。

组合文本填充 + 句子排列产生了最好的下游数字。解码器始终重建原始序列。BART 的输出是完整序列，而非只有破坏的跨度——所以预训练算力比 T5 更高。

### 推理

与 GPT 相同的自回归生成。贪婪/束搜索/top-p 采样均适用。束搜索（宽度 4-5）是翻译和摘要的标准，因为输出分布比对话更窄。

### 2026 年各变体的选择时机

| 任务 | 编码器-解码器？ | 原因 |
|------|--------------|------|
| 翻译 | 通常是 | 清晰的源序列；固定输出分布；束搜索有效 |
| 语音转文本 | 是（Whisper） | 输入模态与输出不同；编码器塑造音频特征 |
| 对话/推理 | 否，仅解码器 | 没有持久的"输入"——对话即序列 |
| 代码补全 | 通常否 | 长上下文的仅解码器更优；Qwen 2.5 Coder 等代码模型是仅解码器 |
| 摘要 | 两者均可 | BART、PEGASUS 曾优于早期仅解码器基准；现代仅解码器 LLM 与之持平 |
| 结构化提取 | 两者均可 | T5 简洁，因为"文本→文本"能吸收任何输出格式 |

约 2022 年以来的趋势：仅解码器接管了编码器-解码器曾拥有的任务，原因是：(a) 指令微调的仅解码器 LLM 通过提示泛化到任何任务；(b) 一个架构比两个更容易扩展；(c) RLHF 假设一个解码器。编码器-解码器在输入模态不同（语音、图像）或束搜索质量重要的地方坚守阵地。

## 动手实现

见 `code/main.py`。我们为玩具语料实现 T5 风格的跨度破坏——本课最有用的单一部分，因为它出现在此后的每个编码器-解码器预训练方案中。

### 步骤一：跨度破坏

```python
def corrupt_spans(tokens, mask_rate=0.15, mean_span=3.0, rng=None):
    """选择总计约 mask_rate token 的跨度。返回 (corrupted_input, target)。"""
    n = len(tokens)
    n_mask = max(1, int(n * mask_rate))
    n_spans = max(1, int(round(n_mask / mean_span)))
    ...
```

目标格式采用 T5 约定：`<sent0> span0 <sent1> span1 ...`。破坏的输入在跨度位置将未改变的 token 与哨兵 token 交错。

### 步骤二：验证往返

给定破坏的输入和目标，重建原始句子。如果你的破坏是可逆的，前向传播就是定义良好的。这是健全性检查——实际训练从不这样做，但测试成本低廉，能捕获跨度记录中的偏移一位错误。

### 步骤三：BART 噪声

五个函数：`token_mask`、`token_delete`、`text_infill`、`sentence_permute`、`document_rotate`。组合其中两个并展示结果。

## 生产使用

HuggingFace 参考代码：

```python
from transformers import T5ForConditionalGeneration, T5Tokenizer
tok = T5Tokenizer.from_pretrained("google/flan-t5-base")
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base")

inputs = tok("translate English to French: Attention is all you need.", return_tensors="pt")
out = model.generate(**inputs, max_new_tokens=32)
print(tok.decode(out[0], skip_special_tokens=True))
```

T5 技巧：任务名称进入输入文本。同一个模型处理数十种任务，因为每个任务都是文本输入、文本输出。2026 年这个模式已被指令微调的仅解码器模型推广，但 T5 首先将其编成体系。

## 上手实践

见 `outputs/skill-seq2seq-picker.md`。该技能根据输入-输出结构、延迟和质量目标，为新任务在编码器-解码器与仅解码器之间做出选择。

## 练习

1. **简单。** 运行 `code/main.py`，对 30 个 token 的句子应用跨度破坏，验证将非哨兵源 token 与解码目标跨度拼接能重现原始内容。
2. **中等。** 实现 BART 的 `text_infill` 噪声：将随机跨度替换为单个 `<mask>` token，解码器必须推断出正确的跨度长度和内容。展示一个示例。
3. **困难。** 在微型英语→猪拉丁语语料（200 对）上微调 `flan-t5-small`。测量 50 对保留集上的 BLEU。与使用相同算力在同一数据上微调 `Llama-3.2-1B` 进行对比。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| 编码器-解码器（Encoder-decoder） | "Seq2seq Transformer" | 两个栈：用于输入的双向编码器，用于输出的带交叉注意力的因果解码器。 |
| 交叉注意力（Cross-attention） | "源与目标对话的地方" | 解码器的 Q × 编码器的 K/V。编码器信息进入解码器的唯一通道。 |
| 跨度破坏（Span corruption） | "T5 的预训练技巧" | 将随机跨度替换为哨兵 token；解码器输出这些跨度。 |
| 去噪目标（Denoising objective） | "BART 的游戏" | 对输入应用噪声函数，训练解码器重建干净序列。 |
| 哨兵 token（Sentinel token） | "`<extra_id_N>` 占位符" | 在源中标记被破坏跨度、在目标中重新标记的特殊 token。 |
| Flan | "指令微调的 T5" | 在 1800+ 个任务上微调的 T5；使编码器-解码器在指令遵循上具有竞争力。 |
| 束搜索（Beam search） | "解码策略" | 每步保留前 k 个部分序列；翻译/摘要的标准方法。 |
| 教师强制（Teacher forcing） | "训练时的输入" | 训练时向解码器输入真实的前一个输出 token，而非采样的 token。 |

## 延伸阅读

- [Raffel et al. (2019). Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](https://arxiv.org/abs/1910.10683) — T5
- [Lewis et al. (2019). BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension](https://arxiv.org/abs/1910.13461) — BART
- [Chung et al. (2022). Scaling Instruction-Finetuned Language Models](https://arxiv.org/abs/2210.11416) — Flan-T5
- [Radford et al. (2022). Robust Speech Recognition via Large-Scale Weak Supervision](https://arxiv.org/abs/2212.04356) — Whisper，2026 年规范的编码器-解码器
- [HuggingFace `modeling_t5.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/t5/modeling_t5.py) — 参考实现
