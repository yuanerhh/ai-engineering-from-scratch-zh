# 音频 Transformer——Whisper 架构

> 音频是频率随时间变化的图像。Whisper 是一个吃梅尔频谱图、说话的 ViT。

**类型：** 学习
**语言：** Python
**前置条件：** Phase 7 · 05（完整 Transformer）、Phase 7 · 08（编码器-解码器）、Phase 7 · 09（ViT）
**时长：** 约 45 分钟

## 问题背景

在 Whisper（OpenAI，Radford et al. 2022）之前，最先进的自动语音识别（ASR）意味着 wav2vec 2.0 和 HuBERT——自监督特征提取器加微调头。质量高，但数据流水线昂贵，域迁移脆弱。多语言语音识别需要每个语言族的独立模型。

Whisper 做了三个押注：

1. **在所有数据上训练。** 从互联网抓取的 68 万小时弱标注音频，涵盖 97 种语言。无干净学术语料库，无音素标注。
2. **多任务单一模型。** 单个解码器通过任务 token 联合训练于转录、翻译、语音活动检测（VAD）、语言识别和时间戳。
3. **标准编码器-解码器 Transformer。** 编码器消费对数梅尔频谱图。解码器自回归生成文本 token。无声码器，无 CTC，无 HMM。

结果：Whisper large-v3 在口音、噪声和零干净标注数据的语言上都具有鲁棒性。2026 年它是每个开源语音助手和大多数商业语音助手的默认语音前端。

## 核心概念

![Whisper 流水线：音频 → 梅尔 → 编码器 → 解码器 → 文本](../assets/whisper.svg)

### 步骤一——重采样 + 加窗

16 kHz 音频。裁剪/填充至 30 秒。计算对数梅尔频谱图：80 个梅尔频带，10 ms 步长 → 约 3000 帧 × 80 个特征。这是 Whisper 看到的"输入图像"。

### 步骤二——卷积茎（Convolutional stem）

两个核大小为 3、步长为 2 的 Conv1D 层将 3000 帧减少到 1500。在不增加大量参数的情况下将序列长度减半。

### 步骤三——编码器

在 1500 个时间步上的 24 层（large 版）Transformer 编码器。正弦位置编码、自注意力、GELU FFN。产生 1500 × 1280 的隐藏状态。

### 步骤四——解码器

24 层 Transformer 解码器。从一个 BPE 词汇表中自回归产生 token，该词汇表是 GPT-2 词汇表的超集，加上几个音频特定的特殊 token。

### 步骤五——任务 token

解码器提示以控制 token 开始，告诉模型该做什么：

```
<|startoftranscript|>  <|en|>  <|transcribe|>  <|0.00|>
```

或

```
<|startoftranscript|>  <|fr|>  <|translate|>   <|0.00|>
```

模型在此约定下训练。你通过前缀控制任务。这是 2026 年指令微调的等价物，但应用于语音。

### 步骤六——输出

带对数概率阈值的束搜索（宽度 5）。当 `<|notimestamps|>` token 不存在时，每 0.02 秒音频预测一次时间戳。

### Whisper 尺寸

| 模型 | 参数量 | 层数 | d_model | 头数 | VRAM (fp16) |
|------|--------|------|---------|------|------------|
| Tiny | 3900 万 | 4 | 384 | 6 | ~1 GB |
| Base | 7400 万 | 6 | 512 | 8 | ~1 GB |
| Small | 2.44 亿 | 12 | 768 | 12 | ~2 GB |
| Medium | 7.69 亿 | 24 | 1024 | 16 | ~5 GB |
| Large | 15.5 亿 | 32 | 1280 | 20 | ~10 GB |
| Large-v3 | 15.5 亿 | 32 | 1280 | 20 | ~10 GB |
| Large-v3-turbo | 8.09 亿 | 32 | 1280 | 20 | ~6 GB（4 层解码器） |

Large-v3-turbo（2024）将解码器从 32 层削减到 4 层。解码速度提升 8×，词错率（WER）回归小于 1 个点。这一解码速度的解锁是 Whisper-turbo 成为 2026 年实时语音助手默认选择的原因。

### Whisper 不做什么

- 无说话人分离（谁在说话）。为此配合 pyannote 使用。
- 本身不支持实时流式——30 秒窗口是固定的。现代封装器（`faster-whisper`、`WhisperX`）通过 VAD + 重叠来附加流式支持。
- 无 30 秒以上的长文本上下文，需要外部分块。实践中效果好，因为人类语音转录很少需要远程上下文。

### 2026 年格局

| 任务 | 模型 | 备注 |
|------|------|------|
| 英语 ASR | Whisper-turbo、Moonshine | Moonshine 在边缘端快 4× |
| 多语言 ASR | Whisper-large-v3 | 97 种语言 |
| 流式 ASR | faster-whisper + VAD | 150 ms 延迟目标可实现 |
| TTS | Piper、XTTS-v2、Kokoro | 编码器-解码器模式，但 Whisper 形态 |
| 音频 + 语言 | AudioLM、SeamlessM4T | 文本 token + 音频 token 在一个 Transformer 中 |

## 动手实现

见 `code/main.py`。我们不训练 Whisper——我们构建对数梅尔频谱图流水线 + 任务 token 提示格式化器。这些是你在生产中实际接触的部分。

### 步骤一：合成音频

生成 16 kHz 采样、440 Hz 的 1 秒正弦波。16000 个样本。

### 步骤二：对数梅尔频谱图（简化版）

完整梅尔频谱图需要 FFT。我们做一个简化的分帧 + 每帧能量版本，无需 `librosa` 即可展示流水线：

```python
def frame_signal(x, frame_size=400, hop=160):
    frames = []
    for start in range(0, len(x) - frame_size + 1, hop):
        frames.append(x[start:start + frame_size])
    return frames
```

帧 = 25 ms，跳步 = 10 ms。匹配 Whisper 的加窗方式。每帧能量代替梅尔频带用于教学。

### 步骤三：填充到 30 秒

Whisper 始终处理 30 秒块。将频谱图填充（或裁剪）至 3000 帧。

### 步骤四：构建提示 token

```python
def whisper_prompt(lang="en", task="transcribe", timestamps=True):
    tokens = ["<|startoftranscript|>", f"<|{lang}|>", f"<|{task}|>"]
    if not timestamps:
        tokens.append("<|notimestamps|>")
    return tokens
```

这就是整个任务控制面。一个 4 个 token 的前缀。

## 生产使用

```python
import whisper
model = whisper.load_model("large-v3-turbo")
result = model.transcribe("meeting.wav", language="en", task="transcribe")
print(result["text"])
print(result["segments"][0]["start"], result["segments"][0]["end"])
```

更快的、OpenAI 兼容版：

```python
from faster_whisper import WhisperModel
model = WhisperModel("large-v3-turbo", compute_type="int8_float16")
segments, info = model.transcribe("meeting.wav", vad_filter=True)
for s in segments:
    print(f"{s.start:.2f} - {s.end:.2f}: {s.text}")
```

**2026 年何时选择 Whisper：**

- 用单一模型进行多语言 ASR。
- 对嘈杂、多样化音频进行鲁棒转录。
- 研究/原型 ASR——最快的起点。

**何时选择其他方案：**

- 边缘设备的超低延迟流式——在相同质量下 Moonshine 优于 Whisper。
- 需要 <200 ms 的实时对话 AI——专用流式 ASR。
- 说话人分离——Whisper 不做此事；附加 pyannote。

## 上手实践

见 `outputs/skill-asr-configurator.md`。该技能为新语音应用选择 ASR 模型、解码参数和预处理流水线。

## 练习

1. **简单。** 运行 `code/main.py`。确认 16 kHz、10 ms 跳步的 1 秒信号的帧数约为 100 帧。30 秒约 3000 帧。
2. **中等。** 使用 `numpy.fft` 构建完整的对数梅尔频谱图。验证 80 个梅尔频带在数值误差范围内与 `librosa.feature.melspectrogram(n_mels=80)` 匹配。
3. **困难。** 实现流式推理：将音频分成 10 秒窗口、2 秒重叠，对每个块运行 Whisper，合并转录结果。在 5 分钟播客样本上与单次处理比较词错率。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| 梅尔频谱图（Mel spectrogram） | "音频图像" | 二维表示：一轴为频率频带，另一轴为时间帧；每格对数能量。 |
| 对数梅尔（Log-mel） | "Whisper 看到的" | 经过对数处理的梅尔频谱图；近似人类对响度的感知。 |
| 帧（Frame） | "一个时间切片" | 25 ms 的样本窗口；以 10 ms 步长重叠。 |
| 任务 token（Task token） | "语音的提示前缀" | 解码器提示中的特殊 token，如 `<|transcribe|>` / `<|translate|>`。 |
| 语音活动检测（VAD） | "找到语音" | 在 ASR 之前去除静音的门控；大幅降低成本。 |
| CTC | "连接时序分类" | 无对齐训练的经典 ASR 损失；Whisper **不**使用它。 |
| Whisper-turbo | "小解码器，完整编码器" | large-v3 编码器 + 4 层解码器；解码速度快 8×。 |
| Faster-whisper | "生产包装器" | CTranslate2 重新实现；int8 量化；比 OpenAI 参考快 4×。 |

## 延伸阅读

- [Radford et al. (2022). Robust Speech Recognition via Large-Scale Weak Supervision](https://arxiv.org/abs/2212.04356) — Whisper 论文
- [OpenAI Whisper repo](https://github.com/openai/whisper) — 参考代码 + 模型权重。阅读 `whisper/model.py`，约 400 行从上到下展示 Conv1D 茎 + 编码器 + 解码器
- [OpenAI Whisper — `whisper/decoding.py`](https://github.com/openai/whisper/blob/main/whisper/decoding.py) — 步骤 5-6 中描述的束搜索 + 任务 token 逻辑；500 行，完全可读
- [Baevski et al. (2020). wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations](https://arxiv.org/abs/2006.11477) — 前身；在某些设置中仍是最优特征
- [SYSTRAN/faster-whisper](https://github.com/SYSTRAN/faster-whisper) — 生产包装器，比参考快 4×
- [Jia et al. (2024). Moonshine: Speech Recognition for Live Transcription and Voice Commands](https://arxiv.org/abs/2410.15608) — 2024 年边缘友好的 ASR，Whisper 形态但更小
- [HuggingFace `modeling_whisper.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/whisper/modeling_whisper.py) — 完整实现（编码器、解码器、交叉注意力、生成），与本课架构图对应
