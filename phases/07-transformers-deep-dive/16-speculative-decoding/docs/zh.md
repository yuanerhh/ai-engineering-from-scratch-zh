# 投机解码——草稿、验证、重复

> 自回归解码是串行的。每个 token 等待前一个。投机解码打破这条链：廉价模型起草 N 个 token，昂贵模型一次前向传播验证所有 N 个。当草稿正确时，你用一次大模型前向传播换来了 N 次生成。

**类型：** 构建
**语言：** Python
**前置条件：** Phase 7 · 07（GPT 因果 LM）、Phase 7 · 12（KV 缓存 & Flash Attention）
**时长：** 约 60 分钟

## 问题背景

70B LLM 在 H100 上采样一个 token 约需 30 ms。3B 草稿模型约需 3 ms。如果让 3B 模型提前起草 5 个 token，然后运行 70B 模型*一次*验证所有 5 个，总计 `5×3 + 30 = 45 ms` 可接受最多 5 个 token——相比直接生成的 `5×30 = 150 ms`。这就是投机解码的全部卖点：用少量额外 GPU 内存（草稿模型）换取 2-4 倍更低的解码延迟。

这个技巧必须保留分布。由 Leviathan et al.（2023）和 Chen et al. 并行引入的投机采样保证输出序列与大模型独自产生的序列**分布完全相同**。无质量权衡，只是更快。

四类草稿-验证对主导 2026 年的推理：

1. **朴素投机（Leviathan 2023）。** 独立草稿模型（如 Llama 3 1B）+ 验证器（如 Llama 3 70B）。
2. **Medusa（Cai 2024）。** 验证器上的多个解码头并行预测位置 `t+1..t+k`。无需独立草稿模型。
3. **EAGLE 系列（Li 2024，2025）。** 复用验证器隐藏状态的轻量级草稿；接受率优于朴素版；典型 3-4 倍。
4. **前瞻解码（Fu 2024）。** Jacobi 迭代；完全不需要草稿模型。自我投机。小众但无依赖。

2026 年每个生产推理栈都默认搭载投机解码。vLLM、TensorRT-LLM、SGLang 和 llama.cpp 都支持至少朴素 + EAGLE-2。

## 核心概念

### 核心算法

给定验证器 `M_q` 和更廉价的草稿 `M_p`：

1. 设 `x_1..x_k` 为已解码的前缀。
2. **起草**：使用 `M_p` 自回归地提出 `d_{k+1}, d_{k+2}, ..., d_{k+N}`，带草稿概率 `p_1..p_N`。
3. **并行验证**：对 `x_1..x_k, d_{k+1}, ..., d_{k+N}` 运行 `M_q` 一次，得到位置 `k+1..k+N+1` 的验证器概率 `q_1..q_{N+1}`。
4. **从左到右逐一接受/拒绝草稿 token**：对每个 `i`，以概率 `min(1, q_i(d_i) / p_i(d_i))` 接受。
5. 在位置 `j` 首次拒绝时：从"残差"分布 `(q_j - p_j)_+` 归一化后采样 `t_j`。`j` 之后的所有草稿被丢弃。
6. 接受所有 `N` 个时：从 `q_{N+1}` 采样一个额外 token `t_{N+1}`（免费奖励 token）。

残差分布技巧是使输出分布与 `M_q` 独立采样完全相同的数学洞察。

### 什么决定加速比

设 `α` = 每个草稿 token 的预期接受率。设 `c` = 草稿与验证器的成本比。每步：

- 朴素生成每个 token 调用一次大模型。
- 投机每 `(1 - α^{N+1}) / (1 - α) ≈ 1/(1-α)` 个 token 调用一次大模型（当 α 较高时）。

`α = 0.75`、`N = 5` 时的典型经验法则：大模型调用减少约 3 倍。草稿成本是 5 倍廉价。总实际时间降低约 2.5 倍。

**α 取决于：**

- 草稿对验证器的近似程度。同一系列 / 同一训练数据显著提升 α。
- 解码策略。贪婪草稿对贪婪验证器：高 α。温度采样：更难匹配；接受率下降。
- 任务类型。代码和结构化输出接受更多（可预测）；自由创意写作接受更少。

### Medusa——无草稿模型的起草

Medusa 用验证器上的额外输出头替换草稿模型。在位置 `t`：

```
共享主干 → 隐藏状态 h_t
    ├── 头_0：预测 t+1 处的 token（标准 LM 头）
    ├── 头_1：预测 t+2 处的 token
    ├── 头_2：预测 t+3 处的 token
    ├── 头_3：预测 t+4 处的 token
```

每个头输出自己的 logit。推理时，从每个头采样得到候选序列，然后用树注意力方案一次性验证所有候选续写。

优点：无第二个模型。缺点：增加可训练参数；需要有监督微调阶段（约 10 亿 token）；接受率略低于使用好草稿的朴素投机。

### EAGLE——通过复用隐藏状态改进草稿

EAGLE-1/2/3（Li et al.，2024-2025）将草稿模型做成一个微型 Transformer（通常 1 层），摄入验证器的最后一层隐藏状态。由于草稿能看到验证器的特征表示，其预测与验证器的输出分布强相关。接受率从约 0.6（朴素）提升到 0.85 以上。

EAGLE-3（2025）添加了对候选续写的树搜索。vLLM 和 SGLang 将 EAGLE-2/3 作为 Llama 3/4 和 Qwen 3 的默认投机路径。

### KV 缓存协调

验证在一次前向传播中向验证器输入 `N` 个草稿 token，这将验证器的 KV 缓存扩展 `N` 个条目。如果某些草稿被拒绝，必须将缓存回滚到已接受的前缀长度。

生产实现（vLLM 的 `--speculative-model`、TensorRT-LLM 的 LookaheadDecoder）用临时 KV 缓冲区处理这一问题。先写入，接受时提交。概念上并不难，但实现上比较繁琐。

## 动手实现

见 `code/main.py`。我们实现核心投机采样算法（拒绝步骤 + 残差分布），使用：

- 一个"大模型"，是手动编码分布上的确定性 softmax（以便解析验证接受数学）。
- 一个"草稿模型"，是大模型的扰动版本。
- 一个产生与直接采样相同边际分布的接受/拒绝循环。

### 步骤一：拒绝步骤

```python
def accept_or_reject(q_prob, p_prob, draft_token, u):
    ratio = q_prob / p_prob if p_prob > 0 else float("inf")
    return u < min(1.0, ratio)
```

`u` 是均匀随机数。`q_prob` 是验证器对草稿 token 的概率。`p_prob` 是草稿模型的概率。Leviathan 定理指出，这个 Bernoulli 决策，加上拒绝时从残差分布采样，完全保留了验证器的分布。

### 步骤二：残差分布

```python
def residual_dist(q, p):
    raw = [max(0.0, qi - pi) for qi, pi in zip(q, p)]
    s = sum(raw)
    return [r / s for r in raw]
```

逐元素用 `p` 减 `q`，将负值截断为零，重新归一化。任何拒绝时从中采样。

### 步骤三：一次投机步骤

```python
def spec_step(prefix, q_model, p_model, N, rng):
    drafts = []
    p_probs = []
    ctx = list(prefix)
    for _ in range(N):
        p_dist = p_model(ctx)
        d = sample(p_dist, rng)
        drafts.append(d)
        p_probs.append(p_dist[d])
        ctx.append(d)

    q_dists = [q_model(prefix + drafts[:i]) for i in range(N + 1)]

    for i, d in enumerate(drafts):
        u = rng.random()
        q_prob = q_dists[i][d]
        p_prob = p_probs[i]
        if u < min(1.0, q_prob / p_prob if p_prob > 0 else float("inf")):
            prefix = prefix + [d]
        else:
            res = residual_dist(q_dists[i], p_model(prefix))
            prefix = prefix + [sample(res, rng)]
            return prefix
    prefix = prefix + [sample(q_dists[N], rng)]
    return prefix
```

五个接受 → 一个奖励 → 一次验证器传播产生六个 token。

### 步骤四：测量接受率

以不同草稿质量级别运行 10,000 次投机步骤。绘制接受率 vs 草稿与验证器分布之间的 KL 散度。你应该看到清晰的单调关系。

### 步骤五：验证分布等价性

经验验证：投机循环产生的 token 直方图应与直接从验证器采样的直方图匹配。这是 Leviathan 定理的实践。卡方检验在采样误差范围内确认。

## 生产使用

生产部署：

```bash
# vLLM 使用 EAGLE
vllm serve meta-llama/Llama-3.1-70B-Instruct \
    --speculative-model /models/llama-3.1-eagle-70b \
    --speculative-draft-tensor-parallel-size 1 \
    --num-speculative-tokens 5

# vLLM 使用朴素草稿模型
vllm serve meta-llama/Llama-3.1-70B-Instruct \
    --speculative-model meta-llama/Llama-3.2-1B-Instruct \
    --num-speculative-tokens 5
```

截至 2026 年中期，TensorRT-LLM 具有最快的 Medusa 路径。`faster-whisper` 用小型草稿为 Whisper-large 包装了投机解码。

**选择草稿策略：**

| 策略 | 何时选择 | 加速比 |
|------|---------|--------|
| 朴素草稿（1B/3B Llama 系列） | 快速原型，无需训练 | 1.8-2.3× |
| Medusa 头 | 可以微调验证器 | 2-3× |
| EAGLE-2 / 3 | 生产，最大速度 | 3-4× |
| 前瞻解码 | 无草稿、无训练、无额外参数 | 1.3-1.6× |

**何时不用投机解码：**

- 1-5 个 token 的单序列生成。开销主导。
- 极具创意 / 高温度采样（α 下降）。
- 内存受限部署（草稿模型增加 VRAM）。

## 上手实践

见 `outputs/skill-spec-decode-picker.md`。该技能为新推理工作负载选择投机解码策略（朴素 / Medusa / EAGLE / 前瞻）和调优参数（N、草稿温度）。

## 练习

1. **简单。** 运行 `code/main.py`。在 50,000 个 token 上以卡方检验 p > 0.05 确认投机 token 分布与验证器直接采样分布匹配。
2. **中等。** 绘制 `α = 0.5, 0.7, 0.85` 时加速比（每次大模型前向传播的 token 数）随 `N` 变化的曲线。确定每个 α 的最优 `N`。（提示：每次验证调用的预期 token 数 = `(1 - α^{N+1}) / (1 - α)`。）
3. **困难。** 实现微型 Medusa：取第 14 课综合项目的 GPT，添加 3 个额外的 LM 头预测位置 t+2、t+3、t+4。在 tinyshakespeare 上以联合多头损失训练。与通过截断同一模型制作的朴素草稿比较接受率。
4. **困难。** 实现回滚：从 10 个 token 前缀的 KV 缓存开始，输入 5 个草稿 token，模拟位置 3 处的拒绝。验证你的缓存在下一次迭代时正确读取"前缀 + 前 2 个已接受草稿"。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| 草稿模型（Draft model） | "廉价的那个" | 提出候选 token 的较小模型；通常比验证器便宜 10-50 倍。 |
| 验证器（Verifier） | "大的那个" | 我们保留其分布的目标模型；每次投机步骤运行一次。 |
| 接受率（α） | "草稿正确的频率" | 验证器接受草稿的每 token 概率。典型 0.7-0.9。 |
| 残差分布（Residual distribution） | "拒绝后备方案" | `(q - p)_+` 归一化；拒绝时从中采样保留验证器的分布。 |
| 奖励 token（Bonus token） | "免费的那个" | 当所有 N 个草稿被接受时，从验证器的下一步分布再采样一个。 |
| Medusa | "无草稿的投机" | 验证器上的多个 LM 头并行预测位置 t+1..t+k。 |
| EAGLE | "隐藏状态草稿" | 以验证器最后一层隐藏状态为条件的微型 Transformer 草稿。 |
| 前瞻解码（Lookahead decoding） | "Jacobi 迭代" | 使用定点迭代的自我投机；无草稿模型。 |
| 树注意力（Tree attention） | "同时验证多个候选" | 分支验证，同时考虑多个草稿续写。 |
| KV 回滚（KV rollback） | "撤销被拒绝的草稿" | 临时 KV 缓冲区；接受时提交，拒绝时丢弃。 |

## 延伸阅读

- [Leviathan, Kalman, Matias (2023). Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192) — 核心算法和等价定理
- [Chen et al. (2023). Accelerating Large Language Model Decoding with Speculative Sampling](https://arxiv.org/abs/2302.01318) — 并行引入；简洁的 Bernoulli-拒绝证明
- [Cai et al. (2024). Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads](https://arxiv.org/abs/2401.10774) — Medusa 论文；树注意力验证
- [Li et al. (2024). EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty](https://arxiv.org/abs/2401.15077) — EAGLE-1；隐藏状态条件草稿
- [Li et al. (2024). EAGLE-2: Faster Inference of Language Models with Dynamic Draft Trees](https://arxiv.org/abs/2406.16858) — EAGLE-2；动态树深度
- [Li et al. (2025). EAGLE-3: Scaling up Inference Acceleration of Large Language Models via Training-Time Test](https://arxiv.org/abs/2503.01840) — EAGLE-3
- [Fu et al. (2024). Break the Sequential Dependency of LLM Inference Using Lookahead Decoding](https://arxiv.org/abs/2402.02057) — 前瞻解码，无草稿方法
- [vLLM docs — Speculative Decoding](https://docs.vllm.ai/en/latest/features/spec_decode.html) — 规范生产参考，四种策略全部配置
- [SafeAILab / EAGLE reference implementation](https://github.com/SafeAILab/EAGLE) — EAGLE-1/2/3 的参考代码
