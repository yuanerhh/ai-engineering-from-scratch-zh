# 超参数调优

> 超参数是训练开始前你拨动的旋钮。调好它们是平庸模型和优秀模型的分水岭。

**类型：** 构建
**语言：** Python
**前置条件：** 第2阶段 第11课（集成方法）
**时间：** ~90分钟

## 学习目标

- 从零实现网格搜索、随机搜索和贝叶斯优化，比较它们的样本效率
- 解释为什么当大多数超参数有效维度低时，随机搜索优于网格搜索
- 使用代理模型和采集函数构建贝叶斯优化循环来引导搜索
- 设计一个通过适当交叉验证避免过拟合验证集的超参数调优策略

## 问题所在

你的梯度提升模型有学习率、树的数量、最大深度、每叶最小样本数、子采样比例和列采样比例。这是六个超参数。如果每个有5个合理值，网格有 5^6 = 15,625 种组合。每次训练需要10秒。全部试完需要43小时计算时间。

网格搜索是最直观的方法，也是大规模场景下最糟糕的方法。随机搜索用更少的计算量能做得更好。贝叶斯优化通过从过去的评估中学习，能做得更好。知道使用哪种策略，以及哪些超参数真正重要，能节省数天的GPU时间。

## 核心概念

### 参数 vs 超参数

参数在训练期间被学习（权重、偏置、分裂阈值）。超参数在训练开始前设置，控制学习如何发生。

| 超参数 | 控制什么 | 典型范围 |
|--------|---------|---------|
| 学习率 | 每次更新的步长 | 0.001 到 1.0 |
| 树/轮次数量 | 训练多久 | 10 到 10,000 |
| 最大深度 | 模型复杂度 | 1 到 30 |
| 正则化（lambda） | 防止过拟合 | 0.0001 到 100 |
| 批量大小 | 梯度估计噪声 | 16 到 512 |
| Dropout率 | 丢弃的神经元比例 | 0.0 到 0.5 |

### 网格搜索（Grid Search）

网格搜索评估所有指定值的每种组合。它是穷举的且易于理解，但随超参数数量呈指数扩展。

```
2个超参数的网格：

  learning_rate: [0.01, 0.1, 1.0]
  max_depth:     [3, 5, 7]

  评估次数：3 x 3 = 9种组合

  (0.01, 3)  (0.01, 5)  (0.01, 7)
  (0.1,  3)  (0.1,  5)  (0.1,  7)
  (1.0,  3)  (1.0,  5)  (1.0,  7)
```

网格搜索有一个根本缺陷：如果一个超参数重要而另一个不重要，大多数评估都被浪费了。从9次评估中，重要参数只获得3个唯一值。

### 随机搜索（Random Search）

随机搜索从分布中采样超参数，而不是从网格中。用同样的9次评估预算，每个超参数可以获得9个唯一值。

```mermaid
flowchart LR
    subgraph Grid Search
        G1[3个唯一学习率]
        G2[3个唯一最大深度]
        G3[9次总评估]
    end

    subgraph Random Search
        R1[9个唯一学习率]
        R2[9个唯一最大深度]
        R3[9次总评估]
    end
```

随机搜索优于网格搜索的原因（Bergstra & Bengio, 2012）：

- 大多数超参数的有效维度低。对于给定问题，6个超参数中通常只有1-2个真正重要。
- 网格搜索在不重要的维度上浪费评估。
- 随机搜索在相同预算下更密集地覆盖重要维度。
- 在60次随机试验后，有95%的概率找到距最优值5%范围内的点（如果搜索空间中存在的话）。

### 贝叶斯优化（Bayesian Optimization）

随机搜索忽略结果。它不知道高学习率导致发散，或深度3始终优于深度10。贝叶斯优化使用过去的评估来决定下一步在哪里搜索。

```mermaid
flowchart TD
    A[定义搜索空间] --> B[评估初始随机点]
    B --> C[用结果拟合代理模型]
    C --> D[使用采集函数选择下一个点]
    D --> E[在该点评估模型]
    E --> F{预算耗尽？}
    F -->|否| C
    F -->|是| G[返回找到的最佳超参数]
```

两个关键组件：

**代理模型：** 一个计算成本低的模型（通常是高斯过程），近似昂贵的目标函数。它在搜索空间的任何点都能给出预测和不确定性估计。

**采集函数：** 通过平衡利用（在已知好点附近搜索）和探索（在不确定性高的区域搜索）来决定下一步在哪里评估。常见选择：

- **期望改进（EI）：** 我们期望在这个点比当前最佳值改进多少？
- **置信上界（UCB）：** 预测值加上不确定性的倍数。UCB越高意味着越有希望或越未被探索。
- **改进概率（PI）：** 这个点超过当前最佳值的概率是多少？

贝叶斯优化通常比随机搜索少用2-5倍的评估次数就能找到更好的超参数。拟合代理模型的开销与训练实际模型相比可以忽略不计。

### 早停（Early Stopping）

不是每次训练都需要完成。如果一个配置在10个轮次后明显很差，停止它继续前进。这是超参数搜索上下文中的早停。

策略：
- **基于耐心：** 如果验证损失连续N个轮次没有改善则停止
- **中位数剪枝：** 如果试验的中间结果比同一步骤完成试验的中位数更差则停止
- **Hyperband：** 给许多配置分配小预算，然后逐步增加最好的配置的预算

Hyperband特别有效。它以每个1个轮次启动81个配置，保留前三分之一，给它们3个轮次，保留前三分之一，以此类推。这比以完整预算评估所有配置快10-50倍。

### 学习率调度器

学习率几乎总是最重要的超参数。调度器在训练期间调整它，而不是保持固定。

| 调度器 | 公式 | 何时使用 |
|--------|------|---------|
| 步进衰减 | 每N个轮次乘以0.1 | 经典CNN训练 |
| 余弦退火 | lr * 0.5 * (1 + cos(pi * t / T)) | 现代默认 |
| 预热+衰减 | 线性增加然后余弦衰减 | Transformer |
| 单循环 | 在一个周期内增加然后减少 | 快速收敛 |
| 平台时减少 | 指标停滞时按因子减少 | 安全默认 |

### 超参数重要性

并非所有超参数都同等重要。对随机森林（Probst等，2019）和梯度提升的研究显示了一致的规律：

**高重要性：**
- 学习率（总是首先调优）
- 估计器/轮次数量（使用早停代替调优）
- 正则化强度

**中等重要性：**
- 最大深度/层数
- 每叶最小样本数/权重衰减
- 子采样比例

**低重要性：**
- 最大特征数（对随机森林）
- 具体的激活函数选择
- 批量大小（在合理范围内）

首先调优重要的超参数，其余的保持默认值。

### 实用策略

```mermaid
flowchart TD
    A[从默认值开始] --> B[粗粒度随机搜索：20-50次试验]
    B --> C[识别重要超参数]
    C --> D[细粒度随机或贝叶斯搜索：在缩小范围内50-100次试验]
    D --> E[使用最佳超参数的最终模型]
    E --> F[在完整训练数据上重新训练]
```

具体工作流程：

1. **从库的默认值开始。** 它们由经验丰富的从业者选择，通常已经达到80%的效果。
2. **粗粒度随机搜索。** 宽范围，20-50次试验。使用早停快速淘汰差的运行。
3. **分析结果。** 哪些超参数与性能相关？缩小搜索空间。
4. **细粒度搜索。** 在缩小的空间中使用贝叶斯优化或聚焦随机搜索，50-100次试验。
5. **在所有训练数据上重新训练**，使用找到的最佳超参数。

### 交叉验证集成

在单个验证集分割上调优超参数是有风险的。最佳超参数可能过拟合到特定的验证折。嵌套交叉验证通过两个循环解决这个问题：

- **外循环**（评估）：将数据分成训练+验证集和测试集。报告无偏性能。
- **内循环**（调优）：将训练+验证集分成训练集和验证集。找到最佳超参数。

```mermaid
flowchart TD
    D[完整数据集] --> O1[外折1：测试]
    D --> O2[外折2：测试]
    D --> O3[外折3：测试]
    D --> O4[外折4：测试]
    D --> O5[外折5：测试]

    O1 --> I1[在剩余数据上进行内5折CV]
    I1 --> T1[折1的最佳超参数]
    T1 --> E1[在外测试折1上评估]

    O2 --> I2[在剩余数据上进行内5折CV]
    I2 --> T2[折2的最佳超参数]
    T2 --> E2[在外测试折2上评估]
```

每个外折独立地找到自己的最佳超参数。外折分数是泛化性能的无偏估计。

使用sklearn：

```python
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor

inner_cv = GridSearchCV(
    GradientBoostingRegressor(),
    param_grid={
        "learning_rate": [0.01, 0.05, 0.1],
        "max_depth": [2, 3, 5],
        "n_estimators": [50, 100, 200],
    },
    cv=5,
    scoring="neg_mean_squared_error",
)

outer_scores = cross_val_score(
    inner_cv, X, y, cv=5, scoring="neg_mean_squared_error"
)

print(f"嵌套CV MSE: {-outer_scores.mean():.4f} +/- {outer_scores.std():.4f}")
```

这很昂贵（5个外折 x 5个内折 x 27个网格点 = 675次模型拟合），但它给你一个可信的性能估计。在论文中报告最终结果或决策风险较高时使用它。

### 实用技巧

**从学习率开始。** 对于基于梯度的方法，它几乎总是最重要的超参数。错误的学习率使其他一切都无关紧要。将其他超参数固定在默认值，先扫描学习率。

**对学习率和正则化使用对数均匀分布。** 0.001和0.01之间的差异与0.1和1.0之间的差异同等重要。线性搜索会在大值端浪费预算。

**使用早停代替调优n_estimators。** 对于提升和神经网络，将n_estimators或epochs设高，让早停决定何时停止。这从搜索中去除了一个超参数。

**预算分配。** 将60%的调优预算花在最重要的2个超参数上。将剩余的40%花在其他所有超参数上。前2个占了大部分性能变化。

**尺度很重要。** 永远不要在对数尺度上搜索批量大小（16、32、64就好）。始终在对数尺度上搜索学习率。将搜索分布与超参数影响模型的方式匹配。

| 模型类型 | 顶级超参数 | 推荐搜索 | 预算 |
|---------|-----------|---------|------|
| 随机森林 | n_estimators, max_depth, min_samples_leaf | 随机搜索，50次试验 | 低（训练快） |
| 梯度提升 | learning_rate, n_estimators, max_depth | 贝叶斯，100次试验+早停 | 中 |
| 神经网络 | learning_rate, weight_decay, batch_size | 贝叶斯或随机，100+次试验 | 高（训练慢） |
| SVM | C, gamma（RBF核） | 对数尺度网格，25-50次试验 | 低（2个参数） |
| Lasso/Ridge | alpha | 对数尺度1D搜索，20次试验 | 极低 |
| XGBoost | learning_rate, max_depth, subsample, colsample | 贝叶斯，100-200次试验+早停 | 中 |

**拿不准时：** 随机搜索，试验次数为超参数数量的2倍（例如，6个超参数 = 最少12次试验）。你会惊讶于50次试验的随机搜索经常击败精心设计的网格搜索。

## 构建它

### 第1步：从零实现网格搜索

`code/tuning.py` 中的代码从零实现了网格搜索、随机搜索和一个简单的贝叶斯优化器。

```python
def grid_search(model_fn, param_grid, X_train, y_train, X_val, y_val):
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    best_score = -float("inf")
    best_params = None
    n_evals = 0

    for combo in itertools.product(*values):
        params = dict(zip(keys, combo))
        model = model_fn(**params)
        model.fit(X_train, y_train)
        score = evaluate(model, X_val, y_val)
        n_evals += 1

        if score > best_score:
            best_score = score
            best_params = params

    return best_params, best_score, n_evals
```

### 第2步：从零实现随机搜索

```python
def random_search(model_fn, param_distributions, X_train, y_train,
                  X_val, y_val, n_iter=50, seed=42):
    rng = np.random.RandomState(seed)
    best_score = -float("inf")
    best_params = None

    for _ in range(n_iter):
        params = {k: sample(v, rng) for k, v in param_distributions.items()}
        model = model_fn(**params)
        model.fit(X_train, y_train)
        score = evaluate(model, X_val, y_val)

        if score > best_score:
            best_score = score
            best_params = params

    return best_params, best_score, n_iter
```

### 第3步：贝叶斯优化（简化版）

核心思想：将高斯过程拟合到观测到的（超参数，分数）对，然后使用采集函数决定下一步在哪里看。

```python
class SimpleBayesianOptimizer:
    def __init__(self, search_space, n_initial=5):
        self.search_space = search_space
        self.n_initial = n_initial
        self.X_observed = []
        self.y_observed = []

    def _kernel(self, x1, x2, length_scale=1.0):
        dists = np.sum((x1[:, None, :] - x2[None, :, :]) ** 2, axis=2)
        return np.exp(-0.5 * dists / length_scale ** 2)

    def _fit_gp(self, X_new):
        X_obs = np.array(self.X_observed)
        y_obs = np.array(self.y_observed)
        y_mean = y_obs.mean()
        y_centered = y_obs - y_mean

        K = self._kernel(X_obs, X_obs) + 1e-4 * np.eye(len(X_obs))
        K_star = self._kernel(X_new, X_obs)

        L = np.linalg.cholesky(K)
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_centered))
        mu = K_star @ alpha + y_mean

        v = np.linalg.solve(L, K_star.T)
        var = 1.0 - np.sum(v ** 2, axis=0)
        var = np.maximum(var, 1e-6)

        return mu, var

    def _expected_improvement(self, mu, var, best_y):
        sigma = np.sqrt(var)
        z = (mu - best_y) / (sigma + 1e-10)
        ei = sigma * (z * norm_cdf(z) + norm_pdf(z))
        return ei

    def suggest(self):
        if len(self.X_observed) < self.n_initial:
            return sample_random(self.search_space)

        candidates = [sample_random(self.search_space) for _ in range(500)]
        X_cand = np.array([to_vector(c) for c in candidates])
        mu, var = self._fit_gp(X_cand)
        ei = self._expected_improvement(mu, var, max(self.y_observed))
        return candidates[np.argmax(ei)]

    def observe(self, params, score):
        self.X_observed.append(to_vector(params))
        self.y_observed.append(score)
```

GP代理在每个候选点给出两样东西：预测分数（mu）和不确定性（var）。期望改进平衡这两者：它偏好模型预测高分数或不确定性高的点。早期，大多数点有高不确定性，所以优化器进行探索。后来，它专注于最有希望的区域。

### 第4步：比较所有方法

在相同的合成目标上运行所有三种方法并比较：

```python
def synthetic_objective(params):
    lr = params["learning_rate"]
    depth = params["max_depth"]
    return -(np.log10(lr) + 2) ** 2 - (depth - 4) ** 2 + 10

param_grid = {
    "learning_rate": [0.001, 0.01, 0.1, 1.0],
    "max_depth": [2, 3, 4, 5, 6, 7, 8],
}

# 网格搜索
grid_best = None
grid_score = -float("inf")
grid_history = []
for combo in itertools.product(*param_grid.values()):
    params = dict(zip(param_grid.keys(), combo))
    score = synthetic_objective(params)
    grid_history.append((params, score))
    if score > grid_score:
        grid_score = score
        grid_best = params

# 随机搜索
param_dist = {
    "learning_rate": ("log_float", 0.001, 1.0),
    "max_depth": ("int", 2, 8),
}

rand_best = None
rand_score = -float("inf")
rand_history = []
rng = np.random.RandomState(42)
for _ in range(28):
    params = {k: sample(v, rng) for k, v in param_dist.items()}
    score = synthetic_objective(params)
    rand_history.append((params, score))
    if score > rand_score:
        rand_score = score
        rand_best = params

# 贝叶斯优化
optimizer = SimpleBayesianOptimizer(param_dist, n_initial=5)
bayes_history = []
for _ in range(28):
    params = optimizer.suggest()
    score = synthetic_objective(params)
    optimizer.observe(params, score)
    bayes_history.append((params, score))
bayes_score = max(s for _, s in bayes_history)

print(f"{'方法':<20} {'最佳分数':>12} {'评估次数':>12}")
print("-" * 50)
print(f"{'网格搜索':<20} {grid_score:>12.4f} {len(grid_history):>12}")
print(f"{'随机搜索':<20} {rand_score:>12.4f} {len(rand_history):>12}")
print(f"{'贝叶斯优化':<20} {bayes_score:>12.4f} {len(bayes_history):>12}")
```

在相同预算下，贝叶斯优化通常最快找到最佳分数，因为它不会在明显差的区域浪费评估。随机搜索比网格搜索覆盖更多范围。网格搜索只在超参数很少且能负担穷举时才胜出。

## 使用它

### 实际中使用 Optuna

Optuna 是认真进行超参数调优的推荐库。它开箱即用地支持剪枝、分布式搜索和可视化。

```python
import optuna

def objective(trial):
    lr = trial.suggest_float("learning_rate", 1e-4, 1e-1, log=True)
    n_est = trial.suggest_int("n_estimators", 50, 500)
    max_depth = trial.suggest_int("max_depth", 2, 10)

    model = GradientBoostingRegressor(
        learning_rate=lr,
        n_estimators=n_est,
        max_depth=max_depth,
    )
    model.fit(X_train, y_train)
    return mean_squared_error(y_val, model.predict(X_val))

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=100)

print(f"最佳参数: {study.best_params}")
print(f"最佳MSE: {study.best_value:.4f}")
```

Optuna 关键特性：
- `suggest_float(..., log=True)` 用于最好在对数尺度上搜索的参数（学习率、正则化）
- `suggest_int` 用于整数参数
- `suggest_categorical` 用于离散选择
- 内置 MedianPruner 用于早停差的试验
- `study.trials_dataframe()` 用于分析

### 带剪枝的 Optuna

剪枝早停没有希望的试验，节省大量计算。以下是模式：

```python
import optuna
from sklearn.model_selection import cross_val_score

def objective(trial):
    params = {
        "learning_rate": trial.suggest_float("lr", 1e-4, 0.5, log=True),
        "max_depth": trial.suggest_int("max_depth", 2, 10),
        "n_estimators": trial.suggest_int("n_estimators", 50, 500),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
    }

    model = GradientBoostingRegressor(**params)
    scores = cross_val_score(model, X_train, y_train, cv=3,
                             scoring="neg_mean_squared_error")
    mean_score = -scores.mean()

    trial.report(mean_score, step=0)
    if trial.should_prune():
        raise optuna.TrialPruned()

    return mean_score

pruner = optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=5)
study = optuna.create_study(direction="minimize", pruner=pruner)
study.optimize(objective, n_trials=200)
```

`MedianPruner` 在同一步骤时，如果试验的中间值比所有完成试验的中位数更差，就停止该试验。剪枝需要调用 `trial.report()` 报告中间指标，调用 `trial.should_prune()` 检查是否应停止试验。`n_startup_trials=10` 确保至少10次试验在剪枝开始前完整完成。这通常节省40-60%的总计算量。

### sklearn 内置调优器

对于快速实验，sklearn 提供 `GridSearchCV`、`RandomizedSearchCV` 和 `HalvingRandomSearchCV`：

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import loguniform, randint

param_dist = {
    "learning_rate": loguniform(1e-4, 0.5),
    "max_depth": randint(2, 10),
    "n_estimators": randint(50, 500),
}

search = RandomizedSearchCV(
    GradientBoostingRegressor(),
    param_dist,
    n_iter=100,
    cv=5,
    scoring="neg_mean_squared_error",
    random_state=42,
    n_jobs=-1,
)
search.fit(X_train, y_train)
print(f"最佳参数: {search.best_params_}")
print(f"最佳CV MSE: {-search.best_score_:.4f}")
```

对学习率和正则化使用来自scipy的 `loguniform`。对整数超参数使用 `randint`。`n_jobs=-1` 标志跨所有CPU核并行化。

### 超参数调优中的常见错误

**通过预处理导致数据泄漏。** 如果在交叉验证前在完整数据集上拟合标准化器，来自验证折的信息泄漏到训练中。始终将预处理放入 `Pipeline`，使其只在训练折上拟合。

**过拟合验证集。** 运行数千次试验实际上是在验证集上训练。使用嵌套交叉验证进行最终性能估计，或保留一个在调优期间从不触碰的单独测试集。

**搜索范围太窄。** 如果最佳值在搜索空间的边界上，说明搜索范围不够宽。最优值可能在你的范围之外。始终检查最佳参数是否在边缘。

**忽略交互效应。** 学习率和估计器数量在提升中强烈交互。低学习率需要更多估计器。独立调优它们比一起调优效果更差。

**对迭代模型不使用早停。** 对于梯度提升和神经网络，将n_estimators或epochs设为高值并使用早停。这严格优于将迭代次数作为超参数调优。

## 练习

1. 以相同的总预算运行网格搜索和随机搜索（例如，50次评估）。比较找到的最佳分数。用不同的种子运行实验10次。随机搜索赢了多少次？

2. 从零实现 Hyperband。从81个配置开始，每个训练1个轮次。每轮保留前1/3并将其预算翻三倍。比较总计算量（所有配置的所有轮次之和）与以完整预算运行81个配置的计算量。

3. 将学习率调度器（余弦退火）添加到第11课的梯度提升实现中。与固定学习率相比是否有帮助？

4. 使用 Optuna 在真实数据集（例如sklearn的乳腺癌数据集）上调优 `RandomForestClassifier`。使用 `optuna.visualization.plot_param_importances(study)` 查看哪些超参数最重要。是否与本课的重要性排名一致？

5. 实现一个简单的采集函数（期望改进）并演示探索与利用。绘制代理模型的均值和不确定性，并显示EI选择下一步在哪里评估。

## 关键术语

| 术语 | 人们说的 | 实际含义 |
|------|---------|---------|
| 超参数（Hyperparameter） | "你选择的设置" | 训练前设置的值，控制学习过程，不从数据中学习 |
| 网格搜索（Grid search） | "尝试每种组合" | 在指定参数网格上穷举搜索，指数级代价 |
| 随机搜索（Random search） | "只是随机采样" | 从分布中采样超参数，比网格搜索更密集地覆盖重要维度 |
| 贝叶斯优化（Bayesian optimization） | "智能搜索" | 使用目标的代理模型决定下一步在哪里评估，平衡探索和利用 |
| 代理模型（Surrogate model） | "廉价近似" | 近似昂贵目标函数的模型（通常是高斯过程） |
| 采集函数（Acquisition function） | "下一步看哪里" | 通过平衡期望改进和不确定性对候选点评分 |
| 早停（Early stopping） | "停止浪费时间" | 验证性能停止改善时提前终止训练 |
| Hyperband | "配置的锦标赛括号" | 自适应资源分配：以小预算启动许多配置，保留最好的并增加其预算 |
| 学习率调度器（Learning rate scheduler） | "训练期间改变lr" | 在训练过程中调整学习率以实现更好收敛的函数 |

## 延伸阅读

- [Bergstra & Bengio: Random Search for Hyper-Parameter Optimization (2012)](https://jmlr.org/papers/v13/bergstra12a.html) -- 证明随机搜索优于网格搜索的论文
- [Snoek et al., Practical Bayesian Optimization of Machine Learning Algorithms (2012)](https://arxiv.org/abs/1206.2944) -- ML的贝叶斯优化
- [Li et al., Hyperband: A Novel Bandit-Based Approach (2018)](https://jmlr.org/papers/v18/16-558.html) -- Hyperband论文
- [Optuna: A Next-generation Hyperparameter Optimization Framework](https://arxiv.org/abs/1907.10902) -- Optuna论文
- [Probst et al., Tunability: Importance of Hyperparameters (2019)](https://jmlr.org/papers/v20/18-444.html) -- 哪些超参数重要
