# 朴素贝叶斯

> "朴素"假设是错的，但它仍然有效。这就是它的美妙之处。

**类型：** 构建
**语言：** Python
**前置条件：** 第2阶段 第01-07课（分类、贝叶斯定理）
**时间：** ~75分钟

## 学习目标

- 从零实现带拉普拉斯平滑的多项式朴素贝叶斯用于文本分类
- 解释为什么朴素独立性假设在数学上是错的，但在实践中仍能产生正确的类别排名
- 比较多项式、伯努利和高斯朴素贝叶斯变体，并为给定特征类型选择正确的变体
- 对高维稀疏数据评估朴素贝叶斯与逻辑回归，并解释偏差-方差权衡

## 问题所在

你需要对文本进行分类。将电子邮件分为垃圾邮件或非垃圾邮件。将客户评论分为正面或负面。将支持工单分类。你有数千个特征（每个词一个）和有限的训练数据。

大多数分类器在这里会崩溃。逻辑回归需要足够的样本来可靠地估计数千个权重。决策树每次只对一个词进行分裂，严重过拟合。在10,000维中的KNN毫无意义，因为每个点与其他所有点的距离都相等。

朴素贝叶斯能处理这个。它做出了一个数学上错误的假设（给定类别，每个特征独立于其他每个特征），但它仍然在文本分类上胜过"更聪明"的模型，特别是在训练集小的情况下。它在一次数据遍历中完成训练。它扩展到数百万个特征。它产生概率估计（尽管由于独立性假设通常校准不好）。

理解为什么错误的假设会导致好的预测，能教给你关于机器学习的某些根本性东西：最好的模型不是最正确的那个，而是对你的数据具有最佳偏差-方差权衡的那个。

## 核心概念

### 贝叶斯定理（快速回顾）

贝叶斯定理翻转条件概率：

```
P(类别 | 特征) = P(特征 | 类别) * P(类别) / P(特征)
```

我们想要 `P(类别 | 特征)` —— 给定文档中的词，文档属于某个类别的概率。我们可以从以下内容计算：
- `P(特征 | 类别)` —— 在这个类别的文档中看到这些词的可能性
- `P(类别)` —— 类别的先验概率（一般来说垃圾邮件有多常见？）
- `P(特征)` —— 证据，对所有类别相同，所以在比较时可以忽略

具有最高 `P(类别 | 特征)` 的类别获胜。

### 朴素独立性假设

精确计算 `P(特征 | 类别)` 需要估计所有特征一起的联合概率。对于10,000个词的词汇表，你需要估计 2^10,000 种可能组合的分布。这是不可能的。

朴素假设：给定类别，每个特征都条件独立。

```
P(w1, w2, ..., wn | 类别) = P(w1 | 类别) * P(w2 | 类别) * ... * P(wn | 类别)
```

不需要一个不可能的联合分布，而是估计n个简单的每特征分布。每个只需要一个计数。

这个假设明显是错误的。"机器"和"学习"在任何文档中都不是独立的。但分类器不需要正确的概率估计。它需要正确的排名——哪个类别的概率最高。独立性假设会引入系统误差，但这些误差对所有类别的影响类似，所以排名保持正确。

### 为什么它仍然有效

三个原因：

1. **排名优于校准。** 分类只需要排名最高的类别正确。即使 P(垃圾邮件) = 0.99999 而真实概率是 0.7，分类器仍然正确地选择垃圾邮件。我们不需要正确的概率，我们需要正确的赢家。

2. **高偏差，低方差。** 独立性假设是一个强先验。它强烈约束模型，防止过拟合。在训练数据有限的情况下，一个稍微错误但稳定的模型，胜过一个理论上正确但极不稳定的模型。这就是偏差-方差权衡的体现。

3. **特征冗余抵消。** 相关特征提供冗余证据。分类器重复计算这些证据，但它对正确的类别也重复计算。如果"机器"和"学习"总是一起出现，两者都为"技术"类别提供证据。朴素贝叶斯计算两次，但对正确的类别计算两次。

第四个实用原因：朴素贝叶斯极其快速。训练是对数据的一次遍历，计算频率。预测是矩阵乘法。你可以在几秒钟内对一百万个文档进行训练。这种速度意味着你可以更快地迭代，尝试更多特征集，运行比慢模型更多的实验。

### 分步数学推导

让我们通过一个具体例子来追踪。假设我们有两个类别：垃圾邮件和非垃圾邮件。我们的词汇表有三个词："free"、"money"、"meeting"。

训练数据：
- 垃圾邮件中出现"free" 80次、"money" 60次、"meeting" 10次（共150个词）
- 非垃圾邮件中出现"free" 5次、"money" 10次、"meeting" 100次（共115个词）
- 40% 的邮件是垃圾邮件，60% 不是

使用拉普拉斯平滑（alpha=1）：

```
P(free | 垃圾邮件)    = (80 + 1) / (150 + 3) = 81/153 = 0.529
P(money | 垃圾邮件)   = (60 + 1) / (150 + 3) = 61/153 = 0.399
P(meeting | 垃圾邮件) = (10 + 1) / (150 + 3) = 11/153 = 0.072

P(free | 非垃圾邮件)    = (5 + 1) / (115 + 3) = 6/118 = 0.051
P(money | 非垃圾邮件)   = (10 + 1) / (115 + 3) = 11/118 = 0.093
P(meeting | 非垃圾邮件) = (100 + 1) / (115 + 3) = 101/118 = 0.856
```

新邮件包含："free"（2次）、"money"（1次）、"meeting"（0次）。

```
log P(垃圾邮件 | 邮件) = log(0.4) + 2*log(0.529) + 1*log(0.399) + 0*log(0.072)
                       = -0.916 + 2*(-0.637) + (-0.919) + 0
                       = -3.109

log P(非垃圾邮件 | 邮件) = log(0.6) + 2*log(0.051) + 1*log(0.093) + 0*log(0.856)
                         = -0.511 + 2*(-2.976) + (-2.375) + 0
                         = -8.838
```

垃圾邮件以很大的差距获胜。"free"出现两次是垃圾邮件的强力证据。注意"meeting"不出现对两个对数求和都贡献零（0 * log(P)）——在多项式朴素贝叶斯中，不存在的词没有影响。是伯努利朴素贝叶斯才明确建模词的缺失。

### 三种变体

朴素贝叶斯有三种形式。每种对 `P(特征 | 类别)` 的建模方式不同。

#### 多项式朴素贝叶斯（Multinomial Naive Bayes）

将每个特征建模为计数。最适合特征是词频或TF-IDF值的文本数据。

```
P(word_i | 类别) = (类别中word_i的计数 + alpha) / (类别中总词数 + alpha * 词汇表大小)
```

`alpha` 是拉普拉斯平滑（见下）。这个变体是文本分类的主力。

#### 高斯朴素贝叶斯（Gaussian Naive Bayes）

将每个特征建模为正态分布。最适合连续特征。

```
P(x_i | 类别) = (1 / sqrt(2 * pi * var)) * exp(-(x_i - mean)^2 / (2 * var))
```

每个类别每个特征都有自己的均值和方差。当特征在每个类别内真正遵循钟形曲线时效果好。

#### 伯努利朴素贝叶斯（Bernoulli Naive Bayes）

将每个特征建模为二元（存在或不存在）。最适合短文本或二元特征向量。

```
P(word_i | 类别) = (包含word_i的类别文档数 + alpha) / (类别总文档数 + 2 * alpha)
```

与多项式不同，伯努利明确惩罚词的缺失。如果"free"通常出现在垃圾邮件中但在这封邮件中不存在，伯努利将其视为反对垃圾邮件的证据。

### 何时使用哪种变体

| 变体 | 特征类型 | 最适合 | 示例 |
|------|---------|--------|------|
| 多项式 | 计数或频率 | 文本分类、词袋模型 | 电子邮件垃圾过滤、主题分类 |
| 高斯 | 连续值 | 具有近似正态特征的表格数据 | Iris分类、传感器数据 |
| 伯努利 | 二元（0/1） | 短文本、二元特征向量 | 短信垃圾过滤、存在/缺失特征 |

### 拉普拉斯平滑（Laplace Smoothing）

如果测试数据中的词从未出现在某个特定类别的训练数据中，会发生什么？

不使用平滑：`P(词 | 类别) = 0/N = 0`。一个零乘以整个乘积使 `P(类别 | 特征) = 0`，无论其他所有证据如何。一个未见词破坏了整个预测，无论有多少其他证据支持它。

拉普拉斯平滑对每个特征计数添加一个小的计数 `alpha`（通常为1）：

```
P(word_i | 类别) = (count(word_i, 类别) + alpha) / (类别中总词数 + alpha * 词汇表大小)
```

使用alpha=1，每个词至少有一个很小的概率。测试邮件中出现"discombobulate"不再会消除垃圾邮件概率。平滑有贝叶斯解释：它等价于在词分布上设置均匀的狄利克雷先验。

更高的alpha意味着更强的平滑（更均匀的分布）。更低的alpha意味着模型更信任数据。Alpha是你调整的超参数。

alpha的效果：

| Alpha | 效果 | 何时使用 |
|-------|------|---------|
| 0.001 | 几乎不平滑，信任数据 | 非常大的训练集，不期望出现未见特征 |
| 0.1 | 轻度平滑 | 大训练集 |
| 1.0 | 标准拉普拉斯平滑 | 默认起点 |
| 10.0 | 重度平滑，扁平化分布 | 非常小的训练集，期望出现许多未见特征 |

### 对数空间计算

将数百个概率（每个都小于1）相乘会导致浮点数下溢。即使真实值是一个非常小的正数，乘积在浮点数中也会变为零。

解决方案：在对数空间中工作。不是乘以概率，而是将它们的对数相加：

```
log P(类别 | x1, x2, ..., xn) = log P(类别) + sum_i log P(xi | 类别)
```

这将预测变成了点积：

```
log_scores = X @ log_feature_probs.T + log_class_priors
prediction = argmax(log_scores)
```

矩阵乘法。这就是朴素贝叶斯预测如此快速的原因——它与单层线性模型的操作相同。

### 朴素贝叶斯 vs 逻辑回归

两者都是文本的线性分类器。区别在于它们建模的内容。

| 方面 | 朴素贝叶斯 | 逻辑回归 |
|------|-----------|---------|
| 类型 | 生成式（建模P(X\|Y)） | 判别式（建模P(Y\|X)） |
| 训练 | 计数频率 | 优化损失函数 |
| 小数据 | 更好（强先验有帮助） | 更差（不足以估计权重） |
| 大数据 | 更差（错误假设有害） | 更好（灵活的决策边界） |
| 特征 | 假设独立 | 处理相关性 |
| 速度 | 单次遍历，非常快 | 迭代优化 |
| 校准 | 概率估计差 | 概率估计更好 |

经验法则：从朴素贝叶斯开始。如果有足够数据且朴素贝叶斯达到瓶颈，则切换到逻辑回归。

### 分类流水线

```mermaid
flowchart LR
    A[原始文本] --> B[分词]
    B --> C[构建词汇表]
    C --> D[计算词频]
    D --> E[应用平滑]
    E --> F[计算对数概率]
    F --> G[预测：argmax P(类别|词)]

    style A fill:#f9f,stroke:#333
    style G fill:#9f9,stroke:#333
```

在实践中，我们在对数空间中工作以避免浮点数下溢。不是乘以许多小概率，而是将它们的对数相加：

```
log P(类别 | 特征) = log P(类别) + sum_i log P(feature_i | 类别)
```

## 构建它

`code/naive_bayes.py` 中的代码从零实现了 MultinomialNB 和 GaussianNB。

### MultinomialNB

从零实现：

1. **fit(X, y)**：对每个类别，计算每个特征的频率。添加拉普拉斯平滑。计算对数概率。存储类别先验（类别频率的对数）。

2. **predict_log_proba(X)**：对每个样本，计算所有类别的 log P(类别) + sum(log P(feature_i | 类别))。这是矩阵乘法：X @ log_probs.T + log_priors。

3. **predict(X)**：返回具有最高对数概率的类别。

```python
class MultinomialNB:
    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def fit(self, X, y):
        classes = np.unique(y)
        n_classes = len(classes)
        n_features = X.shape[1]

        self.classes_ = classes
        self.class_log_prior_ = np.zeros(n_classes)
        self.feature_log_prob_ = np.zeros((n_classes, n_features))

        for i, c in enumerate(classes):
            X_c = X[y == c]
            self.class_log_prior_[i] = np.log(X_c.shape[0] / X.shape[0])
            counts = X_c.sum(axis=0) + self.alpha
            self.feature_log_prob_[i] = np.log(counts / counts.sum())

        return self
```

关键洞察：拟合后，预测只是矩阵乘法加偏置。这就是朴素贝叶斯如此快速的原因。

### GaussianNB

对于连续特征，我们估计每个类别每个特征的均值和方差：

```python
class GaussianNB:
    def __init__(self):
        pass

    def fit(self, X, y):
        classes = np.unique(y)
        self.classes_ = classes
        self.means_ = np.zeros((len(classes), X.shape[1]))
        self.vars_ = np.zeros((len(classes), X.shape[1]))
        self.priors_ = np.zeros(len(classes))

        for i, c in enumerate(classes):
            X_c = X[y == c]
            self.means_[i] = X_c.mean(axis=0)
            self.vars_[i] = X_c.var(axis=0) + 1e-9
            self.priors_[i] = X_c.shape[0] / X.shape[0]

        return self
```

预测对每个特征使用高斯概率密度函数，跨特征相乘（在对数空间中相加）。

## 使用它

使用sklearn，两种变体都是一行代码：

```python
from sklearn.naive_bayes import GaussianNB, MultinomialNB

gnb = GaussianNB()
gnb.fit(X_train, y_train)
print(f"GaussianNB准确率: {gnb.score(X_test, y_test):.3f}")

mnb = MultinomialNB(alpha=1.0)
mnb.fit(X_train_counts, y_train)
print(f"MultinomialNB准确率: {mnb.score(X_test_counts, y_test):.3f}")
```

sklearn文本分类：

```python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

text_clf = Pipeline([
    ("vectorizer", CountVectorizer()),
    ("classifier", MultinomialNB(alpha=1.0)),
])

text_clf.fit(train_texts, train_labels)
accuracy = text_clf.score(test_texts, test_labels)
```

### 带朴素贝叶斯的TF-IDF

原始词计数对每次出现的每个词给予相同的权重。但像"the"和"is"这样的常见词在每个类别中都频繁出现——它们不携带任何信息。TF-IDF（词频-逆文档频率）降低常见词的权重，提高稀有、有判别力词的权重。

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

text_clf = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("classifier", MultinomialNB(alpha=0.1)),
])
```

TF-IDF值是非负的，所以它们适用于MultinomialNB。TF-IDF + MultinomialNB的组合是文本分类最强的基线之一。在训练样本少于10,000的数据集上，它经常胜过更复杂的模型。

### 短文本的BernoulliNB

对于短文本（推文、短信、聊天消息），BernoulliNB可能优于MultinomialNB。短文本词计数低，MultinomialNB依赖的频率信息很嘈杂。BernoulliNB只关心存在或不存在，对短文本更可靠。

```python
from sklearn.naive_bayes import BernoulliNB
from sklearn.feature_extraction.text import CountVectorizer

text_clf = Pipeline([
    ("vectorizer", CountVectorizer(binary=True)),
    ("classifier", BernoulliNB(alpha=1.0)),
])
```

CountVectorizer中的`binary=True`标志将所有计数转换为0/1。

### 何时朴素贝叶斯失败

当独立性假设导致错误排名时（不只是错误概率），朴素贝叶斯会失败：

1. **强特征交互。** 如果类别依赖于两个特征的组合而不是任何单独的特征（类似XOR模式），朴素贝叶斯会完全错过它。
2. **具有相反证据的高度相关特征。** 如果特征A说"垃圾邮件"而特征B说"非垃圾邮件"，但A和B完美相关，朴素贝叶斯会看到并不存在的矛盾证据。
3. **非常大的训练集。** 有了足够的数据，判别模型如逻辑回归会学到真正的决策边界并超越朴素贝叶斯。

## 练习

1. **平滑实验。** 用alpha值 0.01、0.1、1.0、10.0、100.0 训练MultinomialNB。绘制准确率 vs alpha。性能在哪里达到峰值？为什么非常高的alpha会有害？

2. **特征独立性测试。** 取一个真实的文本数据集。挑选两个明显相关的词（"machine"和"learning"）。计算 P(词1|类别) * P(词2|类别) 并与 P(词1 AND 词2|类别) 比较。独立性假设有多错？它影响分类准确率吗？

3. **伯努利实现。** 用BernoulliNB类扩展代码。将词袋转换为二元（存在/不存在），并在文本数据上与MultinomialNB比较准确率。何时伯努利胜出？

4. **朴素贝叶斯 vs 逻辑回归。** 在文本数据上训练两者。从100个训练样本开始增加到10,000。绘制两者的准确率 vs 训练集大小。在什么时候逻辑回归超过朴素贝叶斯？

5. **垃圾邮件过滤器。** 构建完整的垃圾邮件分类器：对原始邮件文本分词、构建词汇表、创建词袋特征、训练MultinomialNB、用精确率和召回率评估（而不只是准确率——为什么？）。

## 关键术语

| 术语 | 人们说的 | 实际含义 |
|------|---------|---------|
| 朴素贝叶斯（Naive Bayes） | "简单的概率分类器" | 应用贝叶斯定理并假设给定类别特征条件独立的分类器 |
| 条件独立（Conditional independence） | "特征互不影响" | P(A, B \| C) = P(A \| C) * P(B \| C) —— 一旦知道C，知道B不会给你关于A的新信息 |
| 拉普拉斯平滑（Laplace smoothing） | "加一平滑" | 对每个特征添加小计数以防止零概率主导预测 |
| 先验（Prior） | "看数据之前你相信什么" | P(类别) —— 在观察任何特征之前每个类别的概率 |
| 似然（Likelihood） | "数据的拟合程度" | P(特征 \| 类别) —— 如果已知类别，观察到这些特征的概率 |
| 后验（Posterior） | "看数据后你相信什么" | P(类别 \| 特征) —— 观察特征后类别的更新概率 |
| 生成式模型（Generative model） | "建模数据如何生成" | 学习P(X \| Y)和P(Y)，然后用贝叶斯定理得到P(Y \| X)的模型 |
| 判别式模型（Discriminative model） | "建模决策边界" | 直接学习P(Y \| X)而不建模X如何生成的模型 |
| 对数概率（Log probability） | "避免下溢" | 使用 log P 而不是 P 来防止许多小数的乘积在浮点数中变为零 |

## 延伸阅读

- [scikit-learn 朴素贝叶斯文档](https://scikit-learn.org/stable/modules/naive_bayes.html) -- 三种变体及数学细节
- [McCallum and Nigam, A Comparison of Event Models for Naive Bayes Text Classification (1998)](https://www.cs.cmu.edu/~knigam/papers/multinomial-aaaiws98.pdf) -- 多项式vs伯努利文本分类的经典比较
- [Rennie et al., Tackling the Poor Assumptions of Naive Bayes Text Classifiers (2003)](https://people.csail.mit.edu/jrennie/papers/icml03-nb.pdf) -- 文本朴素贝叶斯的改进
- [Ng and Jordan, On Discriminative vs. Generative Classifiers (2001)](https://ai.stanford.edu/~ang/papers/nips01-discriminativegenerative.pdf) -- 证明朴素贝叶斯用更少数据比逻辑回归收敛更快
