# 07 - 交叉熵损失 (Cross-Entropy Loss)

## 概念介绍

交叉熵损失衡量预测分布与真实分布之间的差异，是语言模型训练的核心损失函数。

**为什么用交叉熵？**
- 语言模型的任务是预测下一个 token 的概率分布
- 交叉熵衡量"预测分布"与"真实分布"的距离
- 最小化交叉熵 = 最大化真实标签的预测概率

**类比：**
- 想象天气预报：实际是 [晴, 雨, 阴] = [1, 0, 0]
- 预报 A: [0.7, 0.2, 0.1] → 交叉熵 = 0.36（较准）
- 预报 B: [0.1, 0.8, 0.1] → 交叉熵 = 2.30（不准）
- 交叉熵越小，预测越准

## 数学原理

### 交叉熵公式

对于离散分布 $P$（真实）和 $Q$（预测）：

$$H(P, Q) = -\sum_{i} P(i) \log Q(i)$$

### 单标签分类

当真实标签是类别 $y$（one-hot 向量中只有 $y$ 位置为 1）：

$$\text{CE} = -\log Q(y) = -\log \text{softmax}(\text{logits})_y$$

### 与 KL 散度的关系

$$H(P, Q) = H(P) + D_{KL}(P \| Q)$$

- $H(P)$：真实分布的熵（常数）
- $D_{KL}(P \| Q)$：KL 散度，衡量分布差异

最小化交叉熵 ≈ 最小化 KL 散度 ≈ 让预测分布接近真实分布

### 困惑度 (Perplexity)

$$\text{PPL} = e^{H(P, Q)}$$

直观理解：模型在每个位置平均有多少个等可能的选择
- PPL = 1：完美预测
- PPL = 10：平均有 10 个候选
- PPL = 50000：接近随机猜测（词汇表大小）

## 代码说明

### 核心实现

```python
def cross_entropy_loss(logits, targets, reduction='mean'):
    # 计算 log_softmax（数值稳定）
    log_probs = log_softmax(logits, axis=-1)

    # 取目标位置的 log 概率
    loss = -log_probs[..., targets]

    # 应用 reduction
    if reduction == 'mean':
        return np.mean(loss)
```

### 数值稳定性

使用 log_softmax 而非 log(softmax)：
- `log(softmax(x))` 可能出现 `log(0)` → -inf
- `log_softmax(x)` 使用 log-sum-exp 技巧，更稳定

## 使用示例

```python
import numpy as np
from cross_entropy import cross_entropy_loss, perplexity

# 模拟 3 分类问题
logits = np.array([2.0, 1.0, 0.5])
target = 0  # 真实类别

# 计算损失
loss = cross_entropy_loss(logits, target)
print(f"损失: {loss:.4f}")

# 计算困惑度
ppl = perplexity(loss)
print(f"困惑度: {ppl:.2f}")

# 批量处理
logits_batch = np.array([[2.0, 1.0], [0.5, 2.0]])
targets_batch = np.array([0, 1])
loss_batch = cross_entropy_loss(logits_batch, targets_batch)
```

## 可视化

```
交叉熵计算流程:

logits:      [2.0, 1.0, 0.5]
    ↓
softmax:     [0.59, 0.24, 0.17]  (概率分布)
    ↓
target = 0:  取第 0 个概率 0.59
    ↓
-log:        -log(0.59) = 0.53
    ↓
损失:        0.53

困惑度:      exp(0.53) = 1.70
含义:        模型平均有 1.7 个等可能选择
```

## 与其他模块的关系

- **输入**：来自模型输出层的 logits
- **目标**：来自 tokenization 的真实 token IDs
- **输出**：标量损失值，用于反向传播
- **训练**：最小化交叉熵 → 提高预测准确率

## 延伸阅读

- [交叉熵 - 维基百科](https://en.wikipedia.org/wiki/Cross_entropy)
- [KL 散度 - 维基百科](https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence)
