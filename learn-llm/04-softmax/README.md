# 04 - Softmax 函数

## 概念介绍

Softmax 将任意实数向量转换为概率分布，是注意力机制和分类任务的核心组件。

**为什么需要 Softmax？**
- 神经网络输出的是任意实数（logits），不是概率
- 概率需要满足：非负、和为 1
- Softmax 将 logits 转换为满足这些条件的概率分布

**类比：**
- 想象一场比赛，选手得分是 [85, 92, 78]
- Softmax 将得分转换为获奖概率：[0.23, 0.59, 0.18]
- 得分最高的选手获得最大概率

## 数学原理

### Softmax 公式

$$\text{softmax}(x_i) = \frac{e^{x_i}}{\sum_{j=1}^{n} e^{x_j}}$$

### 数值稳定性

直接计算 $e^{x_i}$ 会溢出（如 $x=1000$ 时 $e^{1000}$ 是天文数字）。

**技巧：减去最大值**

$$\text{softmax}(x_i) = \frac{e^{x_i - \max(x)}}{\sum_{j} e^{x_j - \max(x)}}$$

数学上等价，但数值更稳定（指数参数 ≤ 0）。

### Log-Softmax

交叉熵损失中常用 log 概率，直接计算更稳定：

$$\text{log\_softmax}(x_i) = x_i - \log\sum_{j} e^{x_j}$$

使用 log-sum-exp 技巧：

$$\text{log\_softmax}(x_i) = (x_i - \max(x)) - \log\sum_{j} e^{x_j - \max(x)}$$

### 温度参数

$$\text{softmax}(x_i / T)$$

- $T < 1$：分布更尖锐（更确定）
- $T = 1$：标准 softmax
- $T > 1$：分布更平滑（更随机）

## 代码说明

### 核心实现

```python
def softmax(x, axis=-1):
    # 减去最大值，防止溢出
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)

    # 归一化
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)
```

### 关键点

1. **keepdims=True**：保持维度，便于广播
2. **axis 参数**：指定计算 softmax 的维度
3. **数值稳定**：减去最大值是标准做法

## 使用示例

```python
import numpy as np
from softmax import softmax, log_softmax

# 基本使用
logits = np.array([1.0, 2.0, 3.0])
probs = softmax(logits)
print(probs)  # [0.09, 0.24, 0.67]
print(probs.sum())  # 1.0

# 批量处理
logits_batch = np.array([[1.0, 2.0], [3.0, 4.0]])
probs_batch = softmax(logits_batch, axis=-1)

# Log-Softmax
log_probs = log_softmax(logits)
```

## 可视化

```
Softmax 转换:

输入 (logits):    [1.0,  2.0,  3.0 ]
                    ↓
exp:              [2.7,  7.4, 20.1]
                    ↓
归一化:           [0.09, 0.24, 0.67]
                    ↓
概率分布:         和为 1.0 ✓

温度效果:
T=0.5:  [0.01, 0.12, 0.87]  ← 更尖锐
T=1.0:  [0.09, 0.24, 0.67]  ← 标准
T=2.0:  [0.19, 0.30, 0.51]  ← 更平滑
```

## 与其他模块的关系

- **注意力机制**：`softmax(Q @ K^T / sqrt(d))` 将注意力分数转为权重
- **输出层**：`softmax(logits)` 生成下一个 token 的概率分布
- **交叉熵损失**：内部使用 log_softmax 计算损失

## 延伸阅读

- [Softmax 函数](https://en.wikipedia.org/wiki/Softmax_function) - 维基百科
- [Log-Sum-Exp 技巧](https://en.wikipedia.org/wiki/LogSumExp) - 数值计算技巧
