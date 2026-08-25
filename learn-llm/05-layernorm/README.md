# 05 - 层归一化 (Layer Normalization)

## 概念介绍

层归一化对每个样本的最后一维进行归一化，稳定训练过程。

**为什么需要 LayerNorm？**
- 深层网络中，每一层的输入分布会不断变化（内部协变量偏移）
- 这导致训练不稳定，需要更小的学习率
- LayerNorm 将每层的输入归一化到稳定分布，加速训练

**类比：**
- 想象一个班级的考试成绩，平均分可能是 50-90 不等
- LayerNorm 将成绩标准化：减去平均分，除以标准差
- 这样每次考试的成绩都在相似的分布上

## 数学原理

### LayerNorm 公式

$$\text{LN}(x) = \gamma \odot \frac{x - \mu}{\sigma + \epsilon} + \beta$$

其中：
- $\mu = \frac{1}{d}\sum_{i=1}^{d} x_i$：均值
- $\sigma^2 = \frac{1}{d}\sum_{i=1}^{d} (x_i - \mu)^2$：方差
- $\gamma$：可学习的缩放参数（初始化为 1）
- $\beta$：可学习的偏移参数（初始化为 0）
- $\epsilon$：防止除零的小常数（通常 1e-5）

### 与 BatchNorm 的区别

| 特性 | LayerNorm | BatchNorm |
|------|-----------|-----------|
| 归一化维度 | 每个样本的最后一维 | 每个特征的 batch 维度 |
| 依赖 batch | 不依赖 | 依赖 batch 统计 |
| 推理时 | 与训练相同 | 使用移动平均 |
| 适用场景 | NLP、Transformer | CV、CNN |

### 为什么 Transformer 用 LayerNorm？

1. **序列长度可变**：BatchNorm 需要固定 batch 统计，不适合变长序列
2. **独立性**：每个样本独立归一化，不依赖其他样本
3. **训练稳定**：更适合自回归生成任务

## 代码说明

### 核心实现

```python
def forward(self, x):
    # 计算均值和方差（沿最后一维）
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)

    # 归一化
    x_norm = (x - mean) / np.sqrt(var + self.eps)

    # 仿射变换
    return self.gamma * x_norm + self.beta
```

### 关键点

1. **axis=-1**：沿最后一维计算统计量
2. **keepdims=True**：保持维度便于广播
3. **可学习参数**：gamma 和 beta 让模型学习最优的缩放和偏移

## 使用示例

```python
import numpy as np
from layernorm import LayerNorm

# 创建 LayerNorm
ln = LayerNorm(hidden_size=64)

# 输入：batch_size=2, seq_len=10
x = np.random.randn(2, 10, 64)

# 归一化
output = ln(x)

# 验证：均值接近 0，方差接近 1
print(output.mean(axis=-1))  # ≈ 0
print(output.var(axis=-1))   # ≈ 1
```

## 可视化

```
LayerNorm 计算流程:

输入 x:           [3.0, 1.0, 4.0, 1.0, 5.0]
                    ↓
计算均值 μ:       (3+1+4+1+5)/5 = 2.8
                    ↓
计算方差 σ²:      [(3-2.8)² + ... + (5-2.8)²]/5 = 2.16
                    ↓
归一化:           [(3-2.8)/√2.16, ...] = [0.14, -1.22, 0.82, -1.22, 1.49]
                    ↓
仿射变换:         γ * normalized + β
                    ↓
输出:             [0.14, -1.22, 0.82, -1.22, 1.49] (当 γ=1, β=0)
```

## 与其他模块的关系

- **输入**：嵌入向量或上一层的输出
- **输出**：归一化后的向量，传给注意力层或前馈网络
- **位置**：在 Transformer 中通常放在注意力层和前馈网络之前（Pre-Norm）

## 延伸阅读

- [Layer Normalization](https://arxiv.org/abs/1607.06450) - 原始论文
- [RMSNorm](https://arxiv.org/abs/1910.07467) - 简化版本（LLaMA 使用）
