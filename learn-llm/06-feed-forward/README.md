# 06 - 前馈网络 (Feed-Forward Network)

## 概念介绍

前馈网络（FFN）是 Transformer 中的"记忆"组件，存储和转换知识。

**为什么需要 FFN？**
- 注意力层负责"信息聚合"（从其他 token 收集信息）
- FFN 负责"信息转换"（对每个 token 独立处理）
- 两者配合，实现"知道什么"和"怎么用"

**类比：**
- 注意力层像"查阅资料"：从文档中找到相关信息
- FFN 像"思考整理"：对找到的信息进行理解和加工

## 数学原理

### FFN 结构

$$\text{FFN}(x) = W_2 \cdot \text{GELU}(W_1 \cdot x + b_1) + b_2$$

其中：
- $W_1 \in \mathbb{R}^{d \times 4d}$：第一个线性层（扩展维度）
- $W_2 \in \mathbb{R}^{4d \times d}$：第二个线性层（压缩回原维度）
- GELU：激活函数

### 维度变化

```
输入: [batch, seq_len, hidden_size]
  ↓ W1: hidden_size → 4 * hidden_size
中间: [batch, seq_len, 4 * hidden_size]
  ↓ GELU
中间: [batch, seq_len, 4 * hidden_size]
  ↓ W2: 4 * hidden_size → hidden_size
输出: [batch, seq_len, hidden_size]
```

### 为什么中间维度是 4 倍？

- 原始 Transformer 论文中使用 4 倍扩展
- 更大的中间层提供更强的表达能力
- 现代模型（如 LLaMA）使用不同的比例（如 2.7 倍）

### GELU 激活函数

$$\text{GELU}(x) = 0.5x\left(1 + \tanh\left(\sqrt{\frac{2}{\pi}}(x + 0.044715x^3)\right)\right)$$

**与 ReLU 的区别：**
- ReLU：硬截断，x < 0 时梯度为 0
- GELU：平滑过渡，x < 0 时仍有小梯度
- GELU 更适合 Transformer，因为它允许小的负值通过

## 代码说明

### 核心实现

```python
def forward(self, x):
    # 第一层: hidden_size → ffn_size
    h = x @ self.W1 + self.b1

    # 激活函数: GELU
    h = gelu(h)

    # 第二层: ffn_size → hidden_size
    output = h @ self.W2 + self.b2

    return output
```

### 参数初始化

使用 He 初始化（适配 GELU 激活）：
```python
self.W1 = np.random.randn(hidden_size, ffn_size) * np.sqrt(2.0 / hidden_size)
```

## 使用示例

```python
import numpy as np
from feed_forward import FeedForward

# 创建前馈网络
ffn = FeedForward(hidden_size=64, ffn_size=256)

# 输入：batch_size=2, seq_len=10
x = np.random.randn(2, 10, 64)

# 前向传播
output = ffn(x)
print(output.shape)  # (2, 10, 64)
```

## 可视化

```
FFN 数据流:

输入 x:     [0.5, -0.3, 0.8, ...]  (hidden_size=64)
    ↓
W1 扩展:    [0.2, 0.1, -0.4, ...]  (ffn_size=256)
    ↓
GELU:       [0.2, 0.1, 0.0, ...]   (负值被抑制)
    ↓
W2 压缩:    [0.3, -0.1, 0.5, ...]  (hidden_size=64)
    ↓
输出:       与输入形状相同
```

## 与其他模块的关系

- **输入**：来自注意力层的输出（或嵌入向量）
- **输出**：传给下一层的注意力或最终输出
- **位置**：在 Transformer 中紧跟在注意力层之后

## 参数数量分析

假设 $d = 768$（如 GPT-2 small）：
- $W_1$: $768 \times 3072 = 2,359,296$
- $b_1$: $3072$
- $W_2$: $3072 \times 768 = 2,359,296$
- $b_2$: $768$
- **总计**: $4,722,432$ 参数（约 4.7M）

每层 FFN 的参数量约等于注意力层的 2 倍！

## 延伸阅读

- [GELU 论文](https://arxiv.org/abs/1606.08415) - GELU 激活函数
- [GLU Variants](https://arxiv.org/abs/2002.05202) - 现代 FFN 变体（如 SwiGLU）
