# 03 - 位置编码 (Positional Encoding)

## 概念介绍

位置编码为输入序列添加位置信息，使模型能够感知 token 的顺序。

**为什么需要位置编码？**
- Transformer 的注意力机制是**位置无关**的：打乱 token 顺序，注意力计算结果相同
- 语言是有序的："猫吃鱼"和"鱼吃猫"含义不同
- 位置编码告诉模型每个 token 在序列中的位置

**类比：**
- 想象一排座位，每个座位上贴了一个独特的标签
- 位置编码就是这些标签，让模型知道"谁坐在哪里"

## 数学原理

### 正弦位置编码公式

$$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d}}\right)$$

$$PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d}}\right)$$

其中：
- $pos$：位置索引（0, 1, 2, ...）
- $i$：维度索引（0, 1, 2, ..., d/2-1）
- $d$：嵌入维度

### 为什么用正弦/余弦？

1. **唯一性**：每个位置有独特的编码向量
2. **有界性**：值在 [-1, 1] 之间，不会爆炸
3. **相对位置**：$PE_{pos+k}$ 可以表示为 $PE_{pos}$ 的线性函数，便于学习相对位置关系
4. **泛化性**：可以处理比训练时更长的序列

### 计算示例

假设 $d=4$，计算位置 0 和 1 的编码：

```
维度分母: [10000^0, 10000^1] = [1, 10000]

位置 0:
  sin(0/1) = 0,     cos(0/1) = 1
  sin(0/10000) = 0, cos(0/10000) = 1
  PE_0 = [0, 1, 0, 1]

位置 1:
  sin(1/1) = 0.841, cos(1/1) = 0.540
  sin(1/10000) ≈ 0, cos(1/10000) ≈ 1
  PE_1 = [0.841, 0.540, 0.0001, 1]
```

## 代码说明

### 核心实现

```python
def _build_position_encoding(self):
    # 位置索引
    position = np.arange(max_seq_len)[:, np.newaxis]

    # 维度分母（使用 log 空间避免溢出）
    div_term = np.exp(np.arange(0, hidden_size, 2) * -(np.log(10000.0) / hidden_size))

    # 计算 sin 和 cos
    pe[:, 0::2] = np.sin(position * div_term)
    pe[:, 1::2] = np.cos(position * div_term)
```

### 关键技巧

1. **log 空间计算**：`exp(-log(10000) * 2i/d)` 比直接计算 `1/10000^(2i/d)` 更稳定
2. **预计算**：位置编码矩阵只计算一次，前向传播时直接查表
3. **广播机制**：`x + pe[:seq_len]` 自动扩展到 batch 维度

## 使用示例

```python
import numpy as np
from positional_encoding import PositionalEncoding

# 创建位置编码层
pe = PositionalEncoding(hidden_size=64, max_seq_len=100)

# 输入：batch_size=2, seq_len=10
x = np.zeros((2, 10, 64))

# 添加位置编码
output = pe(x)
print(output.shape)  # (2, 10, 64)
```

## 可视化

```
位置编码热力图（简化示例）:

位置\维度   0    1    2    3
   0      0.0  1.0  0.0  1.0
   1      0.8  0.5  0.0  1.0
   2      0.9 -0.4  0.0  1.0
   3      0.1 -0.9  0.0  1.0
   ...

特点:
- 低维度（0,1）变化快 → 捕捉局部位置
- 高维度（2,3）变化慢 → 捕捉全局位置
```

## 与其他模块的关系

- **输入**：来自 `02-embedding` 的嵌入向量
- **输出**：带位置信息的向量，传给注意力层
- **作用**：让注意力机制能够区分不同位置的 token

## 延伸阅读

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - 原始 Transformer 论文
- [RoPE](https://arxiv.org/abs/2104.09864) - 旋转位置编码（现代 LLM 常用）
- [ALiBi](https://arxiv.org/abs/2108.12409) - 另一种位置编码方法
