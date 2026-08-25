# 02 - 词嵌入 (Token Embedding)

## 概念介绍

词嵌入是 LLM 的第一步：将离散的 token ID 转换为连续的向量表示。

**为什么需要词嵌入？**
- Token ID 是离散的整数（如 101, 345, 782），没有语义信息
- 神经网络需要连续的数值输入才能进行计算
- 嵌入向量可以捕获词与词之间的语义关系

**类比：**
- 想象一本字典，每个词有一个页码（token ID）
- 嵌入矩阵就是字典的内容，每一页记录了该词的"特征向量"
- 查找过程就是翻到对应页码，取出向量

## 数学原理

### 嵌入层的数学本质

嵌入层本质上是一个**查找表**（lookup table），不是矩阵乘法：

```
输入: token_id = 3
嵌入矩阵 W: [vocab_size, hidden_size]
输出: W[3] （直接取第3行）
```

**形式化定义：**

给定：
- 词汇表大小 $V$
- 嵌入维度 $d$
- 嵌入矩阵 $\mathbf{W} \in \mathbb{R}^{V \times d}$
- 输入 token ID $x \in \{0, 1, ..., V-1\}$

嵌入操作：
$$\text{Embedding}(x) = \mathbf{W}[x]$$

对于批量输入 $\mathbf{X} \in \mathbb{R}^{B \times S}$（B 为 batch_size，S 为 seq_len）：
$$\text{Embedding}(\mathbf{X}) \in \mathbb{R}^{B \times S \times d}$$

### 为什么不用 one-hot + 矩阵乘法？

理论上，嵌入等价于 one-hot 编码乘以权重矩阵：
$$\text{one\_hot}(x) \cdot \mathbf{W} = \mathbf{W}[x]$$

但直接索引比矩阵乘法高效得多，所以实际实现都用查找表。

## 代码说明

### 核心实现

```python
class Embedding:
    def __init__(self, vocab_size, hidden_size):
        # 嵌入矩阵：随机初始化
        self.weight = np.random.randn(vocab_size, hidden_size)

    def forward(self, input_ids):
        # 核心操作：矩阵索引
        return self.weight[input_ids]
```

### 关键点

1. **权重初始化**：使用正态分布 $N(0, 1)$，实际训练中会通过梯度下降学习
2. **索引操作**：`self.weight[input_ids]` 利用 NumPy 的高级索引，一次操作完成所有查找
3. **形状变换**：输入 `[B, S]` → 输出 `[B, S, d]`，增加了嵌入维度

## 使用示例

```python
import numpy as np
from embedding import Embedding

# 创建嵌入层
embedding = Embedding(vocab_size=1000, hidden_size=64)

# 输入：batch_size=2, seq_len=10
input_ids = np.array([[1, 3, 5], [2, 4, 6]])

# 前向传播
output = embedding(input_ids)
print(output.shape)  # (2, 3, 64)

# 查看第一个 token 的嵌入向量
print(output[0, 0])  # 与 embedding.weight[1] 相同
```

## 可视化

```
Token IDs:        嵌入向量:
┌───┬───┬───┐     ┌─────────────────────┐
│ 1 │ 3 │ 5 │     │ [0.12, -0.34, ...] │  ← token 1 的向量
│ 2 │ 4 │ 6 │     │ [0.56, 0.78, ...]  │  ← token 3 的向量
└───┴───┴───┘     └─────────────────────┘
     ↓                   ↓
  [2, 3]            [2, 3, 64]
```

## 与其他模块的关系

- **输入**：来自 `01-tokenization` 的 token IDs
- **输出**：传给 `03-positional-encoding` 添加位置信息
- **最终**：嵌入向量进入 Transformer 层进行注意力计算

## 延伸阅读

- [Word2Vec](https://arxiv.org/abs/1301.3781) - 经典词嵌入方法
- [GloVe](https://nlp.stanford.edu/projects/glove/) - 另一种词嵌入方法
- 现代 LLM 的嵌入层是可训练的，与模型一起端到端学习
