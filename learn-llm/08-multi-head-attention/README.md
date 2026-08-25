# 08 - 多头注意力 (Multi-Head Attention)

## 概念介绍

多头注意力是 Transformer 的核心机制：让每个 token 能够"关注"其他相关 token。

**为什么需要注意力？**
- 语言中词的含义依赖于上下文
- "苹果很好吃" vs "苹果发布新手机" 中的"苹果"含义不同
- 注意力机制让模型动态地从上下文中获取信息

**类比：**
- 想象阅读一篇文章，遇到不认识的词
- 你会"回头看"相关的句子来理解
- 注意力机制就是这种"回头看"的过程

## 数学原理

### 缩放点积注意力

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

其中：
- $Q$（Query）：查询向量，"我想找什么"
- $K$（Key）：键向量，"我有什么"
- $V$（Value）：值向量，"我的内容是什么"
- $d_k$：键向量的维度（缩放因子）

### 多头机制

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, ..., \text{head}_h)W^O$$
$$\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$$

**为什么要多头？**
- 单头只能学习一种注意力模式
- 多头让模型从不同角度理解关系：
  - 头 1：关注语法关系（主语-谓语）
  - 头 2：关注语义关系（同义词）
  - 头 3：关注位置关系（相邻词）

### 因果掩码

在解码器中，token 不能看到未来的信息：

```
位置:    0  1  2  3
  0    [ 1  0  0  0 ]  ← 位置 0 只能看自己
  1    [ 1  1  0  0 ]  ← 位置 1 能看 0, 1
  2    [ 1  1  1  0 ]  ← 位置 2 能看 0, 1, 2
  3    [ 1  1  1  1 ]  ← 位置 3 能看所有
```

掩码为 0 的位置设为 $-\infty$，softmax 后变成 0。

### 为什么除以 $\sqrt{d_k}$？

- 当 $d_k$ 很大时，$QK^T$ 的值会很大
- 大值导致 softmax 接近 one-hot，梯度消失
- 除以 $\sqrt{d_k}$ 保持方差稳定

## 代码说明

### 核心流程

```python
def forward(self, x):
    # 1. Q, K, V 投影
    Q = x @ self.W_q
    K = x @ self.W_k
    V = x @ self.W_v

    # 2. 分割成多个头
    Q = self._split_heads(Q)  # [B, num_heads, S, head_dim]
    K = self._split_heads(K)
    V = self._split_heads(V)

    # 3. 计算注意力分数
    scores = (Q @ K.T) / sqrt(head_dim)

    # 4. 应用掩码（可选）
    if use_causal_mask:
        scores = np.where(mask == 0, -1e9, scores)

    # 5. Softmax 归一化
    weights = softmax(scores)

    # 6. 加权求和
    output = weights @ V

    # 7. 合并头 + 输出投影
    output = self._merge_heads(output) @ self.W_o
```

### 分头操作

```python
# [B, S, H] → [B, S, num_heads, head_dim] → [B, num_heads, S, head_dim]
x = x.reshape(B, S, num_heads, head_dim).transpose(0, 2, 1, 3)
```

## 使用示例

```python
import numpy as np
from multi_head_attention import MultiHeadAttention

# 创建多头注意力
mha = MultiHeadAttention(hidden_size=64, num_heads=8)

# 输入：batch_size=2, seq_len=10
x = np.random.randn(2, 10, 64)

# 前向传播（带因果掩码，用于解码器）
output = mha(x, use_causal_mask=True)
print(output.shape)  # (2, 10, 64)
```

## 可视化

```
多头注意力流程:

输入 x:           [B, S, H]
    ↓
Q, K, V 投影:     [B, S, H] × 3
    ↓
分头:             [B, num_heads, S, head_dim] × 3
    ↓
注意力分数:       Q @ K^T / √d_k  →  [B, num_heads, S, S]
    ↓
掩码（可选）:     将未来位置设为 -inf
    ↓
Softmax:          [B, num_heads, S, S]（每行和为 1）
    ↓
加权求和:         weights @ V  →  [B, num_heads, S, head_dim]
    ↓
合并头:           [B, S, H]
    ↓
输出投影:         [B, S, H]
```

## 与其他模块的关系

- **输入**：来自 `02-embedding` + `03-positional-encoding` 的向量
- **内部使用**：`04-softmax` 进行归一化
- **输出**：传给 `06-feed-forward` 或下一层注意力
- **组合**：在 `09-transformer-block` 中与 FFN 组合

## 参数数量分析

假设 $d = 768$（如 GPT-2 small）：
- $W_q, W_k, W_v, W_o$：各 $768 \times 768 = 589,824$
- **总计**: $4 \times 589,824 = 2,359,296$（约 2.4M）

## 延伸阅读

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - 原始 Transformer 论文
- [FlashAttention](https://arxiv.org/abs/2205.14135) - 高效注意力实现
- [Multi-Query Attention](https://arxiv.org/abs/1911.02150) - 现代优化变体
