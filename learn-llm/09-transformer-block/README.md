# 09 - Transformer 块 (Transformer Block)

## 概念介绍

Transformer 块是 Transformer 的基本构建单元，组合了多头注意力和前馈网络。

**为什么需要 Transformer 块？**
- 单独的注意力层或 FFN 能力有限
- 堆叠多个 Transformer 块可以学习更复杂的模式
- 每一层逐步提炼信息，从简单到复杂

**类比：**
- 想象一个流水线工厂
- 每个 Transformer 块是一个加工站
- 原材料（输入）经过多个加工站，逐步变成成品（输出）

## 数学原理

### Pre-Norm 结构（现代 GPT 使用）

$$x = x + \text{Attention}(\text{LN}(x))$$
$$x = x + \text{FFN}(\text{LN}(x))$$

**与 Post-Norm 的区别：**

| 结构 | 公式 | 特点 |
|------|------|------|
| Post-Norm | $x = \text{LN}(x + \text{SubLayer}(x))$ | 原始 Transformer，训练不稳定 |
| Pre-Norm | $x = x + \text{SubLayer}(\text{LN}(x))$ | 现代 GPT，训练更稳定 |

### 残差连接

$$\text{output} = x + \text{sublayer}(x)$$

**为什么需要残差连接？**
- 解决梯度消失问题：梯度可以直接流过残差连接
- 保留原始信息：即使子层输出很小，原始信息也能保留
- 便于训练深层网络：几十层甚至上百层都能训练

### 完整流程

```
输入 x
  ↓
LayerNorm → MultiHeadAttention → + x (残差)
  ↓
LayerNorm → FeedForward → + x (残差)
  ↓
输出
```

## 代码说明

### 核心实现

```python
def forward(self, x, use_causal_mask=False):
    # 1. 注意力子层（Pre-Norm + 残差）
    residual = x
    x = self.ln1(x)
    x = self.attention(x, use_causal_mask=use_causal_mask)
    x = x + residual

    # 2. 前馈子层（Pre-Norm + 残差）
    residual = x
    x = self.ln2(x)
    x = self.ffn(x)
    x = x + residual

    return x
```

### 关键点

1. **Pre-Norm**：LayerNorm 在子层之前
2. **残差连接**：`x + sublayer(x)`
3. **因果掩码**：传递给注意力层

## 使用示例

```python
import numpy as np
from transformer_block import TransformerBlock

# 创建 Transformer 块
block = TransformerBlock(hidden_size=64, num_heads=8)

# 输入：batch_size=2, seq_len=10
x = np.random.randn(2, 10, 64)

# 前向传播（带因果掩码，用于解码器）
output = block(x, use_causal_mask=True)
print(output.shape)  # (2, 10, 64)

# 多层堆叠
layers = [TransformerBlock(64, 8) for _ in range(6)]
h = x
for layer in layers:
    h = layer(h, use_causal_mask=True)
```

## 可视化

```
Transformer 块数据流:

输入 x:              [B, S, H]
    ↓
┌─────────────────────────────────────┐
│ 注意力子层                          │
│   LN(x) → MultiHeadAttn → + x     │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ 前馈子层                            │
│   LN(x) → FFN → + x               │
└─────────────────────────────────────┘
    ↓
输出:                [B, S, H]

多层堆叠:
Layer 1 → Layer 2 → ... → Layer N → 最终输出
```

## 与其他模块的关系

- **依赖**：`05-layernorm`, `06-feed-forward`, `08-multi-head-attention`
- **输入**：来自 `02-embedding` + `03-positional-encoding`
- **输出**：传给下一层 Transformer 块或最终输出层
- **组合**：在 `10-gpt-model` 中堆叠多层

## 参数数量分析

假设 $d = 768$，$h = 12$：
- LayerNorm × 2：$768 \times 2 \times 2 = 3,072$
- Multi-Head Attention：$4 \times 768^2 = 2,359,296$
- Feed-Forward：$2 \times 768 \times 3072 + 3072 + 768 = 4,722,432$
- **单层总计**：$7,084,800$（约 7M）

GPT-2 small 有 12 层，总参数约 85M！

## 延伸阅读

- [Pre-LN Transformer](https://arxiv.org/abs/2002.04745) - Pre-Norm 结构分析
- [GPT-2](https://openai.com/blog/better-language-models/) - 使用 Pre-Norm 的 GPT
