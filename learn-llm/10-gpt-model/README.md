# 10 - GPT 模型 (Generative Pre-trained Transformer)

## 概念介绍

GPT 是一个完整的语言模型，组合了前面所有模块，可以训练和生成文本。

**GPT 的核心思想：**
- 预测下一个 token：给定前文，预测下一个最可能的词
- 自回归生成：逐步生成，每一步都基于已生成的内容
- 通过大量文本学习语言模式

**类比：**
- 想象一个"完形填空"游戏
- GPT 学习了大量文本后，能够预测缺失的词
- 生成时，它不断预测下一个词，逐步构建完整的句子

## 数学原理

### GPT 架构

```
输入 token IDs
    ↓
Token Embedding + Positional Embedding
    ↓
N × TransformerBlock (with causal mask)
    ↓
LayerNorm
    ↓
LM Head (Linear → vocab_size)
    ↓
输出 logits
```

### 训练目标

最大化下一个 token 的预测概率：

$$\mathcal{L} = -\sum_{t} \log P(x_t | x_{<t})$$

等价于最小化交叉熵损失。

### 生成过程

自回归生成：

$$P(x_1, x_2, ..., x_n) = \prod_{t=1}^{n} P(x_t | x_{<t})$$

每一步：
1. 输入已生成的 token
2. 获取下一个 token 的概率分布
3. 采样（或取 argmax）
4. 拼接到序列末尾
5. 重复

### 采样策略

**Temperature：**
$$P'(x) = \frac{e^{\log P(x) / T}}{\sum e^{\log P(x) / T}}$$

- $T < 1$：更确定（贪婪）
- $T > 1$：更随机

**Top-k 采样：**
只从概率最高的 k 个 token 中采样。

## 代码说明

### 模型结构

```python
class GPT(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_layers, num_heads):
        # Token + Position Embedding
        self.token_embedding = nn.Embedding(vocab_size, hidden_size)
        self.position_embedding = nn.Embedding(max_seq_len, hidden_size)

        # Transformer 层
        self.layers = nn.ModuleList([
            TransformerBlock(hidden_size, num_heads)
            for _ in range(num_layers)
        ])

        # 最终 LayerNorm + LM Head
        self.ln_final = nn.LayerNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

        # 权重绑定
        self.lm_head.weight = self.token_embedding.weight
```

### 前向传播

```python
def forward(self, input_ids, targets=None):
    # Embedding
    x = token_embedding(input_ids) + position_embedding(positions)

    # Transformer 层
    for layer in self.layers:
        x = layer(x, use_causal_mask=True)

    # LM Head
    logits = self.lm_head(self.ln_final(x))

    # 计算损失
    loss = F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1))

    return logits, loss
```

### 生成

```python
def generate(self, input_ids, max_new_tokens, temperature, top_k):
    for _ in range(max_new_tokens):
        # 前向传播
        logits, _ = self(input_ids[:, -max_seq_len:])

        # 温度 + Top-k 采样
        logits = logits[:, -1] / temperature
        probs = softmax(logits)
        next_token = multinomial(probs)

        # 拼接
        input_ids = cat([input_ids, next_token])

    return input_ids
```

## 使用示例

### 训练

```python
from gpt_model import GPT
from train import train

# 训练模型
train()
```

### 生成

```python
import torch
from gpt_model import GPT

# 加载模型
model = GPT(vocab_size=50, hidden_size=128, num_layers=4, num_heads=4)
model.load_state_dict(torch.load("gpt_model.pth"))

# 生成
input_ids = torch.tensor([[1, 2, 3]])  # 起始 token
generated = model.generate(input_ids, max_new_tokens=50, temperature=0.8)
print(generated)
```

## 训练流程

```
数据准备
    ↓
创建模型 + 优化器
    ↓
┌─────────────────────────────────┐
│ 训练循环                        │
│   for epoch in range(N):        │
│     for batch in data:          │
│       logits, loss = model(x)   │
│       loss.backward()           │
│       optimizer.step()          │
└─────────────────────────────────┘
    ↓
保存模型
    ↓
生成示例
```

## 超参数选择

| 参数 | 小模型 | 中模型 | 大模型 |
|------|--------|--------|--------|
| hidden_size | 128 | 768 | 1024 |
| num_layers | 4 | 12 | 24 |
| num_heads | 4 | 12 | 16 |
| 参数量 | ~1M | ~85M | ~350M |

## 与其他模块的关系

- **依赖**：所有前面的模块（02-09）
- **输入**：来自 `01-tokenization` 的 token IDs
- **输出**：预测的下一个 token 的概率分布
- **训练**：使用 `07-cross-entropy` 计算损失

## 延伸阅读

- [GPT-2](https://openai.com/blog/better-language-models/) - OpenAI 的 GPT-2
- [GPT-3](https://arxiv.org/abs/2005.14165) - GPT-3 论文
- [LLaMA](https://arxiv.org/abs/2302.13971) - Meta 的 LLaMA 模型
