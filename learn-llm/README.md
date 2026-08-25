# LLM 学习框架

从零手写一个简单的大模型训练框架，学习 LLM 的核心技术和原理。

## 项目概述

本项目采用**混合方式**实现：
- **核心算法**：用 NumPy 手写教学版，深入理解底层原理
- **最终训练框架**：用 PyTorch 组合各模块，实现完整训练流程

每个主题独立成模块，最后组合成一个可在小数据集上训练和生成文本的简单 GPT 模型。

## 目录结构

```
learn-llm/
├── 00-math/                    # 数学基础笔记
├── 01-tokenization/            # BPE 分词
├── 02-embedding/               # 词嵌入
├── 03-positional-encoding/     # 位置编码
├── 04-softmax/                 # Softmax 函数
├── 05-layernorm/               # 层归一化
├── 06-feed-forward/            # 前馈神经网络
├── 07-cross-entropy/           # 交叉熵损失
├── 08-multi-head-attention/    # 多头注意力
├── 09-transformer-block/       # Transformer 块
├── 10-gpt-model/               # 最终 GPT 模型（PyTorch）
├── pyproject.toml              # 项目配置
└── README.md                   # 本文件
```

## 模块依赖关系

```
02-embedding ─────────────────────────┐
03-positional-encoding ───────────────┤
04-softmax ───────────────────────────┤
05-layernorm ─────────────────────────┤
                                      ├─→ 08-multi-head-attention ─→ 09-transformer-block ─→ 10-gpt-model
06-feed-forward ─────────────────────┤
07-cross-entropy ────────────────────┘
```

## 快速开始

### 环境准备

```bash
# 使用 uv 安装依赖（推荐）
uv sync

# 或使用 pip
pip install -r requirements.txt
```

### 运行各模块测试

```bash
# 测试各个模块
uv run python 02-embedding/embedding.py
uv run python 03-positional-encoding/positional_encoding.py
uv run python 04-softmax/softmax.py
uv run python 05-layernorm/layernorm.py
uv run python 06-feed-forward/feed_forward.py
uv run python 07-cross-entropy/cross_entropy.py
uv run python 08-multi-head-attention/multi_head_attention.py
uv run python 09-transformer-block/transformer_block.py
uv run python 10-gpt-model/gpt_model.py
```

### 训练 GPT 模型

```bash
cd 10-gpt-model
uv run python train.py
```

## 学习路径

### 阶段一：数学基础（00-math）

学习 LLM 所需的数学知识：
- 线性代数：矩阵运算、向量空间
- 微积分：梯度、链式法则
- 概率论：概率分布、信息论

### 阶段二：基础组件（01-07）

理解 LLM 的基本构建单元：

1. **Tokenization（01）**：将文本转换为 token
2. **Embedding（02）**：将 token 转换为向量
3. **Positional Encoding（03）**：添加位置信息
4. **Softmax（04）**：概率归一化
5. **LayerNorm（05）**：稳定训练
6. **Feed-Forward（06）**：信息转换
7. **Cross-Entropy（07）**：损失计算

### 阶段三：核心机制（08-09）

掌握 Transformer 的核心：

8. **Multi-Head Attention（08）**：注意力机制
9. **Transformer Block（09）**：组合注意力和 FFN

### 阶段四：完整模型（10）

组合所有模块，训练 GPT 模型：
- 前向传播
- 损失计算
- 反向传播
- 参数更新
- 文本生成

## 张量形状约定

| 符号 | 含义 |
|------|------|
| `batch_size` | 批次大小 |
| `seq_len` | 序列长度 |
| `hidden_size` | 隐藏层维度 |
| `num_heads` | 注意力头数 |
| `head_dim` | 每个头的维度 |
| `vocab_size` | 词汇表大小 |
| `ffn_size` | 前馈网络中间层维度 |

## 核心公式速查

### 注意力机制

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

### LayerNorm

$$\text{LN}(x) = \gamma \odot \frac{x - \mu}{\sigma + \epsilon} + \beta$$

### 交叉熵损失

$$\text{CE} = -\sum_i y_i \log \hat{y}_i$$

### Softmax

$$\text{softmax}(x_i) = \frac{e^{x_i}}{\sum_j e^{x_j}}$$

## 参考资料

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Transformer 原始论文
- [GPT-2](https://openai.com/blog/better-language-models/) - GPT-2 介绍
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) - 图解 Transformer
- [Andrej Karpathy - Let's build GPT](https://www.youtube.com/watch?v=kCc8FmEb1nY) - 从零构建 GPT

## 许可证

MIT License
