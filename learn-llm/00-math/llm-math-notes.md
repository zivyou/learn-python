# LLM 数学原理学习笔记

## 目录
- [概率统计](#概率统计)
- [线性代数](#线性代数)
- [微积分](#微积分)
- [优化理论](#优化理论)
- [信息论](#信息论)
- [Transformer 数学原理](#transformer-数学原理)

---

## 概率统计

### 主题

---

## 线性代数

### 主题

---

## 微积分

### 主题

---

## 优化理论

### 主题

---

## 信息论

### 主题

---

## Transformer 数学原理

### Transformer 输入结构原理

**核心概念：从 Token ID 到向量序列的两步转换**

#### 1. 原始输入: Token ID 矩阵
```
形状: [batch_size, seq_len]
类型: 整数 (int64)
```
- `batch_size`: 一次并行训练的序列数量
- `seq_len`: 单个序列中的 token 数量
- **每个元素是 token 的唯一标识符**（如：`[101, 345, 782, ...]`），不是向量

#### 2. Embedding 层: 查找表映射
```
Embedding 矩阵形状: [vocab_size, hidden_size]
  ↓ 查表映射
Transformer 输入形状: [batch_size, seq_len, hidden_size]
```
- Embedding 层本质是一个可训练的查找表
- 输入一个 token ID，取出表中对应的一行向量
- 结果：每个 token 位置上都有一个长度为 `hidden_size` 的向量

#### 3. 概念图示
```
[32, 1024]          →  Embedding 层  →  [32, 1024, 768]
   ↓                                         ↓
整数矩阵                                   3D 张量
(Token IDs)                             (每个位置是向量)
```

**关键点**：真正输入到 Transformer 层的是 3D 张量，不是 2D 矩阵。

---

### 自注意力计算完整流程

**核心公式 (Scaled Dot-Product Attention)**:
```
Attention(Q, K, V) = Softmax( (Q × K^T) / √head_dim ) × V
```

---

#### 1. Q, K, V 投影
```
输入: [batch_size, seq_len, hidden_size]
  ↓ × W_q / W_k / W_v
Q/K/V: [batch_size, seq_len, hidden_size]  ← 形状与输入相同
  ↓ 分头 (reshape)
Q/K/V: [batch_size, num_heads, seq_len, head_dim]
```

**关键等式**: `hidden_size = num_heads × head_dim`

---

#### 2. 注意力分数计算
```
Q × K^T: [batch_size, num_heads, seq_len, seq_len]  ← 原始分数
  ↓ 缩放 (÷ √head_dim)
  ↓ 因果掩码 (上三角设为 -∞, 仅解码器)
  ↓ Softmax 归一化
注意力权重: [batch_size, num_heads, seq_len, seq_len]
```

**注意力矩阵含义**:
- **第 i 行**: 第 i 个 token 如何分配注意力给所有 token
- **第 j 列**: 第 j 个 token 被所有 token 关注的程度
- **每行和为 1** (Softmax 性质)

**缩放的意义**: 防止 Q·K 点积方差过大导致 Softmax 接近 one-hot

---

#### 3. 加权输出
```
注意力权重 × V: [batch_size, num_heads, seq_len, head_dim]
  ↓ 拼接 (concat heads)
输出: [batch_size, seq_len, hidden_size]  ← 形状回到原始输入
```

---

### 权重初始化与参数更新

#### W_q, W_k, W_v 的初始化

**核心原则**：没有"正确"的初始值，只有"合适"的分布。目标是保证训练开始时信号能正常流动。

**1. Xavier/Glorot 初始化（早期 Transformer）**
```python
limit = sqrt(6 / (hidden_size + hidden_size))
W = uniform(-limit, limit)  # 均匀分布
```
- 目标：保持每一层输入输出的方差一致
- 防止：梯度消失或爆炸

**2. He 初始化（现代主流，配合 GELU）**
```python
W = normal(0, sqrt(2 / hidden_size))  # 正态分布
```
- 考虑了激活函数对信号的压缩影响
- LLaMA、GPT 等现代模型使用

**初始化的临界性**：
- ✗ 权重过大 → Q·K^T 过大 → Softmax 接近 one-hot → 梯度消失
- ✗ 权重过小 → Q·K^T 过小 → Softmax 接近均匀分布 → 注意力失效
- 偏置项通常初始化为 0

---

#### 参数学习机制：梯度下降 + 反向传播

**完整训练循环**：
```
1. 前向传播：计算模型输出和预测损失
2. 反向传播：用链式法则计算每个参数的梯度
3. 参数更新：沿梯度反方向调整参数
4. 重复上述步骤
```

---

#### 梯度的物理意义

```
梯度 dL/dW = 损失 L 对参数 W 的偏导数

- dL/dW > 0: W 增大 → 损失增大 → 应让 W 变小
- dL/dW < 0: W 增大 → 损失减小 → 应让 W 变大
- |dL/dW|: 该参数对损失的影响程度
```

---

#### 注意力层的梯度流动路径

梯度沿前向计算的反方向传递：

```
损失 L
  ↓
输出层 (LM Head)
  ↓
FFN 层
  ↓
注意力输出 O = Attention(Q,K,V)
  ├─→ dL/dV = 注意力权重^T × dL/dO
  └─→ dL/dAttention_weights = dL/dO × V^T
      ↓
      Softmax 输入梯度
      ↓
      Q×K^T 梯度
      ├─→ dL/dK = Q^T × 上游梯度
      └─→ dL/dQ = 上游梯度 × K
          ↓
          dL/dW_q = 输入^T × dL/dQ
          dL/dW_k = 输入^T × dL/dK
          dL/dW_v = 输入^T × dL/dV
```

---

#### 参数更新规则 (AdamW 优化器)

```python
W = W - learning_rate × gradient
```

**直观类比：盲人下山**
- 梯度 = 脚下的坡度方向
- 往坡度的反方向（向下）走一小步
- 学习率 = 步长大小
- AdamW = 根据历史坡度给每个方向自适应的步长

---

#### 直观理解

训练句子："The cat sat on the mat. It is very cute."
1. 初始状态："It" 给 "mat" 0.5 注意力，给 "cat" 0.3 注意力
2. 预测 "cute" 出错（因为 "mat" 不 cute）
3. 反向传播：梯度告诉 W_q/W_k/W_v — "下次让 'It' 多关注 'cat'"
4. 参数更新：相关权重被微调，下次注意力分配更合理

**核心洞察**：没有绝对的"正确值"，只有"能让损失更小的值"。每一步微小改进，几百万步迭代后达到"足够好"的状态。

---

## 模型文件结构

### 大模型二进制文件内容

模型文件（通常为 `.safetensors` 或 `.bin` 格式）主要包含以下内容：

**1. 模型参数 (Weights & Biases)**
- **Embedding 层**:
  - 词嵌入矩阵: `[vocab_size, hidden_size]` - 每个词的向量表示
  - 位置嵌入矩阵: `[max_seq_len, hidden_size]` - 每个位置的向量表示
- **Transformer 层** (每层都有):
  - 注意力权重: `W_q`, `W_k`, `W_v` - 每个 `[hidden_size, hidden_size]`
  - 注意力输出投影: `W_o` - `[hidden_size, hidden_size]`
  - 前馈网络第一层: `W_1`, `b_1` - `[hidden_size, ffn_size]`
  - 前馈网络第二层: `W_2`, `b_2` - `[ffn_size, hidden_size]`
  - LayerNorm 权重和偏置: `gamma`, `beta` - `[hidden_size]`
- **输出层**:
  - 最终 LayerNorm: `[hidden_size]`
  - 语言模型头: `[hidden_size, vocab_size]` - 将隐藏状态映射回词汇表

**2. 模型配置信息**
- 超参数: 层数、隐藏层维度、头数、词汇表大小、最大序列长度
- 架构类型: GPT-2, LLaMA, Mistral 等
- 特殊标记定义: `<s>`, `</s>`, `<pad>` 等

**3. 分词器数据**
- BPE 词汇表和合并规则
- 特殊标记映射
- 预处理/后处理配置

**4. 元数据**
- 训练步数、学习率、损失等训练信息
- 模型版本和来源

**参数数量估算 (以 7B 模型为例)**:
- Embedding: ~ vocab_size × hidden_size = 32k × 4096 ≈ 131M
- 每层注意力: ~ 4 × hidden_size² = 4 × 16M = 64M
- 每层 FFN: ~ 2 × hidden_size × ffn_size = 2 × 4096 × 11008 ≈ 90M
- 32 层总参数 ≈ 32 × (64M + 90M) ≈ **7B**

---

## 参考资料

