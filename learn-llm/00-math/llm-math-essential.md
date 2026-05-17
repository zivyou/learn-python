# LLM 数学概念与公式大全

---

## 一、线性代数基础

### 1.1 向量与矩阵运算

| 概念 | 公式/定义 | 应用场景 |
|------|-----------|----------|
| **向量点积** | $\mathbf{a} \cdot \mathbf{b} = \sum_{i=1}^n a_i b_i = \|\mathbf{a}\|\|\mathbf{b}\|\cos\theta$ | 注意力分数计算、相似度 |
| **外积** | $\mathbf{a} \otimes \mathbf{b} = \mathbf{a}\mathbf{b}^T$ | 秩-1矩阵、参数更新 |
| **矩阵乘法** | $(\mathbf{A}\mathbf{B})_{ij} = \sum_k \mathbf{A}_{ik}\mathbf{B}_{kj}$ | 线性变换、前向传播 |
| **Hadamard乘积** | $(\mathbf{A} \odot \mathbf{B})_{ij} = \mathbf{A}_{ij}\mathbf{B}_{ij}$ | 门控机制、LayerNorm |

### 1.2 矩阵分解与性质

| 概念 | 公式/定义 | 应用场景 |
|------|-----------|----------|
| **SVD分解** | $\mathbf{A} = \mathbf{U}\Sigma\mathbf{V}^T$ | 低秩近似、词嵌入分析 |
| **特征值分解** | $\mathbf{A}\mathbf{v} = \lambda\mathbf{v}$ | Hessian分析、优化理论 |
| **矩阵范数** | $\|\mathbf{A}\|_F = \sqrt{\sum_{i,j} \mathbf{A}_{ij}^2}$ | 正则化、梯度裁剪 |

---

## 二、微积分与自动微分

### 2.1 导数与梯度

| 概念 | 公式/定义 | 应用场景 |
|------|-----------|----------|
| **梯度** | $\nabla f(\mathbf{x}) = \left[\frac{\partial f}{\partial x_1}, \dots, \frac{\partial f}{\partial x_n}\right]^T$ | 参数更新方向 |
| **链式法则** | $\frac{dz}{dx} = \frac{dz}{dy} \cdot \frac{dy}{dx}$ | 反向传播核心 |
| **Jacobian矩阵** | $\mathbf{J}_{ij} = \frac{\partial y_i}{\partial x_j}$ | 多变量微分 |
| **Hessian矩阵** | $\mathbf{H}_{ij} = \frac{\partial^2 f}{\partial x_i \partial x_j}$ | 二阶优化、曲率分析 |

### 2.2 常见激活函数导数

| 函数 | 前向 | 导数 |
|------|------|------|
| **Sigmoid** | $\sigma(x) = \frac{1}{1+e^{-x}}$ | $\sigma'(x) = \sigma(x)(1-\sigma(x))$ |
| **ReLU** | $\text{ReLU}(x) = \max(0, x)$ | $\text{ReLU}'(x) = \mathbb{I}(x > 0)$ |
| **GELU** | $\text{GELU}(x) = x\Phi(x)$ | $\Phi(x) + x\phi(x)$ |
| **Swish** | $\text{Swish}(x) = x\sigma(\beta x)$ | $\sigma(\beta x) + \beta x\sigma(\beta x)(1-\sigma(\beta x))$ |

---

## 三、概率论与信息论

### 3.1 概率分布

| 分布 | 概率质量/密度函数 | 应用场景 |
|------|-------------------|----------|
| **Categorical** | $P(X=k) = p_k, \sum p_k = 1$ | Token采样 |
| **Softmax** | $p_i = \frac{e^{z_i}}{\sum_j e^{z_j}}$ | 输出层概率 |
| **Gaussian** | $\mathcal{N}(\mu, \sigma^2) = \frac{1}{\sqrt{2\pi\sigma^2}}e^{-\frac{(x-\mu)^2}{2\sigma^2}}$ | VAE、噪声注入 |

### 3.2 信息论核心概念

| 概念 | 公式 | 意义 |
|------|------|------|
| **熵** | $H(P) = -\sum_i P(i)\log P(i)$ | 不确定性度量 |
| **交叉熵** | $H(P,Q) = -\sum_i P(i)\log Q(i)$ | 损失函数核心 |
| **KL散度** | $D_{KL}(P\|Q) = \sum_i P(i)\log\frac{P(i)}{Q(i)}$ | 分布差异、≥0 |
| **互信息** | $I(X;Y) = H(X) - H(X\|Y)$ | 变量依赖程度 |
| **Perplexity** | $\text{PPL} = 2^{H(P)}$ | 语言模型评估 |

---

## 四、Transformer核心数学

### 4.1 注意力机制

```math
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
```

- $Q \in \mathbb{R}^{n \times d_k}$: Query矩阵
- $K \in \mathbb{R}^{m \times d_k}$: Key矩阵  
- $V \in \mathbb{R}^{m \times d_v}$: Value矩阵
- $\sqrt{d_k}$: 缩放因子，防止点积过大

### 4.2 多头注意力

```math
\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \dots, \text{head}_h)W^O
```
```math
\text{where head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)
```

### 4.3 位置编码

**正弦位置编码：**
```math
PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d}}\right)
```
```math
PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d}}\right)
```

**RoPE（旋转位置编码）：**
```math
f(q, m) = \begin{bmatrix} q_0\cos m\theta_0 - q_1\sin m\theta_0 \\ q_0\sin m\theta_0 + q_1\cos m\theta_0 \\ \vdots \end{bmatrix}
```

---

## 五、归一化技术

### 5.1 LayerNorm

```math
\text{LN}(x) = \gamma \odot \frac{x - \mu}{\sigma + \epsilon} + \beta
```

- $\mu = \frac{1}{d}\sum_{i=1}^d x_i$（均值）
- $\sigma^2 = \frac{1}{d}\sum_{i=1}^d (x_i - \mu)^2$（方差）
- $\gamma, \beta$: 可学习参数

### 5.2 RMSNorm（简化版）

```math
\text{RMSNorm}(x) = \gamma \odot \frac{x}{\sqrt{\frac{1}{d}\sum_{i=1}^d x_i^2 + \epsilon}}
```
*无均值减法，计算更快*

---

## 六、优化算法

### 6.1 SGD带动量

```math
v_t = \beta v_{t-1} + (1-\beta)\nabla L(\theta_t)
```
```math
\theta_{t+1} = \theta_t - \alpha v_t
```

### 6.2 Adam（最常用）

```math
m_t = \beta_1 m_{t-1} + (1-\beta_1)g_t \quad \text{(一阶矩)}
```
```math
v_t = \beta_2 v_{t-1} + (1-\beta_2)g_t^2 \quad \text{(二阶矩)}
```
```math
\hat{m}_t = \frac{m_t}{1-\beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1-\beta_2^t} \quad \text{(偏差修正)}
```
```math
\theta_{t+1} = \theta_t - \frac{\alpha \hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}
```
*典型值：$\beta_1=0.9, \beta_2=0.999, \epsilon=10^{-8}$*

### 6.3 AdamW（权重衰减解耦）

```math
\theta_{t+1} = \theta_t - \alpha\left(\frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} + \lambda \theta_t\right)
```

---

## 七、正则化技术

| 技术 | 公式 | 作用 |
|------|------|------|
| **L2正则** | $L_{\text{total}} = L + \frac{\lambda}{2}\|\theta\|_2^2$ | 权重衰减 |
| **Dropout** | $y = \text{mask} \odot x / p$ | 防止过拟合 |
| **Label Smoothing** | $y'_i = (1-\epsilon)y_i + \epsilon/K$ | 软化标签 |

---

## 八、采样与解码

| 方法 | 公式/描述 | 特点 |
|------|-----------|------|
| **Greedy** | $\hat{x}_t = \arg\max P(x_t\|x_{<t})$ | 确定性、可能重复 |
| **Temperature** | $P'(x) \propto e^{\log P(x)/T}$ | T→0变贪婪，T→∞变均匀 |
| **Top-k** | 仅从概率最高的k个token采样 | 平衡质量与多样性 |
| **Nucleus (Top-p)** | 选择最小集合使$\sum P \geq p$ | 动态选择候选集 |

---

## 九、KV Cache数学

```math
\text{Cache}_t = [K_0, K_1, ..., K_t], [V_0, V_1, ..., V_t]
```

增量推理：
```math
\text{Attn}_t = \text{softmax}\left(\frac{Q_t K_{\leq t}^T}{\sqrt{d}}\right)V_{\leq t}
```

---

## 十、高级概念

### 10.1 LoRA（低秩适配）

```math
h = W_0 x + \Delta W x = W_0 x + BAx
```

- $B \in \mathbb{R}^{d \times r}, A \in \mathbb{R}^{r \times d}$
- $r \ll d$: 秩，通常8-64

### 10.2 FlashAttention（IO感知优化）

利用分块计算，实现$O(N)$内存复杂度：
```math
S_{ij} = Q_i K_j^T, \quad O_i = \sum_j \text{softmax}(S_{ij}) V_j
```
（分块计算，避免完整注意力矩阵实例化）

---

## 总结

LLM的数学核心是**线性代数**（矩阵运算）+ **微积分**（反向传播）+ **概率论**（分布建模）的有机结合。理解这些数学概念对深入掌握LLM工作原理至关重要。
