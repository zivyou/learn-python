"""
Transformer 块模块 (Transformer Block)

组合多头注意力和前馈网络，是 Transformer 的基本构建单元。
使用 Pre-Norm 结构（现代 GPT 使用的方式）。
"""

import numpy as np
import sys
import os

# 添加父目录到路径，以便导入其他模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Optional


# 为了避免循环依赖，直接在这里实现依赖的组件

def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """数值稳定的 softmax"""
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def gelu(x: np.ndarray) -> np.ndarray:
    """GELU 激活函数"""
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))


class LayerNorm:
    """层归一化"""

    def __init__(self, hidden_size: int, eps: float = 1e-5):
        self.hidden_size = hidden_size
        self.eps = eps
        self.gamma = np.ones(hidden_size, dtype=np.float32)
        self.beta = np.zeros(hidden_size, dtype=np.float32)

    def forward(self, x: np.ndarray) -> np.ndarray:
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        x_norm = (x - mean) / np.sqrt(var + self.eps)
        return self.gamma * x_norm + self.beta

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)


class MultiHeadAttention:
    """多头自注意力"""

    def __init__(self, hidden_size: int, num_heads: int):
        assert hidden_size % num_heads == 0
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.W_q = np.random.randn(hidden_size, hidden_size).astype(np.float32) * np.sqrt(2.0 / hidden_size)
        self.W_k = np.random.randn(hidden_size, hidden_size).astype(np.float32) * np.sqrt(2.0 / hidden_size)
        self.W_v = np.random.randn(hidden_size, hidden_size).astype(np.float32) * np.sqrt(2.0 / hidden_size)
        self.W_o = np.random.randn(hidden_size, hidden_size).astype(np.float32) * np.sqrt(2.0 / hidden_size)

    def forward(self, x: np.ndarray, use_causal_mask: bool = False) -> np.ndarray:
        batch_size, seq_len, _ = x.shape

        Q = x @ self.W_q
        K = x @ self.W_k
        V = x @ self.W_v

        # 分头
        Q = Q.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        K = K.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        V = V.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

        # 注意力分数
        scores = (Q @ K.transpose(0, 1, 3, 2)) / np.sqrt(self.head_dim)

        # 因果掩码
        if use_causal_mask:
            mask = np.tril(np.ones((seq_len, seq_len), dtype=np.float32))
            mask = mask[np.newaxis, np.newaxis, :, :]
            scores = np.where(mask == 0, -1e9, scores)

        # Softmax + 加权求和
        weights = softmax(scores, axis=-1)
        output = weights @ V

        # 合并头
        output = output.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.hidden_size)

        # 输出投影
        return output @ self.W_o

    def __call__(self, x: np.ndarray, use_causal_mask: bool = False) -> np.ndarray:
        return self.forward(x, use_causal_mask)


class FeedForward:
    """前馈网络"""

    def __init__(self, hidden_size: int, ffn_size: int = None):
        self.hidden_size = hidden_size
        self.ffn_size = ffn_size or 4 * hidden_size

        self.W1 = np.random.randn(hidden_size, self.ffn_size).astype(np.float32) * np.sqrt(2.0 / hidden_size)
        self.b1 = np.zeros(self.ffn_size, dtype=np.float32)
        self.W2 = np.random.randn(self.ffn_size, hidden_size).astype(np.float32) * np.sqrt(2.0 / self.ffn_size)
        self.b2 = np.zeros(hidden_size, dtype=np.float32)

    def forward(self, x: np.ndarray) -> np.ndarray:
        h = gelu(x @ self.W1 + self.b1)
        return h @ self.W2 + self.b2

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)


class TransformerBlock:
    """
    Transformer 块

    结构（Pre-Norm）:
    x → LayerNorm → MultiHeadAttention → + x (残差)
      → LayerNorm → FeedForward → + (残差)

    参数:
        hidden_size: 隐藏层维度
        num_heads: 注意力头数
        ffn_size: 前馈网络中间层维度（默认 4 * hidden_size）
    """

    def __init__(self, hidden_size: int, num_heads: int, ffn_size: int = None):
        self.hidden_size = hidden_size
        self.num_heads = num_heads

        # 注意力层的 LayerNorm
        self.ln1 = LayerNorm(hidden_size)
        # 前馈网络的 LayerNorm
        self.ln2 = LayerNorm(hidden_size)

        # 多头注意力
        self.attention = MultiHeadAttention(hidden_size, num_heads)
        # 前馈网络
        self.ffn = FeedForward(hidden_size, ffn_size)

    def forward(self, x: np.ndarray, use_causal_mask: bool = False) -> np.ndarray:
        """
        前向传播

        参数:
            x: 输入张量 [batch_size, seq_len, hidden_size]
            use_causal_mask: 是否使用因果掩码

        返回:
            输出张量 [batch_size, seq_len, hidden_size]
        """
        # 1. 注意力子层（Pre-Norm + 残差连接）
        # x → LN → Attention → + x
        residual = x
        x = self.ln1(x)
        x = self.attention(x, use_causal_mask=use_causal_mask)
        x = x + residual

        # 2. 前馈子层（Pre-Norm + 残差连接）
        # x → LN → FFN → + x
        residual = x
        x = self.ln2(x)
        x = self.ffn(x)
        x = x + residual

        return x

    def __call__(self, x: np.ndarray, use_causal_mask: bool = False) -> np.ndarray:
        return self.forward(x, use_causal_mask)


def test_transformer_block():
    """测试 Transformer 块"""
    # 超参数
    hidden_size = 64
    num_heads = 8
    batch_size = 2
    seq_len = 10

    # 创建 Transformer 块
    block = TransformerBlock(hidden_size, num_heads)

    # 模拟输入
    x = np.random.randn(batch_size, seq_len, hidden_size).astype(np.float32)

    print("=" * 50)
    print("测试 1: 基本前向传播")
    print("=" * 50)

    # 前向传播
    output = block(x)
    print(f"输入形状: {x.shape}")       # (2, 10, 64)
    print(f"输出形状: {output.shape}")   # (2, 10, 64)
    assert output.shape == x.shape, "输出形状应该与输入相同"

    print("\n" + "=" * 50)
    print("测试 2: 因果掩码")
    print("=" * 50)

    # 使用因果掩码
    output_causal = block(x, use_causal_mask=True)
    print(f"因果掩码输出形状: {output_causal.shape}")
    assert output_causal.shape == x.shape

    # 验证：因果掩码应该导致不同的输出
    assert not np.array_equal(output, output_causal), "因果掩码应该改变输出"

    print("\n" + "=" * 50)
    print("测试 3: 残差连接效果")
    print("=" * 50)

    # 验证：输出应该与输入有相关性（因为残差连接）
    # 使用一个很小的输入，输出应该接近输入
    x_small = np.random.randn(1, 5, hidden_size).astype(np.float32) * 0.01
    output_small = block(x_small, use_causal_mask=True)

    # 由于残差连接，输出和输入应该有一定的相似性
    # （虽然不完全相同，因为注意力和 FFN 会改变值）
    print(f"输入范围: [{x_small.min():.4f}, {x_small.max():.4f}]")
    print(f"输出范围: [{output_small.min():.4f}, {output_small.max():.4f}]")

    print("\n" + "=" * 50)
    print("测试 4: 参数统计")
    print("=" * 50)

    # 计算参数数量
    ln1_params = block.ln1.gamma.size + block.ln1.beta.size
    ln2_params = block.ln2.gamma.size + block.ln2.beta.size
    attn_params = (block.attention.W_q.size + block.attention.W_k.size +
                   block.attention.W_v.size + block.attention.W_o.size)
    ffn_params = (block.ffn.W1.size + block.ffn.b1.size +
                  block.ffn.W2.size + block.ffn.b2.size)

    total_params = ln1_params + ln2_params + attn_params + ffn_params

    print(f"LayerNorm 1 参数: {ln1_params:,}")
    print(f"LayerNorm 2 参数: {ln2_params:,}")
    print(f"注意力参数: {attn_params:,}")
    print(f"FFN 参数: {ffn_params:,}")
    print(f"总参数: {total_params:,}")

    print("\n" + "=" * 50)
    print("测试 5: 多层堆叠")
    print("=" * 50)

    # 模拟多层 Transformer
    num_layers = 3
    layers = [TransformerBlock(hidden_size, num_heads) for _ in range(num_layers)]

    # 前向传播通过所有层
    h = x
    for i, layer in enumerate(layers):
        h = layer(h, use_causal_mask=True)
        print(f"第 {i+1} 层输出形状: {h.shape}")

    assert h.shape == x.shape, "最终输出形状应该与输入相同"

    print("\n✓ 所有测试通过！")


if __name__ == '__main__':
    test_transformer_block()
