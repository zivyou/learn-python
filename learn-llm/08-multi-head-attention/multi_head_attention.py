"""
多头注意力模块 (Multi-Head Attention)

Transformer 的核心机制：让每个 token 能够"关注"其他相关 token。
多头机制让模型从不同角度理解信息。
"""

import numpy as np
from typing import Optional


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """数值稳定的 softmax"""
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


class MultiHeadAttention:
    """
    多头自注意力层

    参数:
        hidden_size: 输入/输出维度
        num_heads: 注意力头数（必须能整除 hidden_size）

    输入: [batch_size, seq_len, hidden_size]
    输出: [batch_size, seq_len, hidden_size]
    """

    def __init__(self, hidden_size: int, num_heads: int):
        assert hidden_size % num_heads == 0, "hidden_size 必须能被 num_heads 整除"

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads  # 每个头的维度

        # Q, K, V 投影矩阵
        # 使用 He 初始化
        self.W_q = np.random.randn(hidden_size, hidden_size).astype(np.float32) * np.sqrt(2.0 / hidden_size)
        self.W_k = np.random.randn(hidden_size, hidden_size).astype(np.float32) * np.sqrt(2.0 / hidden_size)
        self.W_v = np.random.randn(hidden_size, hidden_size).astype(np.float32) * np.sqrt(2.0 / hidden_size)

        # 输出投影矩阵
        self.W_o = np.random.randn(hidden_size, hidden_size).astype(np.float32) * np.sqrt(2.0 / hidden_size)

    def _split_heads(self, x: np.ndarray) -> np.ndarray:
        """
        将张量分割成多个头

        输入: [batch_size, seq_len, hidden_size]
        输出: [batch_size, num_heads, seq_len, head_dim]
        """
        batch_size, seq_len, _ = x.shape

        # reshape: [B, S, H] → [B, S, num_heads, head_dim]
        x = x.reshape(batch_size, seq_len, self.num_heads, self.head_dim)

        # transpose: [B, S, num_heads, head_dim] → [B, num_heads, S, head_dim]
        return x.transpose(0, 2, 1, 3)

    def _merge_heads(self, x: np.ndarray) -> np.ndarray:
        """
        将多个头合并回来

        输入: [batch_size, num_heads, seq_len, head_dim]
        输出: [batch_size, seq_len, hidden_size]
        """
        batch_size, _, seq_len, _ = x.shape

        # transpose: [B, num_heads, S, head_dim] → [B, S, num_heads, head_dim]
        x = x.transpose(0, 2, 1, 3)

        # reshape: [B, S, num_heads, head_dim] → [B, S, hidden_size]
        return x.reshape(batch_size, seq_len, self.hidden_size)

    def _create_causal_mask(self, seq_len: int) -> np.ndarray:
        """
        创建因果掩码（下三角矩阵）

        用于解码器：防止 token 看到未来的信息

        返回: [seq_len, seq_len] 的掩码，1 表示可见，0 表示不可见
        """
        # 下三角矩阵（包括对角线）
        mask = np.tril(np.ones((seq_len, seq_len), dtype=np.float32))
        return mask

    def forward(
        self,
        x: np.ndarray,
        mask: Optional[np.ndarray] = None,
        use_causal_mask: bool = False
    ) -> np.ndarray:
        """
        前向传播

        参数:
            x: 输入张量 [batch_size, seq_len, hidden_size]
            mask: 可选的外部掩码
            use_causal_mask: 是否使用因果掩码（用于解码器）

        返回:
            注意力输出 [batch_size, seq_len, hidden_size]
        """
        batch_size, seq_len, _ = x.shape

        # 1. Q, K, V 投影
        Q = x @ self.W_q  # [B, S, H]
        K = x @ self.W_k  # [B, S, H]
        V = x @ self.W_v  # [B, S, H]

        # 2. 分割成多个头
        Q = self._split_heads(Q)  # [B, num_heads, S, head_dim]
        K = self._split_heads(K)  # [B, num_heads, S, head_dim]
        V = self._split_heads(V)  # [B, num_heads, S, head_dim]

        # 3. 计算注意力分数
        # scores = Q @ K^T / sqrt(head_dim)
        # 形状: [B, num_heads, S, S]
        scores = (Q @ K.transpose(0, 1, 3, 2)) / np.sqrt(self.head_dim)

        # 4. 应用掩码
        if use_causal_mask:
            # 因果掩码：将未来位置设为 -inf
            causal_mask = self._create_causal_mask(seq_len)  # [S, S]
            # 扩展到 [1, 1, S, S] 以便广播
            causal_mask = causal_mask[np.newaxis, np.newaxis, :, :]
            scores = np.where(causal_mask == 0, -1e9, scores)

        if mask is not None:
            # 外部掩码
            scores = np.where(mask == 0, -1e9, scores)

        # 5. Softmax 归一化
        attention_weights = softmax(scores, axis=-1)  # [B, num_heads, S, S]

        # 6. 加权求和
        # output = attention_weights @ V
        # 形状: [B, num_heads, S, head_dim]
        output = attention_weights @ V

        # 7. 合并多个头
        output = self._merge_heads(output)  # [B, S, H]

        # 8. 输出投影
        output = output @ self.W_o  # [B, S, H]

        return output

    def __call__(
        self,
        x: np.ndarray,
        mask: Optional[np.ndarray] = None,
        use_causal_mask: bool = False
    ) -> np.ndarray:
        """支持函数式调用"""
        return self.forward(x, mask, use_causal_mask)


def test_multi_head_attention():
    """测试多头注意力模块"""
    # 超参数
    hidden_size = 64
    num_heads = 8
    batch_size = 2
    seq_len = 10

    # 创建多头注意力
    mha = MultiHeadAttention(hidden_size, num_heads)

    # 模拟输入
    x = np.random.randn(batch_size, seq_len, hidden_size).astype(np.float32)

    print("=" * 50)
    print("测试 1: 基本前向传播")
    print("=" * 50)

    # 前向传播（无掩码）
    output = mha(x)
    print(f"输入形状: {x.shape}")           # (2, 10, 64)
    print(f"输出形状: {output.shape}")       # (2, 10, 64)
    assert output.shape == x.shape, "输出形状应该与输入相同"

    print("\n" + "=" * 50)
    print("测试 2: 因果掩码")
    print("=" * 50)

    # 使用因果掩码
    output_causal = mha(x, use_causal_mask=True)
    print(f"因果掩码输出形状: {output_causal.shape}")
    assert output_causal.shape == x.shape

    # 验证：因果掩码应该导致不同的输出
    assert not np.array_equal(output, output_causal), "因果掩码应该改变输出"

    print("\n" + "=" * 50)
    print("测试 3: 注意力权重可视化")
    print("=" * 50)

    # 手动计算注意力权重用于可视化
    Q = x @ mha.W_q
    K = x @ mha.W_k
    Q = mha._split_heads(Q)
    K = mha._split_heads(K)
    scores = (Q @ K.transpose(0, 1, 3, 2)) / np.sqrt(mha.head_dim)

    # 无掩码的注意力权重
    weights_no_mask = softmax(scores, axis=-1)
    print(f"注意力权重形状: {weights_no_mask.shape}")  # (2, 8, 10, 10)

    # 有因果掩码的注意力权重
    causal_mask = mha._create_causal_mask(seq_len)
    causal_mask = causal_mask[np.newaxis, np.newaxis, :, :]
    scores_causal = np.where(causal_mask == 0, -1e9, scores)
    weights_causal = softmax(scores_causal, axis=-1)

    # 验证：因果掩码下，每个位置只能关注自己和之前的位置
    print("\n因果掩码下的注意力权重（第一个头，第一个样本）:")
    print("行 = query 位置，列 = key 位置")
    print("0 表示不能关注，>0 表示可以关注")
    print(weights_causal[0, 0, :5, :5].round(2))

    # 验证：上三角应该为 0
    for i in range(seq_len):
        for j in range(i + 1, seq_len):
            assert weights_causal[0, 0, i, j] < 1e-6, f"位置 {i} 不应该关注位置 {j}"

    print("\n" + "=" * 50)
    print("测试 4: 参数形状")
    print("=" * 50)

    print(f"W_q 形状: {mha.W_q.shape}")  # (64, 64)
    print(f"W_k 形状: {mha.W_k.shape}")  # (64, 64)
    print(f"W_v 形状: {mha.W_v.shape}")  # (64, 64)
    print(f"W_o 形状: {mha.W_o.shape}")  # (64, 64)
    print(f"head_dim: {mha.head_dim}")    # 8

    total_params = mha.W_q.size + mha.W_k.size + mha.W_v.size + mha.W_o.size
    print(f"总参数数量: {total_params:,}")

    print("\n✓ 所有测试通过！")


if __name__ == '__main__':
    test_multi_head_attention()
