"""
前馈神经网络模块 (Feed-Forward Network)

Transformer 中的前馈网络层，由两个线性变换和一个激活函数组成。
是 Transformer 的"记忆"组件，存储知识。
"""

import numpy as np


def gelu(x: np.ndarray) -> np.ndarray:
    """
    GELU 激活函数 (Gaussian Error Linear Unit)

    公式: GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))

    比 ReLU 更平滑，是现代 Transformer 的标准激活函数。

    参数:
        x: 输入张量

    返回:
        激活后的张量
    """
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))


def relu(x: np.ndarray) -> np.ndarray:
    """
    ReLU 激活函数

    公式: ReLU(x) = max(0, x)

    参数:
        x: 输入张量

    返回:
        激活后的张量
    """
    return np.maximum(0, x)


class FeedForward:
    """
    前馈网络层

    结构: Linear → GELU → Linear

    参数:
        hidden_size: 输入/输出维度
        ffn_size: 中间层维度（通常为 4 * hidden_size）

    输入: [batch_size, seq_len, hidden_size]
    输出: [batch_size, seq_len, hidden_size]
    """

    def __init__(self, hidden_size: int, ffn_size: int = None):
        self.hidden_size = hidden_size
        self.ffn_size = ffn_size or 4 * hidden_size

        # 第一层线性变换: hidden_size → ffn_size
        # 使用 He 初始化
        self.W1 = np.random.randn(hidden_size, self.ffn_size).astype(np.float32) * np.sqrt(2.0 / hidden_size)
        self.b1 = np.zeros(self.ffn_size, dtype=np.float32)

        # 第二层线性变换: ffn_size → hidden_size
        self.W2 = np.random.randn(self.ffn_size, hidden_size).astype(np.float32) * np.sqrt(2.0 / self.ffn_size)
        self.b2 = np.zeros(hidden_size, dtype=np.float32)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        前向传播

        参数:
            x: 形状为 [batch_size, seq_len, hidden_size] 的输入张量

        返回:
            形状为 [batch_size, seq_len, hidden_size] 的输出张量
        """
        # 第一层: x @ W1 + b1
        # 形状: [B, S, hidden] → [B, S, ffn_size]
        h = x @ self.W1 + self.b1

        # 激活函数: GELU
        h = gelu(h)

        # 第二层: h @ W2 + b2
        # 形状: [B, S, ffn_size] → [B, S, hidden]
        output = h @ self.W2 + self.b2

        return output

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """支持函数式调用"""
        return self.forward(x)


def test_feed_forward():
    """测试前馈网络模块"""
    # 超参数
    hidden_size = 64
    ffn_size = 256  # 4 * hidden_size
    batch_size = 2
    seq_len = 10

    # 创建前馈网络
    ffn = FeedForward(hidden_size, ffn_size)

    # 模拟输入
    x = np.random.randn(batch_size, seq_len, hidden_size).astype(np.float32)

    # 前向传播
    output = ffn(x)

    # 验证输出形状
    print(f"输入形状: {x.shape}")        # (2, 10, 64)
    print(f"输出形状: {output.shape}")    # (2, 10, 64)
    print(f"W1 形状: {ffn.W1.shape}")    # (64, 256)
    print(f"W2 形状: {ffn.W2.shape}")    # (256, 64)

    # 验证：输出形状应该与输入相同
    assert output.shape == x.shape, "输出形状应该与输入相同"

    # 验证：GELU 激活函数
    print("\nGELU 激活函数测试:")
    test_input = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    gelu_output = gelu(test_input)
    print(f"输入: {test_input}")
    print(f"GELU: {gelu_output}")

    # GELU(0) 应该接近 0
    assert np.isclose(gelu(0.0), 0.0), "GELU(0) 应该为 0"

    # 正值应该接近输入本身
    assert gelu(1.0) > 0.8, "GELU(1) 应该接近 1"

    # 负值应该接近 0
    assert gelu(-2.0) > -0.1, "GELU(-2) 应该接近 0"

    # 验证：参数数量
    total_params = ffn.W1.size + ffn.b1.size + ffn.W2.size + ffn.b2.size
    print(f"\n总参数数量: {total_params:,}")
    expected_params = hidden_size * ffn_size + ffn_size + ffn_size * hidden_size + hidden_size
    assert total_params == expected_params, "参数数量不正确"

    print("\n✓ 所有测试通过！")


if __name__ == '__main__':
    test_feed_forward()
