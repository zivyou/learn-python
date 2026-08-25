"""
层归一化模块 (Layer Normalization)

对每个样本的最后一维进行归一化，稳定训练过程。
是 Transformer 的核心组件之一。
"""

import numpy as np


class LayerNorm:
    """
    层归一化层

    公式: LN(x) = gamma * (x - mean) / sqrt(var + eps) + beta

    参数:
        hidden_size: 特征维度
        eps: 防止除零的小常数（默认 1e-5）

    输入: [batch_size, seq_len, hidden_size]
    输出: [batch_size, seq_len, hidden_size]（归一化后）
    """

    def __init__(self, hidden_size: int, eps: float = 1e-5):
        self.hidden_size = hidden_size
        self.eps = eps

        # 可学习参数
        # gamma: 缩放参数，初始化为全 1
        self.gamma = np.ones(hidden_size, dtype=np.float32)
        # beta: 偏移参数，初始化为全 0
        self.beta = np.zeros(hidden_size, dtype=np.float32)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        前向传播

        参数:
            x: 形状为 [..., hidden_size] 的输入张量

        返回:
            归一化后的张量，形状与输入相同
        """
        # 计算均值和方差（沿最后一维）
        mean = np.mean(x, axis=-1, keepdims=True)       # [..., 1]
        var = np.var(x, axis=-1, keepdims=True)         # [..., 1]

        # 归一化
        x_norm = (x - mean) / np.sqrt(var + self.eps)   # [..., hidden_size]

        # 仿射变换
        return self.gamma * x_norm + self.beta

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """支持函数式调用"""
        return self.forward(x)


def test_layernorm():
    """测试层归一化模块"""
    # 超参数
    hidden_size = 64
    batch_size = 2
    seq_len = 10

    # 创建 LayerNorm
    ln = LayerNorm(hidden_size)

    # 模拟输入
    x = np.random.randn(batch_size, seq_len, hidden_size).astype(np.float32)

    # 前向传播
    output = ln(x)

    # 验证输出形状
    print(f"输入形状: {x.shape}")       # (2, 10, 64)
    print(f"输出形状: {output.shape}")   # (2, 10, 64)

    # 验证：归一化后均值应该接近 0
    output_mean = np.mean(output, axis=-1)
    print(f"\n归一化后均值（应该接近 0）: {output_mean[0, :5]}")
    assert np.allclose(output_mean, 0, atol=1e-5), "归一化后均值应该接近 0"

    # 验证：归一化后方差应该接近 1（当 gamma=1, beta=0 时）
    output_var = np.var(output, axis=-1)
    print(f"归一化后方差（应该接近 1）: {output_var[0, :5]}")
    assert np.allclose(output_var, 1, atol=1e-3), "归一化后方差应该接近 1"

    # 验证：可学习参数的形状
    print(f"\ngamma 形状: {ln.gamma.shape}")  # (64,)
    print(f"beta 形状: {ln.beta.shape}")      # (64,)

    # 验证：gamma 和 beta 的初始值
    assert np.allclose(ln.gamma, 1.0), "gamma 应该初始化为 1"
    assert np.allclose(ln.beta, 0.0), "beta 应该初始化为 0"

    print("\n✓ 所有测试通过！")


if __name__ == '__main__':
    test_layernorm()
