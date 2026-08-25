"""
位置编码模块 (Positional Encoding)

为输入序列添加位置信息，使模型能够感知 token 的顺序。
使用正弦/余弦函数生成位置编码向量。
"""

import numpy as np


class PositionalEncoding:
    """
    正弦位置编码层

    参数:
        hidden_size: 嵌入维度（必须为偶数）
        max_seq_len: 最大序列长度

    输入: [batch_size, seq_len, hidden_size]
    输出: [batch_size, seq_len, hidden_size]（加上位置编码）
    """

    def __init__(self, hidden_size: int, max_seq_len: int = 5000):
        self.hidden_size = hidden_size
        self.max_seq_len = max_seq_len

        # 预计算位置编码矩阵
        self.pe = self._build_position_encoding()

    def _build_position_encoding(self) -> np.ndarray:
        """
        构建位置编码矩阵

        公式:
        PE(pos, 2i)   = sin(pos / 10000^(2i/d))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d))

        返回: [max_seq_len, hidden_size] 的位置编码矩阵
        """
        pe = np.zeros((self.max_seq_len, self.hidden_size), dtype=np.float32)

        # 位置索引: [0, 1, 2, ..., max_seq_len-1]
        position = np.arange(self.max_seq_len)[:, np.newaxis]  # [max_seq_len, 1]

        # 维度索引对应的分母: 10000^(2i/d)
        # 使用 log 空间计算避免数值溢出
        div_term = np.exp(
            np.arange(0, self.hidden_size, 2) * -(np.log(10000.0) / self.hidden_size)
        )  # [hidden_size/2]

        # 偶数维度使用 sin，奇数维度使用 cos
        pe[:, 0::2] = np.sin(position * div_term)  # sin 部分
        pe[:, 1::2] = np.cos(position * div_term)  # cos 部分

        return pe

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        前向传播：将位置编码加到输入上

        参数:
            x: 形状为 [batch_size, seq_len, hidden_size] 的输入张量

        返回:
            形状为 [batch_size, seq_len, hidden_size] 的输出张量
        """
        seq_len = x.shape[1]

        # 将位置编码加到输入上
        # pe[:seq_len] 形状为 [seq_len, hidden_size]
        # 广播到 [batch_size, seq_len, hidden_size]
        return x + self.pe[:seq_len]

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """支持函数式调用"""
        return self.forward(x)


def test_positional_encoding():
    """测试位置编码模块"""
    # 超参数
    hidden_size = 64
    max_seq_len = 100
    batch_size = 2
    seq_len = 10

    # 创建位置编码层
    pe = PositionalEncoding(hidden_size, max_seq_len)

    # 模拟输入（全零，便于观察位置编码）
    x = np.zeros((batch_size, seq_len, hidden_size), dtype=np.float32)

    # 前向传播
    output = pe(x)

    # 验证输出形状
    print(f"输入形状: {x.shape}")          # (2, 10, 64)
    print(f"输出形状: {output.shape}")      # (2, 10, 64)
    print(f"位置编码矩阵形状: {pe.pe.shape}")  # (100, 64)

    # 验证：输出应该等于位置编码（因为输入是全零）
    assert np.allclose(output[0], pe.pe[:seq_len]), "位置编码添加不正确！"

    # 验证：不同位置的编码应该不同
    assert not np.array_equal(output[0, 0], output[0, 1]), "不同位置的编码应该不同！"

    # 验证：同一批次中相同位置的编码应该相同
    assert np.array_equal(output[0, 0], output[1, 0]), "相同位置的编码应该相同！"

    # 验证：位置编码的值在合理范围内
    print(f"\n位置编码值范围: [{pe.pe.min():.4f}, {pe.pe.max():.4f}]")
    assert pe.pe.max() <= 1.0 and pe.pe.min() >= -1.0, "位置编码值应该在 [-1, 1] 范围内"

    # 可视化前几个位置的编码
    print("\n前5个位置的编码（前8个维度）:")
    for i in range(5):
        print(f"位置 {i}: {pe.pe[i, :8]}")

    print("\n✓ 所有测试通过！")


if __name__ == '__main__':
    test_positional_encoding()
