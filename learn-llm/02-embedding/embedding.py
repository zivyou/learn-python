"""
词嵌入模块 (Token Embedding)

将离散的 token ID 映射为连续的向量表示。
本质是一个可训练的查找表（lookup table）。
"""

import numpy as np


class Embedding:
    """
    词嵌入层：将 token ID 映射为稠密向量

    参数:
        vocab_size: 词汇表大小
        hidden_size: 嵌入向量维度

    输入: [batch_size, seq_len] 的整数张量（token IDs）
    输出: [batch_size, seq_len, hidden_size] 的浮点张量（嵌入向量）
    """

    def __init__(self, vocab_size: int, hidden_size: int):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size

        # 初始化嵌入矩阵：正态分布 N(0, 1)
        # 形状: [vocab_size, hidden_size]
        self.weight = np.random.randn(vocab_size, hidden_size).astype(np.float32)

    def forward(self, input_ids: np.ndarray) -> np.ndarray:
        """
        前向传播：根据 token IDs 查找对应的嵌入向量

        参数:
            input_ids: 形状为 [batch_size, seq_len] 的整数数组

        返回:
            形状为 [batch_size, seq_len, hidden_size] 的浮点数组
        """
        # 核心操作：矩阵索引
        # input_ids 中的每个整数对应嵌入矩阵的一行
        # 例如：input_ids = [[1, 3], [2, 0]]
        # 输出 = [[weight[1], weight[3]], [weight[2], weight[0]]]
        return self.weight[input_ids]

    def __call__(self, input_ids: np.ndarray) -> np.ndarray:
        """支持函数式调用"""
        return self.forward(input_ids)


def test_embedding():
    """测试词嵌入模块"""
    # 超参数
    vocab_size = 1000   # 词汇表大小
    hidden_size = 64    # 嵌入维度
    batch_size = 2      # 批次大小
    seq_len = 10        # 序列长度

    # 创建嵌入层
    embedding = Embedding(vocab_size, hidden_size)

    # 模拟输入：随机 token IDs
    input_ids = np.random.randint(0, vocab_size, size=(batch_size, seq_len))

    # 前向传播
    output = embedding(input_ids)

    # 验证输出形状
    print(f"输入形状: {input_ids.shape}")   # (2, 10)
    print(f"输出形状: {output.shape}")     # (2, 10, 64)
    print(f"嵌入矩阵形状: {embedding.weight.shape}")  # (1000, 64)

    # 验证查找正确性
    # 检查第一个 token 的嵌入向量是否与权重矩阵中对应行相同
    first_token_id = input_ids[0, 0]
    first_embedding = output[0, 0]
    expected_embedding = embedding.weight[first_token_id]
    assert np.array_equal(first_embedding, expected_embedding), "嵌入查找不正确！"

    print("\n✓ 所有测试通过！")


if __name__ == '__main__':
    test_embedding()
