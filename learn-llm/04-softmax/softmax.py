"""
Softmax 函数模块

将任意实数向量转换为概率分布。
是注意力机制和分类任务的核心组件。
"""

import numpy as np


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    数值稳定的 Softmax 函数

    公式: softmax(x_i) = exp(x_i) / sum(exp(x_j))

    为避免数值溢出，使用减去最大值的技巧:
    softmax(x_i) = exp(x_i - max(x)) / sum(exp(x_j - max(x)))

    参数:
        x: 输入张量
        axis: 计算 softmax 的维度（默认最后一维）

    返回:
        与 x 形状相同的概率分布，指定维度上和为 1
    """
    # 减去最大值，防止 exp 溢出
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)

    # 归一化
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def log_softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    Log-Softmax 函数（数值更稳定）

    公式: log_softmax(x_i) = x_i - log(sum(exp(x_j)))

    使用 log-sum-exp 技巧避免数值问题:
    log_softmax(x_i) = (x_i - max(x)) - log(sum(exp(x_j - max(x))))

    参数:
        x: 输入张量
        axis: 计算的维度

    返回:
        log 概率
    """
    x_max = np.max(x, axis=axis, keepdims=True)
    log_sum_exp = np.log(np.sum(np.exp(x - x_max), axis=axis, keepdims=True))
    return (x - x_max) - log_sum_exp


def test_softmax():
    """测试 Softmax 函数"""
    print("=" * 50)
    print("测试 1: 基本 softmax")
    print("=" * 50)

    x = np.array([1.0, 2.0, 3.0])
    result = softmax(x)
    print(f"输入: {x}")
    print(f"输出: {result}")
    print(f"和: {result.sum():.6f}")  # 应该为 1
    assert np.isclose(result.sum(), 1.0), "softmax 输出之和应该为 1"

    print("\n" + "=" * 50)
    print("测试 2: 大数值稳定性")
    print("=" * 50)

    x_large = np.array([1000.0, 1001.0, 1002.0])
    result_large = softmax(x_large)
    print(f"输入: {x_large}")
    print(f"输出: {result_large}")
    print(f"和: {result_large.sum():.6f}")
    assert not np.any(np.isnan(result_large)), "不应该有 NaN"
    assert np.isclose(result_large.sum(), 1.0), "大数值时 softmax 输出之和应该为 1"

    print("\n" + "=" * 50)
    print("测试 3: 批量处理")
    print("=" * 50)

    x_batch = np.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0]
    ])
    result_batch = softmax(x_batch, axis=-1)
    print(f"输入形状: {x_batch.shape}")
    print(f"输出形状: {result_batch.shape}")
    print(f"每行和: {result_batch.sum(axis=-1)}")  # 应该都是 [1, 1]
    assert np.allclose(result_batch.sum(axis=-1), 1.0), "每行之和应该为 1"

    print("\n" + "=" * 50)
    print("测试 4: log_softmax")
    print("=" * 50)

    x = np.array([1.0, 2.0, 3.0])
    log_probs = log_softmax(x)
    probs = softmax(x)
    print(f"softmax: {probs}")
    print(f"log_softmax: {log_probs}")
    print(f"exp(log_softmax): {np.exp(log_probs)}")
    assert np.allclose(np.exp(log_probs), probs), "exp(log_softmax) 应该等于 softmax"

    print("\n" + "=" * 50)
    print("测试 5: 温度参数效果")
    print("=" * 50)

    logits = np.array([2.0, 1.0, 0.5])

    # 低温 (T=0.5): 分布更尖锐
    probs_cold = softmax(logits / 0.5)
    print(f"低温 (T=0.5): {probs_cold}")

    # 正常温度 (T=1.0)
    probs_normal = softmax(logits / 1.0)
    print(f"正常 (T=1.0): {probs_normal}")

    # 高温 (T=2.0): 分布更平滑
    probs_hot = softmax(logits / 2.0)
    print(f"高温 (T=2.0): {probs_hot}")

    # 低温时最大值的概率应该更大
    assert probs_cold.max() > probs_normal.max() > probs_hot.max(), "温度越低，分布越尖锐"

    print("\n✓ 所有测试通过！")


if __name__ == '__main__':
    test_softmax()
