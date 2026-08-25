"""
交叉熵损失模块 (Cross-Entropy Loss)

衡量预测分布与真实分布之间的差异。
是分类任务和语言模型训练的标准损失函数。
"""

import numpy as np
from typing import Union


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """数值稳定的 softmax（内部使用）"""
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def log_softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """数值稳定的 log_softmax（内部使用）"""
    x_max = np.max(x, axis=axis, keepdims=True)
    log_sum_exp = np.log(np.sum(np.exp(x - x_max), axis=axis, keepdims=True))
    return (x - x_max) - log_sum_exp


def cross_entropy_loss(
    logits: np.ndarray,
    targets: Union[np.ndarray, int],
    reduction: str = 'mean'
) -> np.ndarray:
    """
    交叉熵损失函数

    公式: CE = -sum(y_true * log(y_pred))

    对于单标签分类（targets 为整数）:
    CE = -log(softmax(logits)[target])

    参数:
        logits: 形状为 [..., vocab_size] 的未归一化分数
        targets: 真实标签（整数或 one-hot 向量）
        reduction: 'mean' | 'sum' | 'none'

    返回:
        损失值（标量或张量）
    """
    # 计算 log_softmax（数值更稳定）
    log_probs = log_softmax(logits, axis=-1)

    # 处理不同类型的 targets
    if isinstance(targets, (int, np.integer)):
        # 单个整数标签
        loss = -log_probs[..., targets]
    elif targets.ndim == logits.ndim:
        # one-hot 向量
        loss = -np.sum(targets * log_probs, axis=-1)
    else:
        # 整数标签数组
        # 使用高级索引正确处理批量情况
        # 例如: logits shape [2, 3], targets shape [2]
        # 需要取 log_probs[0, targets[0]], log_probs[1, targets[1]]
        batch_indices = np.arange(logits.shape[0])
        loss = -log_probs[batch_indices, targets]

    # 应用 reduction
    if reduction == 'mean':
        return np.mean(loss)
    elif reduction == 'sum':
        return np.sum(loss)
    else:  # 'none'
        return loss


def perplexity(loss: float) -> float:
    """
    困惑度 (Perplexity)

    公式: PPL = exp(CE)

    困惑度越低，模型预测越准确。
    直观理解：模型在每个位置平均有 PPL 个等可能的选择。

    参数:
        loss: 交叉熵损失

    返回:
        困惑度
    """
    return np.exp(loss)


def test_cross_entropy():
    """测试交叉熵损失函数"""
    print("=" * 50)
    print("测试 1: 基本交叉熵")
    print("=" * 50)

    # 模拟 3 分类问题
    logits = np.array([2.0, 1.0, 0.5])
    target = 0  # 真实类别是第 0 类

    loss = cross_entropy_loss(logits, target)
    print(f"logits: {logits}")
    print(f"target: {target}")
    print(f"loss: {loss:.4f}")

    # 验证：loss 应该是正数
    assert loss > 0, "损失应该为正数"

    # 手动计算验证
    probs = softmax(logits)
    expected_loss = -np.log(probs[target])
    assert np.isclose(loss, expected_loss), "损失计算不正确"

    print("\n" + "=" * 50)
    print("测试 2: 完美预测 vs 错误预测")
    print("=" * 50)

    # 完美预测：logits 中目标类别的值最大
    logits_perfect = np.array([10.0, 0.0, 0.0])
    loss_perfect = cross_entropy_loss(logits_perfect, 0)
    print(f"完美预测损失: {loss_perfect:.4f}")

    # 错误预测：logits 中目标类别的值最小
    logits_wrong = np.array([0.0, 0.0, 10.0])
    loss_wrong = cross_entropy_loss(logits_wrong, 0)
    print(f"错误预测损失: {loss_wrong:.4f}")

    # 完美预测的损失应该更小
    assert loss_perfect < loss_wrong, "完美预测的损失应该更小"

    print("\n" + "=" * 50)
    print("测试 3: 批量处理")
    print("=" * 50)

    # batch_size=2, vocab_size=3
    logits_batch = np.array([
        [2.0, 1.0, 0.5],
        [0.5, 2.0, 1.0]
    ])
    targets_batch = np.array([0, 1])

    loss_batch = cross_entropy_loss(logits_batch, targets_batch)
    print(f"批量 logits:\n{logits_batch}")
    print(f"批量 targets: {targets_batch}")
    print(f"平均损失: {loss_batch:.4f}")

    # 验证：应该是两个样本损失的平均值
    loss_0 = cross_entropy_loss(logits_batch[0], targets_batch[0])
    loss_1 = cross_entropy_loss(logits_batch[1], targets_batch[1])
    expected = (loss_0 + loss_1) / 2
    assert np.isclose(loss_batch, expected), "批量损失应该是平均值"

    print("\n" + "=" * 50)
    print("测试 4: one-hot 标签")
    print("=" * 50)

    logits = np.array([2.0, 1.0, 0.5])
    target_onehot = np.array([1.0, 0.0, 0.0])

    loss_onehot = cross_entropy_loss(logits, target_onehot)
    loss_int = cross_entropy_loss(logits, 0)
    print(f"one-hot 损失: {loss_onehot:.4f}")
    print(f"整数标签损失: {loss_int:.4f}")
    assert np.isclose(loss_onehot, loss_int), "one-hot 和整数标签应该得到相同损失"

    print("\n" + "=" * 50)
    print("测试 5: 困惑度")
    print("=" * 50)

    loss = 2.0
    ppl = perplexity(loss)
    print(f"损失: {loss}")
    print(f"困惑度: {ppl:.2f}")
    assert np.isclose(ppl, np.exp(2.0)), "困惑度计算不正确"

    # 困惑度的直观理解
    print(f"\n困惑度 {ppl:.2f} 意味着：模型在每个位置平均有 {ppl:.0f} 个等可能的选择")

    print("\n" + "=" * 50)
    print("测试 6: 数值稳定性")
    print("=" * 50)

    # 非常大的 logits
    logits_large = np.array([1000.0, 1001.0, 1002.0])
    loss_large = cross_entropy_loss(logits_large, 0)
    print(f"大数值 logits 损失: {loss_large:.4f}")
    assert not np.isnan(loss_large), "不应该有 NaN"
    assert not np.isinf(loss_large), "不应该有 Inf"

    print("\n✓ 所有测试通过！")


if __name__ == '__main__':
    test_cross_entropy()
