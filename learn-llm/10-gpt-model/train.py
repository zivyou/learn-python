"""
GPT 训练脚本

使用简单的文本数据训练 GPT 模型。
演示完整的训练流程：数据准备 → 训练 → 生成。
"""

import torch
import torch.nn.functional as F
from torch.optim import AdamW
import numpy as np
from gpt_model import GPT


def create_sample_data():
    """
    创建示例训练数据

    使用简单的重复模式，让模型学习基本的序列规律。
    """
    # 简单的 token 序列（模拟）
    # 这里使用数字代表 token，实际应用中会使用 tokenizer
    text = """
    the cat sat on the mat
    the dog sat on the log
    the cat is very cute
    the dog is very happy
    a cat and a dog
    the mat is on the floor
    the log is in the forest
    """ * 10  # 重复多次增加数据量

    # 简单的字符级 tokenizer
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    char_to_idx = {ch: i for i, ch in enumerate(chars)}
    idx_to_char = {i: ch for i, ch in enumerate(chars)}

    # 编码
    data = [char_to_idx[ch] for ch in text]

    return data, vocab_size, char_to_idx, idx_to_char


def get_batch(data: list, batch_size: int, seq_len: int):
    """
    获取一个批次的数据

    参数:
        data: 数据列表
        batch_size: 批次大小
        seq_len: 序列长度

    返回:
        input_ids: 输入 token IDs
        targets: 目标 token IDs（向右移一位）
    """
    # 随机选择起始位置
    indices = np.random.randint(0, len(data) - seq_len - 1, size=batch_size)

    input_ids = []
    targets = []

    for idx in indices:
        input_ids.append(data[idx:idx + seq_len])
        targets.append(data[idx + 1:idx + seq_len + 1])

    return torch.tensor(input_ids), torch.tensor(targets)


def train():
    """训练 GPT 模型"""
    # 超参数
    vocab_size = 50       # 词汇表大小（根据数据调整）
    hidden_size = 128     # 隐藏层维度
    num_layers = 4        # Transformer 层数
    num_heads = 4         # 注意力头数
    max_seq_len = 64      # 最大序列长度
    batch_size = 32       # 批次大小
    learning_rate = 3e-4  # 学习率
    num_epochs = 10       # 训练轮数
    seq_len = 32          # 训练序列长度

    # 创建数据
    print("准备数据...")
    data, actual_vocab_size, char_to_idx, idx_to_char = create_sample_data()
    vocab_size = actual_vocab_size  # 使用实际的词汇表大小

    print(f"数据长度: {len(data)}")
    print(f"词汇表大小: {vocab_size}")

    # 创建模型
    print("\n创建模型...")
    model = GPT(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_heads=num_heads,
        max_seq_len=max_seq_len
    )

    # 统计参数
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数数量: {total_params:,}")

    # 优化器
    optimizer = AdamW(model.parameters(), lr=learning_rate)

    # 训练循环
    print("\n开始训练...")
    model.train()

    for epoch in range(num_epochs):
        total_loss = 0
        num_batches = 100  # 每个 epoch 的批次数量

        for batch_idx in range(num_batches):
            # 获取批次数据
            input_ids, targets = get_batch(data, batch_size, seq_len)

            # 前向传播
            logits, loss = model(input_ids, targets)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # 参数更新
            optimizer.step()

            total_loss += loss.item()

        # 打印训练信息
        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")

    # 生成示例
    print("\n" + "=" * 50)
    print("训练完成！生成示例：")
    print("=" * 50)

    model.eval()

    # 使用 "the " 作为起始
    start_text = "the "
    start_ids = [char_to_idx[ch] for ch in start_text]
    input_ids = torch.tensor([start_ids])

    # 生成
    with torch.no_grad():
        generated_ids = model.generate(
            input_ids,
            max_new_tokens=50,
            temperature=0.8,
            top_k=10
        )

    # 解码
    generated_text = "".join([idx_to_char[idx.item()] for idx in generated_ids[0]])
    print(f"起始文本: '{start_text}'")
    print(f"生成文本: '{generated_text}'")

    # 保存模型
    torch.save(model.state_dict(), "gpt_model.pth")
    print("\n模型已保存到 gpt_model.pth")


if __name__ == '__main__':
    train()
