"""
GPT 模块 (Generative Pre-trained Transformer)

使用 PyTorch 实现的完整 GPT 模型。
组合了前面所有模块：Embedding, Positional Encoding, Transformer Block, etc.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional


class MultiHeadAttention(nn.Module):
    """多头自注意力（PyTorch 版本）"""

    def __init__(self, hidden_size: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        assert hidden_size % num_heads == 0

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        # Q, K, V 投影
        self.W_q = nn.Linear(hidden_size, hidden_size)
        self.W_k = nn.Linear(hidden_size, hidden_size)
        self.W_v = nn.Linear(hidden_size, hidden_size)

        # 输出投影
        self.W_o = nn.Linear(hidden_size, hidden_size)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        use_causal_mask: bool = False
    ) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        # Q, K, V 投影
        Q = self.W_q(x)  # [B, S, H]
        K = self.W_k(x)  # [B, S, H]
        V = self.W_v(x)  # [B, S, H]

        # 分头: [B, S, H] → [B, num_heads, S, head_dim]
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # 计算注意力分数
        # scores = Q @ K^T / sqrt(d_k)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # 因果掩码
        if use_causal_mask:
            mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device))
            mask = mask.unsqueeze(0).unsqueeze(0)  # [1, 1, S, S]
            scores = scores.masked_fill(mask == 0, float('-inf'))

        # Softmax + Dropout
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # 加权求和
        output = torch.matmul(attention_weights, V)  # [B, num_heads, S, head_dim]

        # 合并头: [B, num_heads, S, head_dim] → [B, S, H]
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)

        # 输出投影
        return self.W_o(output)


class FeedForward(nn.Module):
    """前馈网络（PyTorch 版本）"""

    def __init__(self, hidden_size: int, ffn_size: int = None, dropout: float = 0.0):
        super().__init__()
        ffn_size = ffn_size or 4 * hidden_size

        self.linear1 = nn.Linear(hidden_size, ffn_size)
        self.linear2 = nn.Linear(ffn_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class TransformerBlock(nn.Module):
    """Transformer 块（Pre-Norm 结构）"""

    def __init__(self, hidden_size: int, num_heads: int, ffn_size: int = None, dropout: float = 0.0):
        super().__init__()

        # LayerNorm
        self.ln1 = nn.LayerNorm(hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)

        # 注意力
        self.attention = MultiHeadAttention(hidden_size, num_heads, dropout)
        # 前馈网络
        self.ffn = FeedForward(hidden_size, ffn_size, dropout)

    def forward(self, x: torch.Tensor, use_causal_mask: bool = True) -> torch.Tensor:
        # 注意力子层（Pre-Norm + 残差）
        residual = x
        x = self.ln1(x)
        x = self.attention(x, use_causal_mask=use_causal_mask)
        x = x + residual

        # 前馈子层（Pre-Norm + 残差）
        residual = x
        x = self.ln2(x)
        x = self.ffn(x)
        x = x + residual

        return x


class GPT(nn.Module):
    """
    GPT 模型

    结构:
    - Token Embedding + Positional Embedding
    - N 层 TransformerBlock
    - 最终 LayerNorm
    - LM Head（线性投影到词汇表）

    参数:
        vocab_size: 词汇表大小
        hidden_size: 隐藏层维度
        num_layers: Transformer 层数
        num_heads: 注意力头数
        max_seq_len: 最大序列长度
        dropout: Dropout 概率
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int = 128,
        num_layers: int = 4,
        num_heads: int = 4,
        max_seq_len: int = 256,
        dropout: float = 0.1
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.max_seq_len = max_seq_len

        # Token Embedding
        self.token_embedding = nn.Embedding(vocab_size, hidden_size)

        # Positional Embedding（可学习）
        self.position_embedding = nn.Embedding(max_seq_len, hidden_size)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Transformer 层
        self.layers = nn.ModuleList([
            TransformerBlock(hidden_size, num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])

        # 最终 LayerNorm
        self.ln_final = nn.LayerNorm(hidden_size)

        # LM Head（语言模型头）
        # 通常与 token_embedding 共享权重
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

        # 权重绑定：LM Head 与 Token Embedding 共享权重
        self.lm_head.weight = self.token_embedding.weight

        # 初始化权重
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """初始化权重"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        前向传播

        参数:
            input_ids: 输入 token IDs [batch_size, seq_len]
            targets: 目标 token IDs [batch_size, seq_len]（训练时使用）

        返回:
            logits: [batch_size, seq_len, vocab_size]
            loss: 如果提供 targets，返回损失值
        """
        batch_size, seq_len = input_ids.shape

        # 位置索引
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)

        # Embedding
        token_emb = self.token_embedding(input_ids)      # [B, S, H]
        pos_emb = self.position_embedding(positions)      # [1, S, H]

        # Token + Position Embedding
        x = self.dropout(token_emb + pos_emb)

        # Transformer 层
        for layer in self.layers:
            x = layer(x, use_causal_mask=True)

        # 最终 LayerNorm
        x = self.ln_final(x)

        # LM Head
        logits = self.lm_head(x)  # [B, S, vocab_size]

        # 计算损失（如果提供 targets）
        loss = None
        if targets is not None:
            # 重塑为 [B*S, vocab_size] 和 [B*S]
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1)
            )

        return logits, loss

    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: Optional[int] = None
    ) -> torch.Tensor:
        """
        自回归生成

        参数:
            input_ids: 起始 token IDs [1, seq_len]
            max_new_tokens: 最大生成 token 数
            temperature: 温度参数
            top_k: Top-k 采样

        返回:
            生成的 token IDs [1, seq_len + max_new_tokens]
        """
        self.eval()

        with torch.no_grad():
            for _ in range(max_new_tokens):
                # 截断到最大长度
                input_crop = input_ids[:, -self.max_seq_len:]

                # 前向传播
                logits, _ = self(input_crop)

                # 取最后一个位置的 logits
                logits = logits[:, -1, :] / temperature

                # Top-k 采样
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = float('-inf')

                # Softmax + 采样
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

                # 拼接
                input_ids = torch.cat([input_ids, next_token], dim=1)

        return input_ids


def test_gpt_model():
    """测试 GPT 模型"""
    # 超参数
    vocab_size = 1000
    hidden_size = 64
    num_layers = 2
    num_heads = 4
    max_seq_len = 32
    batch_size = 2
    seq_len = 10

    # 创建模型
    model = GPT(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_heads=num_heads,
        max_seq_len=max_seq_len
    )

    print("=" * 50)
    print("测试 1: 前向传播")
    print("=" * 50)

    # 模拟输入
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    targets = torch.randint(0, vocab_size, (batch_size, seq_len))

    # 前向传播
    logits, loss = model(input_ids, targets)

    print(f"输入形状: {input_ids.shape}")       # (2, 10)
    print(f"输出形状: {logits.shape}")          # (2, 10, 1000)
    print(f"损失值: {loss.item():.4f}")

    assert logits.shape == (batch_size, seq_len, vocab_size)
    assert loss is not None

    print("\n" + "=" * 50)
    print("测试 2: 模型参数统计")
    print("=" * 50)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"总参数数量: {total_params:,}")
    print(f"可训练参数: {trainable_params:,}")

    print("\n" + "=" * 50)
    print("测试 3: 文本生成")
    print("=" * 50)

    # 起始 token
    start_tokens = torch.randint(0, vocab_size, (1, 5))

    # 生成
    generated = model.generate(start_tokens, max_new_tokens=10, temperature=0.8)

    print(f"起始 tokens: {start_tokens[0].tolist()}")
    print(f"生成 tokens: {generated[0].tolist()}")
    print(f"生成长度: {generated.shape[1]}")

    assert generated.shape[1] == start_tokens.shape[1] + 10

    print("\n" + "=" * 50)
    print("测试 4: 模型结构")
    print("=" * 50)

    print("模型结构:")
    print(model)
    print(f"\n层数: {len(model.layers)}")
    print(f"隐藏维度: {model.hidden_size}")
    print(f"最大序列长度: {model.max_seq_len}")

    print("\n✓ 所有测试通过！")


if __name__ == '__main__':
    test_gpt_model()
