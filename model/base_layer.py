"""
@Author: Lebron Hong
@Date: 2026-01-29
@LastEditTime: 2026-01-29
@Description: 大模型基础的一些层设置
"""

import torch
import torch.nn as nn
import math

# embedding层+绝对位置编码层
class InputEmbeddingLayer(nn.Module):
    """
    输入嵌入层,结合词嵌入和绝对位置编码。

    该层将输入的 token IDs 转换为词向量表示,并添加绝对位置编码信息。

    Attributes:
        embedding (nn.Embedding): 词嵌入层,将 token ID 映射到词向量。
        context_length (int): 最大序列长度(上下文长度)。
        position_embedding (nn.Embedding): 绝对位置嵌入层,为每个位置添加位置信息。

    Args:
        vocab_size (int): 词表大小,决定 token ID 的范围 [0, vocab_size-1]。
        hidden_size (int): 嵌入向量的维度大小,词向量和位置向量均为该维度。
        max_length (int): 模型支持的最大序列长度,用于位置嵌入的数量。

    Example:
        >>> vocab_size, hidden_size, max_length = 1000, 768, 512
        >>> embedding_layer = InputEmbeddingLayer(vocab_size, hidden_size, max_length)
        >>> input_ids = torch.randint(0, vocab_size, (2, 10))  # batch_size=2, seq_len=10
        >>> output = embedding_layer(input_ids)
        >>> print(output.shape)  # torch.Size([2, 10, 768])
    """

    def __init__(self, vocab_size, hidden_size, max_length):
        super(InputEmbeddingLayer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.context_length = max_length
        self.position_embedding = nn.Embedding(self.context_length, hidden_size)

    def forward(self, input_ids):
        """
        前向传播计算。

        将输入 token IDs 转换为词向量,并添加对应的位置编码。

        Args:
            input_ids (torch.Tensor): 输入 token IDs,形状为 (batch_size, seq_len)。
                                     每个元素的值应在 [0, vocab_size) 范围内。

        Returns:
            torch.Tensor: 嵌入后的张量,形状为 (batch_size, seq_len, hidden_size)。
                         包含词嵌入和位置编码的和。

        Raises:
            IndexError: 如果 input_ids 中的值超出 [0, vocab_size) 范围。
            IndexError: 如果 seq_len 超过 max_length。
        """
        output = self.embedding(input_ids) \
            + self.position_embedding(torch.arange(0, input_ids.size(1)))
        return output

class MultiHeadAttention(nn.Module):
    """
    多头注意力机制层。
    """
    def __init__(self, input_size, hidden_size, context_length, 
            bias=False, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        # 自注意力层参数构建
        self.W_k = nn.Linear(input_size, hidden_size, bias=bias)
        self.W_q = nn.Linear(input_size, hidden_size, bias=bias)
        self.W_v = nn.Linear(input_size, hidden_size, bias=bias)

        # 进行掩码处理
        self.register_buffer(
            "mask", 
            torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

        # dropout层
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        batch_size, seq_len, input_size = x.shape

        keys = self.W_k(x)
        querys = self.W_q(x)
        values = self.W_v(x)
        
        # 计算注意力分数
        attention_scores = querys @ keys.transpose(-2, -1)

        # 增加掩码（就地执行，不占用内存）
        attention_scores.masked_fill_(self.mask.bool()[:seq_len, :seq_len], -torch.inf)

        ## 针对对嵌入维度进行归一化，避免梯度过小
        attention_scores /= math.sqrt(keys.shape[-1])

        # 计算注意力权重，使用softmax
        attention_weights = torch.softmax(attention_scores, dim=-1)

        # dropout化注意力权重
        attention_weights = self.dropout(attention_weights)

        # 计算上下文向量
        context_vector = attention_weights @ values

        return context_vector

if __name__ == "__main__":
    batch_size, seq_len, input_size, hidden_size = 2, 6, 6, 2

    attention_layer = MultiHeadAttention(input_size, hidden_size, seq_len, dropout=0.5)
    input_tensor = torch.randn(batch_size, seq_len, input_size)
    output = attention_layer(input_tensor)
    print(output.shape)