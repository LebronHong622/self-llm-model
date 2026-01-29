"""
@Author: Lebron Hong
@Date: 2026-01-29
@LastEditTime: 2026-01-29
@Description: 大模型基础的一些层设置
"""

import torch
import torch.nn as nn

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