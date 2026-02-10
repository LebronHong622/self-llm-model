"""
@Author: Lebron Hong
@Date: 2026-01-29
@LastEditTime: 2026-01-29
@Description: 大模型基础的一些基础层设置
"""

import torch
import torch.nn as nn
import math
from config.config import ModelConfig

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
    多头自注意力机制层 (Multi-Head Self-Attention)。
    
    这是 Transformer 架构的核心组件，通过并行计算多组注意力头来捕获输入序列中
    不同子空间的信息。每个注意力头独立学习不同的注意力模式，最终合并所有头的输出。
    
    Args:
        input_size (int): 输入特征的维度大小。
        hidden_size (int): 隐藏层维度，也是 Q/K/V 投影后的维度。
        num_heads (int): 注意力头的数量，hidden_size 必须能被其整除。
        context_length (int): 支持的最大序列长度。
        bias (bool, optional): 线性层是否使用偏置项，默认为 False。
        dropout (float, optional): Dropout 概率，默认为 0.1。
    
    Example:
        >>> batch_size, seq_len, input_size, hidden_size = 2, 10, 512, 512
        >>> num_heads, context_length = 8, 512
        >>> mha = MultiHeadAttention(input_size, hidden_size, num_heads, context_length)
        >>> x = torch.randn(batch_size, seq_len, input_size)
        >>> output = mha(x)
        >>> print(output.shape)  # torch.Size([2, 10, 512])
    
    Reference:
        "Attention Is All You Need" - Vaswani et al., 2017
        https://arxiv.org/abs/1706.03762
    """
    def __init__(self, input_size, hidden_size, num_heads, context_length, 
            bias=False, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.context_length = context_length
        self.heads_size = hidden_size // num_heads

        # 自注意力层参数构建
        self.W_k = nn.Linear(input_size, hidden_size, bias=bias)
        self.W_q = nn.Linear(input_size, hidden_size, bias=bias)
        self.W_v = nn.Linear(input_size, hidden_size, bias=bias)

        # 输出层参数构建
        self.W_o = nn.Linear(hidden_size, hidden_size, bias=bias)

        # 进行掩码处理，使用register_buffer来注册一个缓冲区，
        # 该缓冲区将被保存到磁盘，并在模型保存和加载时被序列化和反序列化。
        # 掩码矩阵，用于避免自注意力中的自注意力
        # 上对角为1，其余为0，用于避免自注意力中的自注意力
        self.register_buffer(
            "mask", 
            torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

        # dropout层
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        前向传播计算多头自注意力。
        
        执行完整的自注意力计算流程：投影 → 分头 → 注意力计算 → 掩码 → 
        Softmax → Dropout → 加权求和 → 合并头 → 输出投影。
        
        Args:
            x (torch.Tensor): 输入张量，形状为 (batch_size, seq_len, input_size)。
                            batch_size: 批次大小
                            seq_len: 序列长度（必须 <= context_length）
                            input_size: 输入特征维度
        
        Returns:
            torch.Tensor: 注意力层的输出，形状为 (batch_size, seq_len, hidden_size)。
                         输出与输入的 batch_size 和 seq_len 保持一致。
        
        Raises:
            IndexError: 如果 seq_len 超过预定义的 context_length。
            RuntimeError: 如果输入维度不匹配。
        
        Note:
            当前实现使用就地操作 (masked_fill_) 来节省内存。
            缩放因子 sqrt(d_k) 用于防止点积过大导致 Softmax 梯度消失。
        """
        batch_size, seq_len, input_size = x.shape

        keys = self.W_k(x)
        querys = self.W_q(x)
        values = self.W_v(x)

        # 将输入张量分成num_heads份，并reshape为(batch_size, seq_len, num_heads, head_size)
        keys = keys.view(batch_size, seq_len, self.num_heads, self.heads_size)
        querys = querys.view(batch_size, seq_len, self.num_heads, self.heads_size)
        values = values.view(batch_size, seq_len, self.num_heads, self.heads_size)

        # 将num_heads维度移到第2维，即(batch_size, num_heads, seq_len, head_size)
        keys = keys.transpose(1, 2)
        querys = querys.transpose(1, 2)
        values = values.transpose(1, 2)
        
        # 计算注意力分数
        attention_scores = querys @ keys.transpose(-2, -1)

        # 增加掩码（就地执行，不占用内存）
        mask_bool = self.mask.bool()[:seq_len, :seq_len]
        attention_scores.masked_fill_(mask_bool, -torch.inf)

        # 针对对嵌入维度进行归一化，避免梯度过小
        attention_scores /= math.sqrt(keys.shape[-1])

        # 计算注意力权重，使用softmax
        attention_weights = torch.softmax(attention_scores, dim=-1)

        # dropout化注意力权重
        attention_weights = self.dropout(attention_weights)

        # 计算上下文向量，并转置回(batch_size, seq_len, num_heads, head_size)
        context_vector = (attention_weights @ values).transpose(1, 2)

        # 将num_heads维度移到第3维，即(batch_size, seq_len, hidden_size)
        context_vector = context_vector.reshape(batch_size, seq_len, self.hidden_size)

        # 输出层
        output = self.W_o(context_vector)

        return output

class TransformerBlock(nn.Module):
    """
    Transformer 块，包含一个多头自注意力层和一个前馈神经网络层。
    
    Args:
        input_size (int): 输入特征的维度大小。
        hidden_size (int): 隐藏层维度，也是 Q/K/V 投影后的维度。
        num_heads (int): 注意力头的数量，hidden_size 必须能被其整除。
        context_length (int): 支持的最大序列长度。
        dropout (float, optional): Dropout 概率，默认为 0.1。
    
    Example:
        >>> batch_size, seq_len, input_size, hidden_size = 2, 10, 512, 512
        >>> num_heads, context_length = 8, 512
        >>> block = TransformerBlock(input_size, hidden_size, num_heads, context_length)
        >>> x = torch.randn(batch_size, seq_len, input
    """
    
    def __init__(self, config: ModelConfig):
        super(TransformerBlock, self).__init__()
        self.layer_norm1 = NormLayer(config.input_size, config.norm_eps)

        self.mha = MultiHeadAttention(
            config.input_size, 
            config.hidden_size, 
            config.num_heads, 
            config.context_length,
            config.qwk_bias,
            config.dropout
        )

        self.dropout1 = nn.Dropout(config.dropout)

        self.layer_norm2 = NormLayer(config.hidden_size, config.norm_eps)

        self.ffn = FeedForwardLayer(config.hidden_size, config.expansion_factor)

        self.dropout2 = nn.Dropout(config.dropout)
    
    def forward(self, x):
        """
        前向传播计算 Transformer 块。
        
        执行完整的 Transformer 块计算流程：
        注意力 → 层归一化 → Dropout → 前馈神经网络 → 层归一化 → Dropout → 输出。
        
        Args:
            x (torch.Tensor): 输入
        """
        shortcut = x
        x = self.layer_norm1(x)
        x = self.mha(x)
        x = self.dropout1(x)
        x = x + shortcut

        shortcut = x
        x = self.layer_norm2(x)
        x = self.ffn(x)
        x = self.dropout2(x)
        x = x + shortcut
        return x

class NormLayer(nn.Module):
    """
    层归一化层 (Layer Normalization)。
    
    层归一化通过对每个样本的特征进行归一化，使其具有零均值和单位方差。
    它在 Transformer 中被广泛使用，以提高模型的性能和稳定性。
    
    Args:
        emb_size (int): 输入特征的维度大小。
        eps (float, optional): 防止除零错误的极小值，默认为 1e-6。
    
    Example:
        >>> batch_size, seq_len, emb_size = 2, 10, 512
        >>> norm = NormLayer(emb_size)
        >>> x = torch.randn(batch_size, seq_len, emb_size)
        >>> output = norm(x)
        >>> print(output.shape)  # torch.Size([2, 10, 512])
    """
    
    def __init__(self, emb_size: int, eps: float = 1e-5):
        super(NormLayer, self).__init__()
        self.eps = eps
        # 可学习参数，用于缩放和偏移
        self.gamma = nn.Parameter(torch.ones(emb_size))
        self.beta = nn.Parameter(torch.zeros(emb_size))
    
    def forward(self, x):
        """
        前向传播计算层归一化。
        
        执行完整的层归一化计算流程：
        计算均值和方差 → 归一化 → 缩放和偏移。
        
        Args:
            x (torch.Tensor): 输入张量，形状为 (batch_size, seq_len, input_size)。
                            batch_size: 批次大小
                            seq_len: 序列长度
                            input_size: 输入特征维度
        
        Returns:
            torch.Tensor: 层归一化后的张量，形状为 (batch_size, seq_len, input_size)。
                         输出与输入的 batch_size 和 seq_len
        """
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / (std + self.eps)
        return self.gamma * norm_x + self.beta

class GELU(nn.Module):
    """
    GELU 激活函数 (Gaussian Error Linear Unit)。
    
    GELU 是最近在 Transformer 中被提出的一种激活函数，
    它通过将输入映射到正态分布来模拟非线性。
    
    Example:
        >>> batch_size, seq_len, emb_size = 2, 10, 512
        >>> gelu = GELU()
        >>> x = torch.randn(batch_size, seq_len, emb_size)
        >>> output = gelu(x)
        >>> print(output.shape)  # torch.Size([2, 10, 512])
    """
    def __init__(self):
        super(GELU, self).__init__()
    
    def forward(self, x):
        """
        前向传播计算 GELU。
        
        Args:
            x (torch.Tensor): 输入张量，形状为 (batch_size, seq_len, input_size)。
                            batch_size: 批次大小
                            se
        """

        return 0.5 * x *(1 + torch.tanh(torch.sqrt(torch.tensor(2 / torch.pi)) * (x + 0.044715 * torch.pow(x, 3))))

class FeedForwardLayer(nn.Module):
    """
    前馈神经网络层 (Feed Forward Layer)。
    
    前馈神经网络层是 Transformer 中常用的结构，
    它通过两个线性层和一个激活函数来实现非线性变换。
    
    Args:
        emb_size (int): 输入特征的维度大小。
        expansion_factor (int, optional): 扩展因子，默认为 4。
    
    Example:
        >>> batch_size, seq_len, emb_size = 2, 10, 512
        >>> ff = FeedForwardLayer(emb_size)
        >>> x = torch.randn(batch_size, seq_len, emb_size)
        >>> output = ff(x)
        >>> print(output.shape)  # torch.Size([2, 10, 512])
    """
    
    def __init__(self, emb_size: int, expansion_factor: int = 4):
        super(FeedForwardLayer, self).__init__()
        self.feed_forward_layer = nn.Sequential(
            nn.Linear(emb_size, expansion_factor * emb_size),
            nn.GELU(),
            nn.Linear(expansion_factor * emb_size, emb_size),
        )
    
    def forward(self, x):
        """
        前向传播计算前馈神经网络层。
        
        Args:
            x (torch.Tensor): 输入张量，形状为 (batch_size, seq_len, input_size)。
                            batch_size: 批次大小
                            seq_len: 序列长度
                            input_size: 输入特征维度
        
        Returns:
            torch.Tensor: 前馈神经网络层的输出，形状为 (batch_size, seq_len, input_size)。
                         输出与输入的 batch_size 和 seq_len 保持一致。
        """
        return self.feed_forward_layer(x)
