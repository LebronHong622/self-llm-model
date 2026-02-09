"""
GPT模型
"""

import torch
import torch.nn as nn
from config.config import ModelConfig
from model.base_layer import InputEmbeddingLayer, TransformerBlock, NormLayer

class SelfGPTModel(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        # 嵌入编码 + 绝对位置编码
        self.embedding = InputEmbeddingLayer(config.vocab_size, config.hidden_size, config.max_length)

        # dropout层
        self.dropout = nn.Dropout(config.dropout)

        # transformer块
        self.transformers = nn.Sequential(*[TransformerBlock(config) for _ in range(config.num_layers)])

        # 归一化层
        self.norm_layer = nn.LayerNorm(config.hidden_size)

        # 输出层
        self.output_layer = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, input_ids):
        # 嵌入编码 + 绝对位置编码
        embedding = self.embedding(input_ids)

        # dropout层
        embedding = self.dropout(embedding)

        # transformer块
        output = self.transformers(embedding)

        # 归一化层
        output = self.norm_layer(output)

        # 输出层
        output = self.output_layer(output)

        return output        
