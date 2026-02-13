# -*- coding: utf-8 -*-
"""
@Date: 2026-01-29
@Description: Unit tests for InputEmbeddingLayer
"""
import pytest
import torch
from model.base_layer import InputEmbeddingLayer

class TestInputEmbeddingLayer:
    """测试 InputEmbeddingLayer 类的输出形状"""

    def test_output_shape_basic(self):
        """测试基本情况的输出形状"""
        vocab_size, hidden_size, context_length = 1000, 768, 512
        layer = InputEmbeddingLayer(vocab_size, hidden_size, context_length)
        
        batch_size, seq_len = 2, 10
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        
        output = layer(input_ids)
        assert output.shape == (batch_size, seq_len, hidden_size)

    def test_output_shape_single_batch(self):
        """测试单个样本的输出形状"""
        vocab_size, hidden_size, context_length = 1000, 512, 256
        layer = InputEmbeddingLayer(vocab_size, hidden_size, context_length)
        
        seq_len = 15
        input_ids = torch.randint(0, vocab_size, (1, seq_len))
        
        output = layer(input_ids)
        assert output.shape == (1, seq_len, hidden_size)

    def test_output_shape_large_batch(self):
        """测试大批量的输出形状"""
        vocab_size, hidden_size, context_length = 5000, 1024, 1024
        layer = InputEmbeddingLayer(vocab_size, hidden_size, context_length)
        
        batch_size, seq_len = 32, 20
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        
        output = layer(input_ids)
        assert output.shape == (batch_size, seq_len, hidden_size)

    def test_output_shape_max_length(self):
        """测试最大序列长度的输出形状"""
        vocab_size, hidden_size, context_length = 1000, 768, 512
        layer = InputEmbeddingLayer(vocab_size, hidden_size, context_length)

        batch_size = 4
        input_ids = torch.randint(0, vocab_size, (batch_size, context_length))

        output = layer(input_ids)
        assert output.shape == (batch_size, context_length, hidden_size)

    def test_output_shape_variable_lengths(self):
        """测试不同序列长度的输出形状"""
        vocab_size, hidden_size, context_length = 1000, 768, 512
        layer = InputEmbeddingLayer(vocab_size, hidden_size, context_length)
        
        test_cases = [
            (2, 1),
            (2, 5),
            (2, 10),
            (2, 50),
            (2, 100),
            (2, 256),
        ]
        
        for batch_size, seq_len in test_cases:
            input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
            output = layer(input_ids)
            assert output.shape == (batch_size, seq_len, hidden_size), \
                f"Failed for batch_size={batch_size}, seq_len={seq_len}"
