# -*- coding: utf-8 -*-
"""
@Date: 2026-02-13
@Description: Unit tests for calc_loss_loader function in train_utils.py
"""
import pytest
import torch
from torch.utils.data import TensorDataset, DataLoader
from model.base_layer import InputEmbeddingLayer
from trainer.train_utils import calc_loss_loader


class MockModel(torch.nn.Module):
    """Mock model for testing"""

    def __init__(self, vocab_size, hidden_size):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, hidden_size)
        self.linear = torch.nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        return self.linear(x)


class TestCalcLossLoader:
    """测试 calc_loss_loader 函数"""

    @pytest.fixture
    def device(self):
        """测试设备"""
        return torch.device('cpu')

    @pytest.fixture
    def mock_data(self):
        """生成模拟数据"""
        batch_size = 4
        seq_len = 10
        vocab_size = 100

        inputs = torch.randint(0, vocab_size, (batch_size * 10, seq_len))
        targets = torch.randint(0, vocab_size, (batch_size * 10, seq_len))

        return inputs, targets, vocab_size

    @pytest.fixture
    def mock_dataloader(self, mock_data):
        """创建模拟数据加载器"""
        inputs, targets, _ = mock_data
        dataset = TensorDataset(inputs, targets)
        dataloader = DataLoader(dataset, batch_size=4, shuffle=False)
        return dataloader

    @pytest.fixture
    def mock_model(self, mock_data):
        """创建模拟模型"""
        _, _, vocab_size = mock_data
        hidden_size = 768
        return MockModel(vocab_size, hidden_size)

    def test_calc_loss_all_batches(self, mock_dataloader, mock_model, device):
        """测试计算所有批次的损失"""
        loss = calc_loss_loader(mock_dataloader, mock_model, device)

        assert isinstance(loss, float)
        assert not torch.isnan(torch.tensor(loss))
        assert loss > 0

    def test_calc_loss_specific_batches(self, mock_dataloader, mock_model, device):
        """测试计算指定批次数的损失"""
        num_batches = 3
        loss = calc_loss_loader(mock_dataloader, mock_model, device, num_batch=num_batches)

        assert isinstance(loss, float)
        assert not torch.isnan(torch.tensor(loss))
        assert loss > 0

    def test_calc_loss_single_batch(self, mock_dataloader, mock_model, device):
        """测试计算单个批次的损失"""
        num_batches = 1
        loss = calc_loss_loader(mock_dataloader, mock_model, device, num_batch=num_batches)

        assert isinstance(loss, float)
        assert not torch.isnan(torch.tensor(loss))
        assert loss > 0

    def test_calc_loss_zero_batches(self, mock_data, mock_model, device):
        """测试零批次时返回 nan"""
        inputs, targets, _ = mock_data
        dataset = TensorDataset(inputs, targets)
        dataloader = DataLoader(dataset, batch_size=4, shuffle=False)

        # 创建空的数据加载器
        empty_dataloader = DataLoader(TensorDataset(
            torch.empty(0, 10, dtype=torch.long),
            torch.empty(0, 10, dtype=torch.long)
        ), batch_size=4)

        loss = calc_loss_loader(empty_dataloader, mock_model, device, num_batch=0)

        assert isinstance(loss, float)
        assert torch.isnan(torch.tensor(loss))

    def test_calc_loss_with_none_num_batch(self, mock_dataloader, mock_model, device):
        """测试 num_batch 为 None 时计算所有批次"""
        loss_all = calc_loss_loader(mock_dataloader, mock_model, device, num_batch=None)
        loss_explicit = calc_loss_loader(mock_dataloader, mock_model, device, num_batch=len(mock_dataloader))

        assert loss_all == loss_explicit

    def test_calc_loss_consistency(self, mock_dataloader, mock_model, device):
        """测试多次调用结果一致（模型无随机性时）"""
        torch.manual_seed(42)
        loss1 = calc_loss_loader(mock_dataloader, mock_model, device)

        torch.manual_seed(42)
        loss2 = calc_loss_loader(mock_dataloader, mock_model, device)

        assert loss1 == loss2

    def test_calc_loss_large_dataset(self, device):
        """测试大数据集的损失计算"""
        batch_size = 8
        seq_len = 20
        vocab_size = 500
        num_samples = 100

        inputs = torch.randint(0, vocab_size, (num_samples, seq_len))
        targets = torch.randint(0, vocab_size, (num_samples, seq_len))

        dataset = TensorDataset(inputs, targets)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        model = MockModel(vocab_size, 1024)

        loss = calc_loss_loader(dataloader, model, device)

        assert isinstance(loss, float)
        assert not torch.isnan(torch.tensor(loss))
        assert loss > 0

    def test_calc_loss_num_batch_exceeds_dataloader(self, mock_dataloader, mock_model, device):
        """测试 num_batch 超过 dataloader 实际批次数"""
        num_batches = len(mock_dataloader) + 10
        loss = calc_loss_loader(mock_dataloader, mock_model, device, num_batch=num_batches)

        assert isinstance(loss, float)
        assert not torch.isnan(torch.tensor(loss))
        assert loss > 0
