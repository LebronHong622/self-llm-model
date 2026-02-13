# -*- coding: utf-8 -*-
# @Time    : 2026/01/27
# @Author  : <NAME>
# @File    : train_utils.py
# @Description: 训练工具函数
from torch.utils.data import DataLoader, Dataset
import torch

# dataloader加载数据公共函数
def create_dataloader(dataset: Dataset, batch_size: int, num_workers: int = 0, shuffle: bool = True, drop_last: bool = True):
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle, drop_last=drop_last)
    return dataloader

def calc_batch_loss(input_batch, target_batch, model, device):
    """
    计算批次的交叉熵损失
    
    Args:
        input_batch (Tensor): 输入数据批次张量
        target_batch (Tensor): 目标标签批次张量
        model (nn.Module): 用于预测的模型
        device (torch.device): 计算设备(如'cuda'或'cpu')
    
    Returns:
        Tensor: 计算得到的交叉熵损失值
    """
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    output_batch = model(input_batch)
    loss = torch.nn.functional.cross_entropy(
        output_batch.flatten(0, 1), 
        target_batch.flatten())
    return loss

def calc_loss_loader(data_loader, model, device, num_batch = None):
    """
    计算数据加载器中所有批次的交叉熵损失
    
    Args:
        data_loader (DataLoader): 数据加载器
        model (nn.Module): 用于预测的模型
        device (torch.device): 计算设备(如'cuda'或'cpu')
        num_batch (int, optional): 要计算的批次数(默认为None，即计算所有批次)
    
    Returns:
        Tensor: 计算得到的交叉熵损失值
    """
    total_loss = 0.0
    # dataloader中的batch数量获取
    num_batch = len(data_loader) if num_batch is None else num_batch
    if num_batch == 0:
        return float('nan')
    for batch_idx, (input_batch, target_batch) in enumerate(data_loader):
        if batch_idx >= num_batch:
            break
        loss = calc_batch_loss(input_batch, target_batch, model, device)
        total_loss += loss.item()
    return total_loss / num_batch
