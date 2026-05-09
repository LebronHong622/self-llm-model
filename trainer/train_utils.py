# -*- coding: utf-8 -*-
# @Time    : 2026/01/27
# @Author  : <NAME>
# @File    : train_utils.py
# @Description: 训练工具函数
from torch.utils.data import DataLoader, Dataset
import torch
import tiktoken
from dataset.llm_dataset import PretrainedDataset

# dataloader加载数据公共函数
def create_dataloader(txt, batch_size, max_length, stride,
                         shuffle=True, drop_last=True, num_workers=0):
    # Initialize the tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")

    # Create dataset
    dataset = PretrainedDataset(txt, tokenizer, max_length, stride)

    # Create dataloader
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers)

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


# ==================== 分类任务工具函数 ====================

def calc_classification_batch_loss(input_batch, labels, model, device):
    """
    计算分类任务单批次的交叉熵损失。

    逻辑:
        1. 模型前向传播得到 logits [B, seq_len, num_classes]
        2. 取序列最后一个 token 的 logits: [B, num_classes]
        3. 与 labels 计算 CrossEntropyLoss

    Args:
        input_batch (Tensor): 输入 token ids，shape [B, seq_len]
        labels (Tensor):      分类标签，shape [B]
        model (nn.Module):    分类模型（输出头已替换为 num_classes）
        device (torch.device): 计算设备

    Returns:
        Tensor: 标量损失值
    """
    input_batch = input_batch.to(device)
    labels = labels.to(device)

    logits = model(input_batch)        # [B, seq_len, num_classes]
    cls_logits = logits[:, -1, :]      # [B, num_classes] 取最后 token

    loss = torch.nn.functional.cross_entropy(cls_logits, labels)
    return loss


def calc_classification_loss_loader(data_loader, model, device, num_batch=None):
    """
    计算分类任务数据加载器中的平均交叉熵损失。

    Args:
        data_loader (DataLoader): 分类数据集加载器（返回 input_ids, labels）
        model (nn.Module):        分类模型
        device (torch.device):    计算设备
        num_batch (int, optional): 限制计算的 batch 数，None 表示全部

    Returns:
        float: 平均损失值
    """
    total_loss = 0.0
    num_batch = len(data_loader) if num_batch is None else num_batch

    if num_batch == 0:
        return float('nan')

    for batch_idx, (input_batch, labels) in enumerate(data_loader):
        if batch_idx >= num_batch:
            break
        loss = calc_classification_batch_loss(input_batch, labels, model, device)
        total_loss += loss.item()

    return total_loss / num_batch


def calc_classification_accuracy_loader(data_loader, model, device, num_batch=None):
    """
    计算分类任务数据加载器中的准确率。

    Args:
        data_loader (DataLoader): 分类数据集加载器（返回 input_ids, labels）
        model (nn.Module):        分类模型
        device (torch.device):    计算设备
        num_batch (int, optional): 限制计算的 batch 数，None 表示全部

    Returns:
        float: 准确率（0.0 ~ 1.0）
    """
    model.eval()
    correct = 0
    total = 0
    num_batch = len(data_loader) if num_batch is None else num_batch

    if num_batch == 0:
        return float('nan')

    with torch.no_grad():
        for batch_idx, (input_batch, labels) in enumerate(data_loader):
            if batch_idx >= num_batch:
                break

            input_batch = input_batch.to(device)
            labels = labels.to(device)

            logits = model(input_batch)       # [B, seq_len, num_classes]
            cls_logits = logits[:, -1, :]     # [B, num_classes]
            preds = cls_logits.argmax(dim=-1) # [B]

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return correct / total if total > 0 else 0.0
