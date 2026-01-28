# -*- coding: utf-8 -*-
# @Time    : 2026/01/27
# @Author  : <NAME>
# @File    : train_utils.py
# @Description: 训练工具函数
from torch.utils.data import DataLoader, Dataset

# dataloader加载数据公共函数
def create_dataloader(dataset: Dataset, batch_size: int, num_workers: int = 0, shuffle: bool = True, drop_last: bool = True):
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle, drop_last=drop_last)
    return dataloader
