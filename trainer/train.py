# -*- coding: utf-8 -*-
"""
训练脚本

功能：
    - 加载 the-verdict.txt 数据集
    - 分为训练集和测试集
    - 计算首次运行后的训练集损失和测试集损失
"""

import torch
from config.config import get_config, Environment
from model.self_gpt import SelfGPTModel
from trainer.train_utils import create_dataloader, calc_loss_loader


def train():
    """主训练函数"""
    # 1. 加载 test 环境配置
    config = get_config(Environment.TEST)

    # 2. 读取 the-verdict.txt 数据集
    dataset_path = "dataset/the-verdict.txt"
    with open(dataset_path, "r", encoding="utf-8") as f:
        raw_text = f.read()

    # 3. 切分数据集：90% 训练集，10% 测试集
    split_idx = int(0.9 * len(raw_text))
    train_text = raw_text[:split_idx]
    val_text = raw_text[split_idx:]

    # 4. 获取序列长度参数
    max_length = min(config.model.context_length, config.data.max_seq_length)
    batch_size = config.training.batch_size

    # 5. 创建训练集和测试集 DataLoader
    train_loader = create_dataloader(
        train_text,
        batch_size=batch_size,
        max_length=max_length,
        stride=max_length,
        shuffle=True,
        drop_last=True,
        num_workers=0
    )
    val_loader = create_dataloader(
        val_text,
        batch_size=batch_size,
        max_length=max_length,
        stride=max_length,
        shuffle=False,
        drop_last=False,
        num_workers=0
    )

    # 6. 初始化模型
    device = torch.device(config.training.device)
    model = SelfGPTModel(config.model)
    model.to(device)

    # 6.5 加载 GPT-2 预训练权重
    if config.training.load_pretrained:
        from model.gpt_download import download_and_load_gpt2
        from model.load_gpt2_weights import load_gpt2_weights

        settings, params = download_and_load_gpt2("124M", "model/gpt2")
        load_gpt2_weights(model, params, config.model)
    else:
        print("[train] 使用随机初始化权重")

    # 7. 计算训练集损失
    train_loss = calc_loss_loader(train_loader, model, device)

    # 8. 计算测试集损失
    val_loss = calc_loss_loader(val_loader, model, device)

    # 9. 输出结果
    print("=" * 60)
    print(f"数据集: {dataset_path}")
    print(f"训练集样本数: {len(train_loader.dataset)}")
    print(f"测试集样本数: {len(val_loader.dataset)}")
    print("-" * 60)
    print(f"训练集损失 (首次运行): {train_loss:.4f}")
    print(f"测试集损失 (首次运行): {val_loss:.4f}")
    print("=" * 60)

    return train_loss, val_loss


if __name__ == "__main__":
    train()
