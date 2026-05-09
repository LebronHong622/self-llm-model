# -*- coding: utf-8 -*-
"""
分类微调训练脚本 — 使用 SpamClassificationDataset + DataLoader 加载训练/验证/测试数据

功能模块:
    1. create_classification_dataloaders()   — 数据集加载（已有）
    2. setup_classification_model()          — 模型加载 + 输出头替换(2类) + 参数冻结
    3. train_model()                         — 训练循环
    4. train_classification()                — 入口编排函数
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from config.config import get_config, Environment
from model.self_gpt import SelfGPTModel
from dataset.fine_tune_classfication import (
    SpamClassificationDataset,
    TRAIN_PATH,
    VALID_PATH,
    TEST_PATH,
)


def create_classification_dataloaders(
    tokenizer,
    batch_size: int = 32,
    max_length: int | None = None,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    创建分类任务的 train / val / test DataLoader。

    Args:
        tokenizer:      tiktoken 编码器实例
        batch_size:     批次大小
        max_length:     最大序列长度，None 时自适应取最长文本
        num_workers:    DataLoader 工作进程数

    Returns:
        (train_loader, val_loader, test_loader)
    """
    # 构建三个 Dataset（CSV 已存在则直接加载）
    train_dataset = SpamClassificationDataset(
        csv_path=TRAIN_PATH,
        tokenizer=tokenizer,
        max_length=max_length,
    )
    val_dataset = SpamClassificationDataset(
        csv_path=VALID_PATH,
        tokenizer=tokenizer,
        max_length=train_dataset.max_len,  # 与训练集保持一致
    )
    test_dataset = SpamClassificationDataset(
        csv_path=TEST_PATH,
        tokenizer=tokenizer,
        max_length=train_dataset.max_len,  # 与训练集保持一致
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    print("=" * 50)
    print("分类数据集 DataLoader 创建完成")
    print(f"  训练集: {len(train_dataset)} 条, {len(train_loader)} batches")
    print(f"  验证集: {len(val_dataset)} 条, {len(val_loader)} batches")
    print(f"  测试集: {len(test_dataset)} 条, {len(test_loader)} batches")
    print(f"  max_length (序列长度): {train_dataset.max_len}")
    print("=" * 50)

    return train_loader, val_loader, test_loader


def setup_classification_model(
    config,
    device: torch.device,
    num_classes: int = 2,
) -> SelfGPTModel:
    """
    加载/初始化模型 -> 替换分类头(二分类) -> 冻结非目标参数 -> 返回模型。

    可训练参数仅保留:
        - norm_layer (最终 LayerNorm)
        - transformers[-1] (最后一个 transformer 块)
        - output_layer (新分类头)

    Args:
        config:       配置对象 (config.model / config.training 均需可访问)
        device:       计算设备
        num_classes:  分类类别数，默认 2 (ham / spam)

    Returns:
        已调整好并移至 device 的模型
    """
    # ---- 1. 初始化模型 ----
    model = SelfGPTModel(config.model)

    # ---- 2. 可选加载 GPT-2 预训练权重 ----
    if config.training.load_pretrained:
        from model.gpt_download import download_and_load_gpt2
        from model.load_gpt2_weights import load_gpt2_weights

        settings, params = download_and_load_gpt2("124M", "model/gpt2")
        load_gpt2_weights(model, params, config.model)
        print("[setup_classification_model] 已加载 GPT-2 预训练权重")
    else:
        print("[setup_classification_model] 使用随机初始化权重")

    # ---- 3. 替换输出头: vocab_size -> num_classes ----
    old_output = model.output_layer
    model.output_layer = nn.Linear(config.model.hidden_size, num_classes)
    print(
        f"[setup_classification_model] 输出头已替换: "
        f"{old_output} -> Linear({config.model.hidden_size}, {num_classes})"
    )

    # ---- 4. 冻结参数：除 norm_layer / 最后一个 transformer 块 / output_layer 外全部冻结 ----
    total_params = 0
    frozen_params = 0
    trainable_params = 0

    for name, param in model.named_parameters():
        total_params += param.numel()
        param.requires_grad = False
        frozen_params += param.numel()

    # 解冻 norm_layer
    for param in model.norm_layer.parameters():
        param.requires_grad = True
        trainable_params += sum(p.numel() for p in model.norm_layer.parameters())
    # 减去重复计数（上面已计入 frozen）
    frozen_params -= sum(p.numel() for p in model.norm_layer.parameters())

    # 解冻最后一个 transformer 块
    last_transformer = model.transformers[-1]
    for param in last_transformer.parameters():
        param.requires_grad = True
    frozen_params -= sum(p.numel() for p in last_transformer.parameters())
    trainable_params += sum(p.numel() for p in last_transformer.parameters())

    # 解冻新的 output_layer（分类头）
    for param in model.output_layer.parameters():
        param.requires_grad = True
    frozen_params -= sum(p.numel() for p in model.output_layer.parameters())
    trainable_params += sum(p.numel() for p in model.output_layer.parameters())

    # 移至目标设备
    model.to(device)

    # 打印参数统计
    print(f"\n{'=' * 50}")
    print("模型已适配为分类任务")
    print(f"  总参数量:     {total_params:,}")
    print(f"  可训练参数:   {trainable_params:,} ({trainable_params / total_params * 100:.1f}%)")
    print(f"  冻结参数:     {frozen_params:,} ({frozen_params / total_params * 100:.1f}%)")
    print(f"  可训练模块:")
    print(f"    - norm_layer")
    print(f"    - transformers[-1] (第 {config.model.num_layers} 层)")
    print(f"    - output_layer (Linear({config.model.hidden_size}, {num_classes}))")
    print(f"  设备: {device}")
    print(f"{'=' * 50}\n")

    return model


def train_model(
    model: SelfGPTModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config,
    device: torch.device,
    num_epochs: int | None = None,
):
    """
    执行分类微调训练循环。

    每个 epoch 包含:
        - Train 阶段: 前向传播 -> 取最后一个 token logits -> CE Loss -> 反向传播
        - Val 阶段:   前向传播 -> 计算 accuracy

    Args:
        model:        已准备好的分类模型（输出头为 2 类，参数已冻结）
        train_loader: 训练集 DataLoader
        val_loader:   验证集 DataLoader
        config:       配置对象
        device:       计算设备
        num_epochs:   训练轮数，None 时使用 config.training.num_epochs
    """
    epochs = num_epochs or config.training.num_epochs

    # 损失函数：交叉熵（二分类）
    criterion = nn.CrossEntropyLoss()

    # 优化器：仅传入可训练参数
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.training.learning_rate,
    )

    print(f"=== 开始训练 ({epochs} epochs) ===")
    print(f"  优化器参数量: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    for epoch in range(1, epochs + 1):
        # ---- Train Phase ----
        model.train()
        train_loss_sum = 0.0
        train_correct = 0
        train_total = 0

        for batch_idx, (input_ids, labels) in enumerate(train_loader):
            input_ids = input_ids.to(device)
            labels = labels.to(device)

            # 前向传播
            logits = model(input_ids)               # [B, seq_len, 2]
            cls_logits = logits[:, -1, :]           # [B, 2] 取最后一个 token 位置
            loss = criterion(cls_logits, labels)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 统计
            train_loss_sum += loss.item() * input_ids.size(0)
            preds = cls_logits.argmax(dim=-1)
            train_correct += (preds == labels).sum().item()
            train_total += input_ids.size(0)

        train_loss = train_loss_sum / train_total
        train_acc = train_correct / train_total

        # ---- Val Phase ----
        model.eval()
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for input_ids, labels in val_loader:
                input_ids = input_ids.to(device)
                labels = labels.to(device)

                logits = model(input_ids)
                cls_logits = logits[:, -1, :]
                preds = cls_logits.argmax(dim=-1)

                val_correct += (preds == labels).sum().item()
                val_total += input_ids.size(0)

        val_acc = val_correct / val_total

        print(
            f"Epoch {epoch}/{epochs} | "
            f"[Train] Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
            f"[Val] Acc: {val_acc:.4f}"
        )

    print("训练完成")


def train_classification(
    environment: Environment = Environment.TEST,
    max_length: int | None = None,
    num_classes: int = 2,
):
    """
    分类微调训练入口：数据加载 -> 模型加载与调整 -> 整合训练 -> 返回全部组件。

    流程:
        1. create_classification_dataloaders()  — 加载 train / val / test 数据
        2. setup_classification_model()         — 初始化模型 + 替换输出头(2类) + 冻结参数
        3. train_model()                        — 执行训练循环

    Args:
        environment:  环境配置（TEST / DEV / PROD）
        max_length:   序列长度，None 时自适应
        num_classes:  分类类别数，默认 2 (ham / spam)

    Returns:
        (train_loader, val_loader, test_loader, model, device, config)
    """
    # ---- Step 1: 加载配置 ----
    config = get_config(environment)

    # ---- Step 2: 数据集加载 ----
    import tiktoken

    tokenizer = tiktoken.get_encoding("gpt2")
    train_loader, val_loader, test_loader = create_classification_dataloaders(
        tokenizer=tokenizer,
        batch_size=config.training.batch_size,
        max_length=max_length,
        num_workers=config.training.num_workers,
    )

    # 各取一个 batch 验证 shape
    for name, loader in [("train", train_loader), ("val", val_loader), ("test", test_loader)]:
        inputs, labels = next(iter(loader))
        print(f"[{name}] input_ids: {inputs.shape}, labels: {labels.shape}")

    # ---- Step 3: 模型加载与调整 ----
    device = torch.device(config.training.device)
    model = setup_classification_model(
        config=config,
        device=device,
        num_classes=num_classes,
    )

    # ---- Step 4: 整合训练 ----
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device,
    )

    return train_loader, val_loader, test_loader, model, device, config


if __name__ == "__main__":
    # 完整流程测试：数据加载 -> 模型适配 -> 训练 -> 前向验证
    train_loader, val_loader, test_loader, model, device, config = train_classification(
        max_length=64,
        num_classes=2,
    )

    # 前向传播验证：取一个 batch 跑一遍模型
    model.eval()
    with torch.no_grad():
        sample_inputs, sample_labels = next(iter(train_loader))
        sample_inputs = sample_inputs.to(device)
        outputs = model(sample_inputs)
        cls_logits = outputs[:, -1, :]

        print("\n" + "=" * 50)
        print("前向传播测试通过")
        print(f"  输入:          {sample_inputs.shape}")
        print(f"  输出 logits:   {outputs.shape}")
        print(f"  分类 logits:   {cls_logits.shape} (取最后 token)")
        print(f"  标签:          {sample_labels.shape}")
        print("=" * 50)
