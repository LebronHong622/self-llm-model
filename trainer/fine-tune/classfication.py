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
) -> dict:
    """
    执行分类微调训练循环（增强版）。

    变更点:
        1. num_epochs 从 config.training.num_epochs 获取
        2. 每个 epoch 结束后分别计算 train/val 的 loss 和 accuracy
        3. 返回 history 字典供绘图和后续分析

    Args:
        model:        已准备好的分类模型（输出头为 2 类，参数已冻结）
        train_loader: 训练集 DataLoader
        val_loader:   验证集 DataLoader
        config:       配置对象（读取 training.num_epochs / training.learning_rate）
        device:       从 config.training.device 获取的计算设备

    Returns:
        history: 包含 train_losses, val_losses, train_accs, val_accs 的字典
    """
    from trainer.train_utils import (
        calc_classification_loss_loader,
        calc_classification_accuracy_loader,
    )

    num_epochs = config.training.num_epochs
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )

    history = {
        "train_losses": [],
        "val_losses": [],
        "train_accs": [],
        "val_accs": [],
    }

    print(f"=== 开始训练 ({num_epochs} epochs) ===")
    print(f"  优化器参数量: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    for epoch in range(1, num_epochs + 1):
        # ---- Train Phase ----
        model.train()
        for input_ids, labels in train_loader:
            input_ids = input_ids.to(device)
            labels = labels.to(device)

            logits = model(input_ids)
            cls_logits = logits[:, -1, :]
            loss = criterion(cls_logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # ---- Epoch 级别指标计算 ----
        train_loss = calc_classification_loss_loader(train_loader, model, device)
        train_acc = calc_classification_accuracy_loader(train_loader, model, device)
        val_loss = calc_classification_loss_loader(val_loader, model, device)
        val_acc = calc_classification_accuracy_loader(val_loader, model, device)

        history["train_losses"].append(train_loss)
        history["val_losses"].append(val_loss)
        history["train_accs"].append(train_acc)
        history["val_accs"].append(val_acc)

        print(
            f"Epoch {epoch}/{num_epochs} | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}"
        )

    print("训练完成\n")
    return history


def save_training_curves(
    history: dict,
    save_dir: str = "output",
    prefix: str = "classification",
):
    """
    绘制训练过程曲线并保存为 PNG。

    Args:
        history:   训练历史字典，包含:
                     - train_losses: List[float]
                     - val_losses:   List[float]
                     - train_accs:   List[float]
                     - val_accs:     List[float]
        save_dir:  图片保存目录
        prefix:    文件名前缀
    """
    import os
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(save_dir, exist_ok=True)
    epochs_range = range(1, len(history["train_losses"]) + 1)

    # ---- 图 1: 损失曲线 ----
    plt.figure(figsize=(10, 5))
    plt.plot(epochs_range, history["train_losses"], marker="o", label="Train Loss")
    plt.plot(epochs_range, history["val_losses"], marker="s", label="Val Loss")
    plt.title("Training & Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    loss_path = os.path.join(save_dir, f"{prefix}_loss_curve.png")
    plt.savefig(loss_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[save_training_curves] 损失曲线已保存: {loss_path}")

    # ---- 图 2: 准确率曲线 ----
    plt.figure(figsize=(10, 5))
    plt.plot(epochs_range, history["train_accs"], marker="o", label="Train Acc")
    plt.plot(epochs_range, history["val_accs"], marker="s", label="Val Acc")
    plt.title("Training & Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True, alpha=0.3)
    acc_path = os.path.join(save_dir, f"{prefix}_acc_curve.png")
    plt.savefig(acc_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[save_training_curves] 准确率曲线已保存: {acc_path}")


def save_classification_model(
    model: SelfGPTModel,
    config,
    device: torch.device,
    num_classes: int = 2,
    save_dir: str = "model/fine-tune/classfication",
) -> str:
    """
    将训练完成的分类模型保存到指定目录。

    保存内容 (单个 .pt 文件):
        - model_state_dict: 模型全部参数（含替换后的分类头）
        - config_dict:      完整配置（config.model_dump()）
        - num_classes:      类别数
        - label_map:        标签映射 {0: "ham", 1: "spam"}

    Args:
        model:       训练完成后的分类模型
        config:      配置对象
        device:      当前设备（用于日志）
        num_classes: 分类类别数
        save_dir:    保存目录

    Returns:
        保存文件的绝对路径
    """
    import os

    os.makedirs(save_dir, exist_ok=True)

    label_map = {0: "ham (not spam)", 1: "spam"}

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "config_dict": config.model_dump(),
        "num_classes": num_classes,
        "label_map": label_map,
    }

    save_path = os.path.join(save_dir, "classification_model.pth")
    torch.save(checkpoint, save_path)

    print(f"\n[save_classification_model] 模型已保存: {save_path}")
    print(f"  设备: {device}")
    print(f"  类别数: {num_classes}, 标签映射: {label_map}")

    return os.path.abspath(save_path)


def train_classification(
    environment: Environment = Environment.TEST,
    max_length: int | None = None,
    num_classes: int = 2,
) -> tuple:
    """
    分类微调训练入口（增强版）。

    新增功能:
        1. 训练轮数从 config.training.num_epochs 读取
        2. 每个 epoch 记录 train/val loss + accuracy
        3. 自动绘制并保存 loss/accuracy 曲线图
        4. 最终输出 train / val / test 三集准确率

    Args:
        environment:  环境配置（TEST / DEV / PROD）
        max_length:   序列长度，None 时自适应
        num_classes:  分类类别数，默认 2 (ham / spam)

    Returns:
        (history, model, device, config)
    """
    from trainer.train_utils import calc_classification_accuracy_loader

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

    # ---- Step 3: 模型加载与调整（device 从 config 获取）----
    device = torch.device(config.training.device)
    model = setup_classification_model(
        config=config,
        device=device,
        num_classes=num_classes,
    )

    # ---- Step 4: 训练（返回 history，num_epochs 从 config 获取）----
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device,
    )

    # ---- Step 5: 绘制曲线 ----
    save_training_curves(history, save_dir="output", prefix="classification")

    # ---- Step 6: 最终评估 ----
    final_train_acc = calc_classification_accuracy_loader(train_loader, model, device)
    final_val_acc = calc_classification_accuracy_loader(val_loader, model, device)
    final_test_acc = calc_classification_accuracy_loader(test_loader, model, device)

    # ---- Step 7: 打印最终结果 ----
    print("\n" + "=" * 55)
    print("分类微调训练完成 — 最终结果")
    print("=" * 55)
    print(f"  训练集准确率:   {final_train_acc:.4f} ({final_train_acc*100:.2f}%)")
    print(f"  验证集准确率:   {final_val_acc:.4f} ({final_val_acc*100:.2f}%)")
    print(f"  测试集准确率:   {final_test_acc:.4f} ({final_test_acc*100:.2f}%)")
    print("=" * 55)

    # ---- Step 8: 保存训练完成的模型 ----
    save_classification_model(
        model=model,
        config=config,
        device=device,
        num_classes=num_classes,
        save_dir="model/fine-tune/classfication",
    )

    return history, model, device, config


if __name__ == "__main__":
    history, model, device, config = train_classification(
        num_classes=2,
    )
