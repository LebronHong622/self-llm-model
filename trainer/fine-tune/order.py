import torch
import tiktoken
from functools import partial
from torch.utils.data import DataLoader
from config.config import get_config, Environment
from dataset.fine_tune_order import (
    OrderInstructionDataset,
    instruction_collate_fn,
    DEFAULT_DATA_PATH,
)


def create_instruction_dataloader(
    tokenizer,
    batch_size: int,
    max_length: int | None,
    device: torch.device,
    data_path: str = DEFAULT_DATA_PATH,
    num_workers: int = 0,
) -> DataLoader:
    """
    创建指令微调 DataLoader。

    步骤:
        1. 加载 OrderInstructionDataset
        2. 使用 functools.partial 预填充 device 参数到 collate_fn
        3. 创建 DataLoader
    """
    # Step 1: 加载 Dataset
    dataset = OrderInstructionDataset(data_path=data_path, tokenizer=tokenizer)

    # Step 2: 使用 partial 绑定 device、max_length 等参数
    collate_fn = partial(
        instruction_collate_fn,
        device=device,
        max_length=max_length,
        pad_token_id=50256,
        ignore_index=-100,
    )

    # Step 3: 创建 DataLoader
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
        drop_last=True,
    )

    print("=" * 50)
    print("指令微调 DataLoader 创建完成")
    print(f"  数据集: {data_path}")
    print(f"  样本数: {len(dataset)}")
    print(f"  batch_size: {batch_size}")
    print(f"  max_length: {max_length}")
    print(f"  device: {device}")
    print("=" * 50)

    return loader


