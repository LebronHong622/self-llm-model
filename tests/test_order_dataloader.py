import sys
from pathlib import Path
import torch
import tiktoken

fine_tune_dir = Path(__file__).parent.parent / "trainer" / "fine-tune"
sys.path.insert(0, str(fine_tune_dir))
from order import create_instruction_dataloader


def test_dataloader():
    """
    测试: 1) 数据加载 2) partial 绑定 3) DataLoader 分批次 4) 逻辑正确性
    """
    PAD = 50256
    IGNORE = -100
    device = torch.device("cpu")

    # ---- 1. 加载 tokenizer ----
    tokenizer = tiktoken.get_encoding("gpt2")

    # ---- 2. 测试参数 ----
    batch_size = 4
    max_length = 512

    # ---- 3. 使用 partial 创建 DataLoader ----
    loader = create_instruction_dataloader(
        tokenizer=tokenizer,
        batch_size=batch_size,
        max_length=max_length,
        device=device,
    )

    # ---- 4. 测试：取一个 batch 验证 ----
    input_ids, labels = next(iter(loader))

    print(f"\n--- Batch 输出信息 ---")
    print(f"input_ids shape: {input_ids.shape}, device: {input_ids.device}")
    print(f"labels shape:    {labels.shape}, device: {labels.device}")

    # 断言 1: shape 正确
    assert input_ids.shape == labels.shape, "input_ids 和 labels shape 不一致"
    assert input_ids.shape[0] == batch_size, "batch_size 不匹配"

    # 断言 2: device 正确
    assert input_ids.device == device, "input_ids device 不匹配"
    assert labels.device == device, "labels device 不匹配"

    # 断言 3: labels 是 input_ids 右移 1 位（ignore_index 位置除外）
    for i in range(input_ids.shape[0]):
        L = input_ids.shape[1]
        for j in range(L - 1):
            if labels[i, j].item() != IGNORE:
                assert labels[i, j].item() == input_ids[i, j + 1].item(), (
                    f"sample[{i}] pos {j}: labels[{j}]={labels[i,j].item()} "
                    f"!= inputs[{j+1}]={input_ids[i,j+1].item()}"
                )

    # 断言 4: labels 中 padding 区域 ignore_index 处理正确
    #   规则：首个 pad_token_id 保留，后续全部替换为 -100
    for i in range(labels.shape[0]):
        found_first_pad = False
        for j in range(labels.shape[1]):
            val = labels[i, j].item()
            if val == PAD:
                if found_first_pad:
                    raise AssertionError(
                        f"sample[{i}] pos {j}: 非首个 pad 应为 -100"
                    )
                found_first_pad = True
            elif val == IGNORE:
                if not found_first_pad:
                    raise AssertionError(
                        f"sample[{i}] pos {j}: 首个 pad 前不应出现 -100"
                    )

    print("[OK] labels 是 input_ids 右移 1 位")
    print("[OK] labels 中 padding 区域 ignore_index 处理正确")
    print("[OK] DataLoader 测试全部通过")


if __name__ == "__main__":
    test_dataloader()
