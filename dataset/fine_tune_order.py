import json
import torch
from pathlib import Path
from torch.utils.data import Dataset

DEFAULT_DATA_PATH = "dataset/fine-tune/order/train.json"

ALPACA_TEMPLATE_NO_INPUT = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n"
    "### Response:\n{output}"
)

ALPACA_TEMPLATE_WITH_INPUT = (
    "Below is an instruction that describes a task, paired with an input "
    "that provides further context. Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n"
    "### Input:\n{input}\n\n"
    "### Response:\n{output}"
)

ALPACA_TEMPLATE_NO_OUTPUT = (
    "Below is an instruction that describes a task.\n\n"
    "### Instruction:\n{instruction}"
)


class OrderInstructionDataset(Dataset):
    """
    指令微调数据集 — JSON 加载 + Alpaca 提示词格式化 + tokenizer 编码（原始数据，无填充/右移）

    Args:
        data_path:     JSON 数据文件路径（每条含 instruction / input / output 字段）
        tokenizer:     tiktoken 编码器实例
    """

    def __init__(
        self,
        data_path: str = DEFAULT_DATA_PATH,
        tokenizer=None,
    ):
        self.tokenizer = tokenizer

        # 加载并格式化数据
        raw_data = self._load_json(data_path)
        self.formatted_texts = [self._format_alpaca(sample) for sample in raw_data]

        # 预编码所有文本（保持原始长度，不填充不截断）
        self._all_encoded = [
            self.tokenizer.encode(text) for text in self.formatted_texts
        ]

    def __len__(self) -> int:
        return len(self._all_encoded)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return torch.tensor(self._all_encoded[idx], dtype=torch.long)

    # ==================== 私有方法 ====================

    @staticmethod
    def _load_json(path: str) -> list[dict]:
        """从 JSON 文件加载数据列表。"""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        print(f"[OrderInstructionDataset] 从 {path} 加载了 {len(data)} 条样本")
        return data

    @classmethod
    def _format_alpaca(cls, sample: dict) -> str:
        """将单条数据转换为 Alpaca 提示词风格。"""
        instruction = sample.get("instruction", "").strip()
        input_text = sample.get("input", "").strip()
        output = sample.get("output", "").strip()

        if not output:
            # 无 output：只保留 instruction 部分
            if input_text:
                text = ALPACA_TEMPLATE_WITH_INPUT.format(
                    instruction=instruction, input=input_text, output=""
                )
                # 去掉 "### Response:\n" 后缀
                return text.rsplit("### Response:\n", 1)[0].rstrip()
            else:
                return ALPACA_TEMPLATE_NO_OUTPUT.format(instruction=instruction)
        elif input_text:
            return ALPACA_TEMPLATE_WITH_INPUT.format(**sample)
        else:
            return ALPACA_TEMPLATE_NO_INPUT.format(**sample)



# ==================== Collate Function ====================

def instruction_collate_fn(
    batch: list[torch.Tensor],
    *,
    pad_token_id: int = 50256,
    ignore_index: int = -100,
    max_length: int | None = None,
    device: str | torch.device = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """指令微调数据的批处理函数。

    处理流程（参照《从零构建大模型》批次组织方式）：

    1. 每条序列末尾追加一个 ``pad_token_id`` 作为结束标记
    2. 填充至 ``batch_max_length = max(len(item)+1)``
    3. 从填充后的完整序列中切分：``inputs = padded[:-1]``, ``labels = padded[1:]``
    4. labels 中除首个 ``pad_token_id`` 外，其余替换为 ``ignore_index``
    5. 可选截断至 ``max_length``

    Args:
        batch:         一个 batch 的样本列表，每个元素为 1-D torch.Tensor (变长)
        pad_token_id:  填充 / 结束 token 的 id
        ignore_index:  labels 中需忽略的值（默认 -100）
        max_length:    序列最大允许长度；None 表示不限制
        device:        输出张量所在设备

    Returns:
        (input_ids, labels) 各 shape 为 [B, L] 的 LongTensor
    """
    # 计算批次最大长度（每条序列 +1 个  padding）
    batch_max_length = max(len(item) + 1 for item in batch)

    inputs_lst, labels_lst = [], []

    for item in batch:
        new_item = item.tolist()
        # 末尾追加  作为结束标记
        new_item += [pad_token_id]
        # 填充至批次最大长度
        padded = new_item + [pad_token_id] * (batch_max_length - len(new_item))
        # 切分：inputs 去最后一个 token，labels 右移一位
        inputs = torch.tensor(padded[:-1], dtype=torch.long)
        labels = torch.tensor(padded[1:], dtype=torch.long)

        # labels 中 padding 位置处理：首个 pad 保留，其余 → ignore_index
        mask = labels == pad_token_id
        indices = torch.nonzero(mask).squeeze()
        if indices.numel() > 1:
            labels[indices[1:]] = ignore_index

        # 可选截断
        if max_length is not None:
            inputs = inputs[:max_length]
            labels = labels[:max_length]

        inputs_lst.append(inputs)
        labels_lst.append(labels)

    # 堆叠并转移到目标设备
    input_ids = torch.stack(inputs_lst).to(device)
    labels = torch.stack(labels_lst).to(device)

    return input_ids, labels


# ==================== 测试代码 ====================
if __name__ == "__main__":
    import tiktoken
    from torch.utils.data import DataLoader

    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = OrderInstructionDataset(data_path=DEFAULT_DATA_PATH, tokenizer=tokenizer)

    print(f"\nDataset size: {len(dataset)}")

    # 打印首条 Alpaca 格式文本预览
    print(f"\nFormatted text preview (first sample):\n{dataset.formatted_texts[0][:200]}...")

    # 取单条样本验证（返回单个 tensor）
    sample = dataset[0]
    print(f"\nSample[0] shape: {sample.shape}")
    print(f"Sample[0] (前10 tokens): {sample[:10].tolist()}")

    # ========== collate_fn 验证 ==========
    PAD = 50256
    loader = DataLoader(
        dataset, batch_size=4, shuffle=True,
        collate_fn=lambda b: instruction_collate_fn(b, max_length=512),
    )
    input_ids, labels = next(iter(loader))

    print(f"\n--- collate_fn 输出 ---")
    print(f"input_ids shape: {input_ids.shape}")   # [B, L]
    print(f"labels shape:   {labels.shape}")       # [B, L]

    # 验证 1: 每行内部 labels 确实是 inputs 右移 1 位（在 ignore_index 替换前一致）
    IGNORE = -100
    for i in range(input_ids.shape[0]):
        L = input_ids.shape[1]
        for j in range(L - 1):
            if labels[i, j].item() != IGNORE:
                assert labels[i, j].item() == input_ids[i, j + 1].item(), (
                    f"sample[{i}] pos {j}: labels[{j}]={labels[i,j].item()} "
                    f"!= inputs[{j+1}]={input_ids[i,j+1].item()}"
                )
    print("[OK] labels 是 input_ids 右移 1 位")
    print("[OK] labels 是 input_ids 右移 1 位")

    # 验证 2: labels 中 padding 区域 ignore_index 处理正确
    IGNORE = -100
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
    print("[OK] labels 中 padding 区域 ignore_index 处理正确")
