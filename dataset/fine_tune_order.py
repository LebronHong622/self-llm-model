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


# ==================== 测试代码 ====================
if __name__ == "__main__":
    import tiktoken

    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = OrderInstructionDataset(data_path=DEFAULT_DATA_PATH, tokenizer=tokenizer)

    print(f"\nDataset size: {len(dataset)}")

    # 打印首条 Alpaca 格式文本预览
    print(f"\nFormatted text preview (first sample):\n{dataset.formatted_texts[0][:200]}...")

    # 取单条样本验证（返回单个 tensor）
    sample = dataset[0]
    print(f"\nSample[0] shape: {sample.shape}")
    print(f"Sample[0] (前10 tokens): {sample[:10].tolist()}")
