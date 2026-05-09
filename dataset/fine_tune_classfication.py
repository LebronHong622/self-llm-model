import pandas as pd
import torch
from pathlib import Path
from torch.utils.data import Dataset

DEFAULT_DATA_PATH = "dataset/sms_spam_collection/SMSSpamCollection.tsv"
DEFAULT_SEP = "\t"
DEFAULT_NAMES = ["Label", "Message"]
OUTPUT_DIR = "dataset/fine-tune/classfication"
TRAIN_PATH = f"{OUTPUT_DIR}/train.csv"
VALID_PATH = f"{OUTPUT_DIR}/validation.csv"
TEST_PATH = f"{OUTPUT_DIR}/test.csv"


class SpamClassificationDataset(Dataset):
    """
    短信分类数据集 — CSV 优先加载 / TSV 回退 + tokenizer 编码 + 截断填充

    Args:
        csv_path:       切分后的 CSV 路径（如 train.csv）
        tsv_path:       原始 TSV 源文件路径（CSV 不存在时使用）
        tokenizer:      tiktoken 编码器实例
        max_length:     最大序列长度，None 时自动取数据集中最长文本
        pad_token_id:   填充 token id，默认 50256 (<|endoftext|>)
        sep:            TSV 分隔符
        names:          TSV 列名
        random_state:   随机种子
    """

    def __init__(
        self,
        csv_path: str,
        tsv_path: str = DEFAULT_DATA_PATH,
        tokenizer=None,
        max_length: int | None = None,
        pad_token_id: int = 50256,
        sep: str = DEFAULT_SEP,
        names: list[str] | None = None,
        random_state: int = 123,
    ):
        self.tokenizer = tokenizer
        self.pad_token_id = pad_token_id
        self.names = names or DEFAULT_NAMES
        self.random_state = random_state

        # 加载数据（CSV 优先，否则从 TSV 生成）
        self.df, self._split_name = self._load_data(csv_path, tsv_path, sep)

        # 预编码所有文本 + 统计 max_len
        self._all_encoded = [self.tokenizer.encode(text) for text in self.df["Message"]]
        if max_length is not None:
            self.max_len = max_length
        else:
            self.max_len = max(len(e) for e in self._all_encoded) if self._all_encoded else 0

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        input_ids = self._encode_and_pad(self._all_encoded[idx])
        label = torch.tensor(
            1 if self.df.iloc[idx]["Label"] == "spam" else 0,
            dtype=torch.long,
        )
        return input_ids, label

    # ==================== 私有方法 ====================

    def _load_data(self, csv_path: str, tsv_path: str, sep: str) -> tuple[pd.DataFrame, str]:
        """csv 存在直接读 csv；否则读 tsv 并平衡切分后缓存 CSV。"""
        if Path(csv_path).exists():
            print(f"[{self.__class__.__name__}] 发现已保存的数据集 {csv_path}，直接加载...")
            return pd.read_csv(csv_path), Path(csv_path).stem
        else:
            print(f"[{self.__class__.__name__}] 未发现 {csv_path}，从 TSV 生成...")
            df = self._balance_and_split_from_tsv(tsv_path, sep)
            return df, Path(csv_path).stem

    def _balance_and_split_from_tsv(self, tsv_path: str, sep: str) -> pd.DataFrame:
        """从 TSV 源文件读取 → 平衡 → 切分 → 保存 CSV → 返回对应 split 的 DataFrame"""
        balanced_df, original_df = self._load_and_balance_data(tsv_path, sep)

        print("原始数据集:")
        print(original_df["Label"].value_counts())
        print(f"总计: {len(original_df)} 条\n")

        print("平衡后数据集:")
        print(balanced_df["Label"].value_counts())
        print(f"总计: {len(balanced_df)} 条\n")

        train_df, val_df, test_df = self._random_split(balanced_df)
        self._save_splits(train_df, val_df, test_df)
        print(f"已保存到 {OUTPUT_DIR}/\n")

        # 根据 csv_path 对应的 split 返回
        split_map = {
            "train": train_df,
            "validation": val_df,
            "test": test_df,
        }
        return split_map.get(self._split_name, train_df)

    def _load_and_balance_data(
        self, data_path: str, sep: str,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """读取 TSV 数据并平衡（ham 下采样至与 spam 数量一致）。"""
        df = pd.read_csv(data_path, sep=sep, header=None, names=self.names)

        num_spam = (df["Label"] == "spam").sum()
        ham_subset = df[df["Label"] == "ham"].sample(n=num_spam, random_state=self.random_state)

        balanced_df = pd.concat(
            [ham_subset, df[df["Label"] == "spam"]],
            ignore_index=True,
        )

        return balanced_df, df

    @staticmethod
    def _random_split(
        df: pd.DataFrame,
        train_frac: float = 0.7,
        validation_frac: float = 0.1,
        random_state: int = 123,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """按比例随机切分为 train / validation / test。"""
        df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)

        train_end = int(len(df) * train_frac)
        validation_end = train_end + int(len(df) * validation_frac)

        return df[:train_end], df[train_end:validation_end], df[validation_end:]

    @staticmethod
    def _save_splits(train_df, val_df, test_df, output_dir: str = OUTPUT_DIR):
        """保存三份切分结果为 CSV。"""
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        train_df.to_csv(TRAIN_PATH, index=None)
        val_df.to_csv(VALID_PATH, index=None)
        test_df.to_csv(TEST_PATH, index=None)

    def _encode_and_pad(self, encoded_tokens: list[int]) -> torch.Tensor:
        """按 max_len 截断或 pad，返回 tensor。"""
        if len(encoded_tokens) > self.max_len:
            encoded_tokens = encoded_tokens[:self.max_len]
        else:
            encoded_tokens = encoded_tokens + [self.pad_token_id] * (self.max_len - len(encoded_tokens))

        return torch.tensor(encoded_tokens, dtype=torch.long)


# ==================== 测试代码 ====================
if __name__ == "__main__":
    import tiktoken
    from torch.utils.data import DataLoader

    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = SpamClassificationDataset(
        csv_path=TRAIN_PATH,
        tsv_path=DEFAULT_DATA_PATH,
        tokenizer=tokenizer
    )

    print(f"\nDataset size: {len(dataset)}")
    print(f"Max length: {dataset.max_len}")
    print(f"Split name: {dataset._split_name}")

    # 取单条样本验证
    input_ids, label = dataset[0]
    print(f"\nSample[0] input_ids shape: {input_ids.shape}")
    print(f"Sample[0] input_ids (前10): {input_ids[:10].tolist()}")
    print(f"Sample[0] label: {label.item()} ({'spam' if label.item() == 1 else 'ham'})")

    # DataLoader 批量测试
    loader = DataLoader(dataset, batch_size=4, shuffle=True)
    for batch_inputs, batch_labels in loader:
        print(f"\nBatch input_ids shape: {batch_inputs.shape}")
        print(f"Batch labels shape: {batch_labels.shape}")
        break

    # 标签分布验证
    labels = [dataset[i][1].item() for i in range(len(dataset))]
    from collections import Counter
    dist = Counter(labels)
    print(f"\n标签分布: ham={dist.get(0, 0)}, spam={dist.get(1, 0)}")

    # 二次运行：CSV 已存在应直接加载
    print("\n--- 二次运行测试 ---")
    dataset2 = SpamClassificationDataset(
        csv_path=TRAIN_PATH,
        tokenizer=tokenizer,
        max_length=64,
    )
    assert len(dataset) == len(dataset2), "加载数据不一致！"
    print("✓ 直接加载与重新生成结果一致")
