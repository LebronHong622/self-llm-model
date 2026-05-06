import pandas as pd
from pathlib import Path

DEFAULT_DATA_PATH = "dataset/sms_spam_collection/SMSSpamCollection.tsv"
DEFAULT_SEP = "\t"
DEFAULT_NAMES = ["Label", "Message"]
OUTPUT_DIR = "dataset/fine-tune/classfication"
TRAIN_PATH = f"{OUTPUT_DIR}/train.csv"
VALID_PATH = f"{OUTPUT_DIR}/validation.csv"
TEST_PATH = f"{OUTPUT_DIR}/test.csv"


def load_and_balance_data(
    data_path: str = DEFAULT_DATA_PATH,
    sep: str = DEFAULT_SEP,
    names: list = DEFAULT_NAMES,
    random_state: int = 123
):
    """读取 TSV 数据并平衡数据集（ham 下采样至与 spam 数量一致）。"""
    df = pd.read_csv(data_path, sep=sep, header=None, names=names)

    num_spam = (df["Label"] == "spam").sum()
    ham_subset = df[df["Label"] == "ham"].sample(n=num_spam, random_state=random_state)

    balanced_df = pd.concat(
        [ham_subset, df[df["Label"] == "spam"]],
        ignore_index=True
    )

    return balanced_df, df


def random_split(df, train_frac=0.7, validation_frac=0.1, random_state=123):
    """按比例随机切分数据集。"""
    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    train_end = int(len(df) * train_frac)
    validation_end = train_end + int(len(df) * validation_frac)

    return df[:train_end], df[train_end:validation_end], df[validation_end:]


def save_splits(train_df, validation_df, test_df, output_dir=OUTPUT_DIR):
    """保存切分后的数据集到 CSV。"""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    train_df.to_csv(TRAIN_PATH, index=None)
    validation_df.to_csv(VALID_PATH, index=None)
    test_df.to_csv(TEST_PATH, index=None)


def load_splits(output_dir=OUTPUT_DIR):
    """从 CSV 加载已保存的数据集切分。"""
    train_df = pd.read_csv(TRAIN_PATH)
    validation_df = pd.read_csv(VALID_PATH)
    test_df = pd.read_csv(TEST_PATH)
    return train_df, validation_df, test_df


def prepare_data(output_dir=OUTPUT_DIR, random_state=123):
    """
    主入口：如果 CSV 已存在直接加载；否则重新生成并保存。

    Returns:
        train_df, validation_df, test_df, original_df
    """
    if (Path(TRAIN_PATH).exists()
            and Path(VALID_PATH).exists()
            and Path(TEST_PATH).exists()):
        print("发现已保存的数据集，直接加载...")
        train_df, validation_df, test_df = load_splits()
    else:
        print("未发现已保存的数据集，开始生成...")
        balanced_df, original_df = load_and_balance_data(random_state=random_state)

        print("原始数据集:")
        print(original_df["Label"].value_counts())
        print(f"总计: {len(original_df)} 条\n")

        print("平衡后数据集:")
        print(balanced_df["Label"].value_counts())
        print(f"总计: {len(balanced_df)} 条\n")

        train_df, validation_df, test_df = random_split(balanced_df, random_state=random_state)

        save_splits(train_df, validation_df, test_df)
        print("已保存到 dataset/fine-tune/classfication/\n")

    print("切分结果:")
    print(f"  训练集: {len(train_df)} 条")
    print(f"  验证集: {len(validation_df)} 条")
    print(f"  测试集: {len(test_df)} 条")

    return train_df, validation_df, test_df


# ==================== 测试代码 ====================
if __name__ == "__main__":
    train_df, validation_df, test_df = prepare_data()

    print(f"\n训练集标签分布:")
    print(train_df["Label"].value_counts())
    print(f"\n验证集标签分布:")
    print(validation_df["Label"].value_counts())
    print(f"\n测试集标签分布:")
    print(test_df["Label"].value_counts())

    # 二次运行测试"直接加载"分支
    print("\n--- 二次运行测试 ---")
    train_df2, _, _ = prepare_data()
    assert len(train_df) == len(train_df2), "加载数据不一致！"
    print("✓ 直接加载与重新生成结果一致")
