# import requests
# import zipfile
# import os
# from pathlib import Path

# url = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
# zip_path = "sms_spam_collection.zip"
# extracted_path = "sms_spam_collection"
# data_file_path = Path(extracted_path) / "SMSSpamCollection.tsv"


# # 下载微调分类的数据集
# def download_and_unzip_spam_data(url, zip_path, extracted_path, data_file_path):
#     if data_file_path.exists():
#         print(f"{data_file_path} already exists. Skipping download and extraction.")
#         return

#     # Downloading the file
#     response = requests.get(url, stream=True, timeout=60)
#     response.raise_for_status()
#     with open(zip_path, "wb") as out_file:
#         for chunk in response.iter_content(chunk_size=8192):
#             if chunk:
#                 out_file.write(chunk)

#     # Unzipping the file
#     with zipfile.ZipFile(zip_path, "r") as zip_ref:
#         zip_ref.extractall(extracted_path)

#     # Add .tsv file extension
#     original_file_path = Path(extracted_path) / "SMSSpamCollection"
#     os.rename(original_file_path, data_file_path)
#     print(f"File downloaded and saved as {data_file_path}")


# try:
#     download_and_unzip_spam_data(url, zip_path, extracted_path, data_file_path)
#     print("success download and unzip data")
# except (requests.exceptions.RequestException, TimeoutError) as e:
#     print(f"Primary URL failed: {e}. Trying backup URL...")
#     url = "https://f001.backblazeb2.com/file/LLMs-from-scratch/sms%2Bspam%2Bcollection.zip"
#     download_and_unzip_spam_data(url, zip_path, extracted_path, data_file_path)

## 下载指令微调数据集
import json
import os
import random

import requests


def download_and_load_file(file_path, url):
    if not os.path.exists(file_path):
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        text_data = response.text
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(text_data)

    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    return data


def split_and_save_dataset(data, output_dir, ratios=(0.85, 0.10, 0.05), seed=42):
    """
    将数据按比例随机切分为训练集、验证集和测试集并保存为 JSON 文件。

    Args:
        data:       完整数据列表
        output_dir: 输出目录路径
        ratios:     (train, validation, test) 比例，默认 (0.85, 0.10, 0.05)
        seed:       随机种子

    Returns:
        (train_data, val_data, test_data) 三元组
    """
    assert abs(sum(ratios) - 1.0) < 1e-9, "ratios 之和必须等于 1.0"

    random.seed(seed)
    shuffled = data.copy()
    random.shuffle(shuffled)

    n = len(shuffled)
    train_end = int(n * ratios[0])
    val_end = train_end + int(n * ratios[1])

    train_data = shuffled[:train_end]
    val_data = shuffled[train_end:val_end]
    test_data = shuffled[val_end:]

    os.makedirs(output_dir, exist_ok=True)

    splits = {
        "train": train_data,
        "validation": val_data,
        "test": test_data,
    }

    for name, subset in splits.items():
        path = os.path.join(output_dir, f"{name}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(subset, f, ensure_ascii=False, indent=2)
        print(f"已保存 {name} 集 ({len(subset)} 条): {path}")

    return train_data, val_data, test_data


# The book originally used the following code below
# However, urllib uses older protocol settings that
# can cause problems for some readers using a VPN.
# The `requests` version above is more robust
# in that regard.

"""
import urllib

def download_and_load_file(file_path, url):

    if not os.path.exists(file_path):
        with urllib.request.urlopen(url) as response:
            text_data = response.read().decode("utf-8")
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(text_data)

    else:
        with open(file_path, "r", encoding="utf-8") as file:
            text_data = file.read()

    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    return data
"""


file_path = "./dataset/fine-tune/order/instruction-data.json"
output_dir = "./dataset/fine-tune/order"
url = (
    "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch"
    "/main/ch07/01_main-chapter-code/instruction-data.json"
)

data = download_and_load_file(file_path, url)
print(f"总数据条数: {len(data)}\n")

train_data, val_data, test_data = split_and_save_dataset(
    data, output_dir, ratios=(0.85, 0.10, 0.05), seed=42
)

print(f"\n训练集: {len(train_data)} 条 ({len(train_data)/len(data)*100:.1f}%)")
print(f"验证集: {len(val_data)} 条   ({len(val_data)/len(data)*100:.1f}%)")
print(f"测试集: {len(test_data)} 条   ({len(test_data)/len(data)*100:.1f}%)")
