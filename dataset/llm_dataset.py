# llm训练数据加载模块
from datasets import load_dataset

if __name__ == "__main__":
    dataset = load_dataset("json", data_files="./pretrain_hq.jsonl")
    print(dataset)