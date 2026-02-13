# llm训练数据加载模块
from datasets import load_dataset
from torch.utils.data import Dataset
import torch

class PretrainedDataset(Dataset):
    '''
    预训练数据集类
    '''
    def __init__(self, dataset_filepath, tokenizer, max_length=512):
        self.dataset = load_dataset("json", data_files=dataset_filepath, split="train")
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        item = self.dataset[index]
        text = str(item["text"])
        input_ids = self.tokenizer(text, add_special_tokens=False, max_length=self.max_length - 2, truncation=True).input_ids
        # 添加上起始和结束的token
        input_ids = [self.tokenizer.bos_token_id] + input_ids + [self.tokenizer.eos_token_id]
        # 填充到max_length
        input_ids = input_ids + [self.tokenizer.pad_token_id] * (self.max_length - len(input_ids))
        # 转成tensor形式
        input_ids = torch.tensor(input_ids)

        # 构造labels
        labels = input_ids.clone()
        # 把padding token的label设置为-100，这样在计算loss时会被忽略
        labels[labels == self.tokenizer.pad_token_id] = -100

        return input_ids, labels

