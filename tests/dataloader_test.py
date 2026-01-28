import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from trainer.train_utils import create_dataloader
from dataset.llm_dataset import PretrainedDataset
from transformers import AutoTokenizer

def test_pretrained_dataloader():
    tokenizer = AutoTokenizer.from_pretrained("./tokenizer")
    dataset_path = "./dataset/pretrain_hq.jsonl"
    pretrained_dataset = PretrainedDataset(dataset_path, tokenizer)
    dataloader = create_dataloader(pretrained_dataset, 2)
    dataloader_iter = iter(dataloader)
    first_batch = next(dataloader_iter)
    print(first_batch)

if __name__ == "__main__":
    test_pretrained_dataloader()

