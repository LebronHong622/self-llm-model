from trainer.train_utils import create_dataloader
from dataset.llm_dataset import PretrainedDataset
from transformers import AutoTokenizer

with open("tests/the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

dataloader = create_dataloader(raw_text, batch_size=1, max_length=4, stride=1, shuffle=False)
data_iter = iter(dataloader)
first_batch = next(data_iter)
print(first_batch)

