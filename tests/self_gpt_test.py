import tiktoken
import torch
from model.self_gpt import SelfGPTModel
from config.config import ModelConfig
from model.base_layer import InputEmbeddingLayer, TransformerBlock, NormLayer


tokenizer = tiktoken.get_encoding("gpt2")
batch = []
txt1 = "Every effort moves you"
txt2 = "Every day holds a"

batch.append(torch.tensor(tokenizer.encode(txt1)))
batch.append(torch.tensor(tokenizer.encode(txt2)))
batch = torch.stack(batch, dim=0)

torch.manual_seed(123)
config = ModelConfig(vocab_size=50257, hidden_size=768, context_length=1024)
model = SelfGPTModel(config)
logits = model(batch)
print(logits.shape)

# 获取整个模型的参数数量（嵌入层和输出层未共享权重）
total_params = sum(p.numel() for p in model.parameters())
print(f"Total number of parameters: {total_params:,}")

# 获取嵌入层和输出层的参数数量
print("Token embedding layer shape:", model.embedding.embedding.weight.shape)
print("Output layer shape:", model.output_layer.weight.shape)

# 获取前馈层和注意力层的参数数量
transformer_0 = model.transformers[0]
ffn_total_params = sum(p.numel() for p in transformer_0.ffn.parameters())
print(f"FFN total number of parameters: {ffn_total_params:,}")

attn_total_params = sum(p.numel() for p in transformer_0.mha.parameters())
print(f"Attention total number of parameters: {attn_total_params:,}")
