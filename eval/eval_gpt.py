"""
Evaluate GPT-2 model on a given dataset.
"""
import torch
import tiktoken
from model.self_gpt import SelfGPTModel
from config.config import ModelConfig

def generate_text_simple(model, idx, max_new_tokens, context_length):
    """
    Generate text from a given input.
    """
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_length:]
        logits = model(idx_cond)
        logits = logits[:, -1, :]
        next_token = torch.argmax(logits, dim=-1)
        idx = torch.cat((idx, next_token.unsqueeze(0)), dim=1)
    return idx

def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={'<endoftext>'})
    # 增加batch维度
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    return encoded_tensor

def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)
    return tokenizer.decode(flat.tolist())

tokenizer = tiktoken.get_encoding("gpt2")
text = "Every effort moves you"

config = ModelConfig(vocab_size=50257, hidden_size=768, context_length=1024)
model = SelfGPTModel(config)

token_ids = generate_text_simple(
    model, 
    text_to_token_ids(text, tokenizer), 
    10, 
    config.context_length
)

print("Output text\n:", token_ids_to_text(token_ids, tokenizer))
