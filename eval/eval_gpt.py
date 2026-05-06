"""
Evaluate GPT-2 model on a given dataset.
"""
import torch
import tiktoken
from model.self_gpt import SelfGPTModel
from config.config import get_config, Environment

def generate(model, idx, max_new_tokens, context_length, 
                        temperature=0.0, top_k=None, end_id=None):
    """
    Generate text from a given input.
    """
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_length:]
        model.eval()
        with torch.no_grad():
            logits = model(idx_cond)

        # 修改成由温度和top-k控制的采样
        logits = logits[:, -1, :]
        if top_k is not None:
            top_logits, _ = torch.topk(logits, top_k)
            min_logit = top_logits[:, -1]
            logits = torch.where(logits < min_logit, torch.tensor(float('-inf')).to(logits.device), logits)
        if temperature > 0.0:
            logits = logits / temperature
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
        else: 
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
        if next_token == end_id:
            break
        idx = torch.cat((idx, next_token), dim=1)
    return idx

def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={'<endoftext>'})
    # 增加batch维度
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    return encoded_tensor

def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)
    return tokenizer.decode(flat.tolist())

# 从配置文件加载全部参数
config = get_config(Environment.TEST)

# 初始化模型
model = SelfGPTModel(config.model)

# 根据 config.eval.load_pretrained 决定是否加载 GPT-2 预训练权重
if config.eval.load_pretrained:
    from model.gpt_download import download_and_load_gpt2
    from model.load_gpt2_weights import load_gpt2_weights
    settings, params = download_and_load_gpt2(
        config.eval.gpt2_model_size,
        config.eval.gpt2_model_dir
    )
    load_gpt2_weights(model, params, config.model)
    print(f"[eval] 已加载 GPT-2 {config.eval.gpt2_model_size} 预训练权重")

device = torch.device(config.eval.device)
model.to(device).eval()

tokenizer = tiktoken.get_encoding("gpt2")

input_ids = text_to_token_ids(config.eval.prompt, tokenizer).to(device)

token_ids = generate(
    model,
    input_ids,
    config.eval.max_new_tokens,
    config.model.context_length,
    temperature=config.eval.temperature,
    top_k=config.eval.top_k
)

print("Generated text:\n", token_ids_to_text(token_ids, tokenizer))
