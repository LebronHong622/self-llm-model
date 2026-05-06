# -*- coding: utf-8 -*-
"""
GPT-2 权重加载模块

将 OpenAI GPT-2 124M 的 checkpoint 参数赋值给 SelfGPTModel。

参数来源：model/gpt_download.py -> download_and_load_gpt2() 返回的 params dict

重要: GPT-2 checkpoint 来自 TensorFlow，其 Dense 层权重形状为 [in, out]，
      而 PyTorch nn.Linear 权重为 [out, in]。因此所有线性层权重需要转置。

参数映射关系：
    GPT-2 (TF checkpoint)              SelfGPTModel
    ──────────────────────────         ─────────────────────────────
    wte [50257, 768]                   embedding.embedding.weight
    wpe [1024, 768]                    embedding.position_embedding.weight
    h{i}.attn.c_attn.w [768, 2304].T   transformers.{i}.mha.W_q/k/v.weight (拆分后各转置)
    h{i}.attn.c_attn.b [2304]          transformers.{i}.mha.W_q/k/v.bias   (拆分3份)
    h{i}.attn.c_proj.w [768, 768].T    transformers.{i}.mha.W_o.weight
    h{i}.attn.c_proj.b [768]           transformers.{i}.mha.W_o.bias
    h{i}.ln_1.g / .b [768]             transformers.{i}.layer_norm1.weight / bias
    h{i}.ln_2.g / .b [768]             transformers.{i}.layer_norm2.weight / bias
    h{i}.mlp.c_fc.w [3072, 768].T      transformers.{i}.ffn.feed_forward_layer[0].weight
    h{i}.mlp.c_fc.b [3072]             transformers.{i}.ffn.feed_forward_layer[0].bias
    h{i}.mlp.c_proj.w [768, 3072].T    transformers.{i}.ffn.feed_forward_layer[2].weight
    h{i}.mlp.c_proj.b [768]            transformers.{i}.ffn.feed_forward_layer[2].bias
    ln_f.g / .b [768]                  norm_layer.weight / bias
    wte [50257, 768]                   output_layer.weight (权重共享，无需额外转置)
"""

import torch
import numpy as np


def _assign(dest: torch.nn.Parameter, src: np.ndarray) -> None:
    """将 numpy 数组拷贝到模型参数（原地操作）。"""
    dest.data.copy_(torch.from_numpy(src))


def load_gpt2_weights(model, params: dict, config) -> None:
    """
    将 GPT-2 checkpoint 参数加载到 SelfGPTModel 实例中。

    Args:
        model: SelfGPTModel 实例，架构参数需与 GPT-2 124M 一致。
        params: gpt_download.download_and_load_gpt2() 返回的参数字典。
        config: ModelConfig 配置对象，提供 num_layers 等信息。
    """
    num_layers = config.num_layers

    # ── 1. Token Embedding（无需转置）─────────────────────────────
    _assign(model.embedding.embedding.weight, params["wte"])

    # ── 2. Position Embedding（无需转置）──────────────────────────
    _assign(model.embedding.position_embedding.weight, params["wpe"])

    # ── 3. Transformer Blocks ───────────────────────────────────
    for i in range(num_layers):
        block = model.transformers[i]
        bp = params["blocks"][i]

        # --- Attention: c_attn → W_q, W_k, W_v ---
        # TF 形状: [hidden_size, 3*hidden_size], 需转置并沿 dim=1 拆为 3 份
        c_attn_w = bp["attn"]["c_attn"]["w"].T  # [2304, 768]
        c_attn_b = bp["attn"]["c_attn"]["b"]     # [2304]
        q_w, k_w, v_w = np.split(c_attn_w, 3, axis=0)  # 每个 [768, 768]
        q_b, k_b, v_b = np.split(c_attn_b, 3, axis=0)

        _assign(block.mha.W_q.weight, q_w)
        _assign(block.mha.W_q.bias, q_b)
        _assign(block.mha.W_k.weight, k_w)
        _assign(block.mha.W_k.bias, k_b)
        _assign(block.mha.W_v.weight, v_w)
        _assign(block.mha.W_v.bias, v_b)

        # --- Attention: c_proj → W_o ---
        _assign(block.mha.W_o.weight, bp["attn"]["c_proj"]["w"].T)
        _assign(block.mha.W_o.bias, bp["attn"]["c_proj"]["b"])

        # --- LayerNorm: ln_1 / ln_2（1D，无需转置）---
        _assign(block.layer_norm1.weight, bp["ln_1"]["g"])
        _assign(block.layer_norm1.bias, bp["ln_1"]["b"])
        _assign(block.layer_norm2.weight, bp["ln_2"]["g"])
        _assign(block.layer_norm2.bias, bp["ln_2"]["b"])

        # --- FFN: c_fc / c_proj ---
        _assign(block.ffn.feed_forward_layer[0].weight, bp["mlp"]["c_fc"]["w"].T)
        _assign(block.ffn.feed_forward_layer[0].bias, bp["mlp"]["c_fc"]["b"])
        _assign(block.ffn.feed_forward_layer[2].weight, bp["mlp"]["c_proj"]["w"].T)
        _assign(block.ffn.feed_forward_layer[2].bias, bp["mlp"]["c_proj"]["b"])

    # ── 4. Final LayerNorm（1D，无需转置）────────────────────────
    # 注意：TF checkpoint 中 ln_f 的 g/b 直接存储在顶层 key
    _assign(model.norm_layer.weight, params["g"])
    _assign(model.norm_layer.bias, params["b"])

    # ── 5. Output Layer (权重共享: wte 无需转置，PyTorch Linear自动处理) ──
    _assign(model.output_layer.weight, params["wte"])

    print(f"[load_gpt2_weights] 成功加载 GPT-2 124M 预训练权重，共 {num_layers} 层 Transformer")
