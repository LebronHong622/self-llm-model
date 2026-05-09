"""验证 setup_classification_model() — 使用 importlib 绕过 fine-tune 目录名中的连字符"""

import sys
import torch
import importlib.util

# 用 importlib 加载（fine-tune 含连字符，不能直接 import）
_spec = importlib.util.spec_from_file_location(
    "classfication_mod", "trainer/fine-tune/classfication.py"
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

setup_classification_model = _mod.setup_classification_model
from config.config import get_config, Environment

config = get_config(Environment.TEST)
device = torch.device("cpu")

print("=" * 55)
print("[TEST] setup_classification_model() 功能验证")
print("=" * 55)

model = setup_classification_model(config, device, num_classes=2)

# ---- 1. 输出头 ----
print("\n--- 验证1: 输出头 ---")
ol = model.output_layer
assert isinstance(ol, torch.nn.Linear) and ol.out_features == 2
print(f"  [PASS] Linear({ol.in_features}, {ol.out_features})")

# ---- 2. 冻结状态 ----
print("\n--- 验证2: 冻结策略 ---")
status = {}
for name, param in model.named_parameters():
    top = name.split(".")[0]
    if top not in status:
        status[top] = {"t": 0, "f": 0}
    key = "t" if param.requires_grad else "f"
    status[top][key] += param.numel()

for m, c in status.items():
    tag = "✓可训练" if c["t"] > 0 else "❌冻结"
    print(f"  {m:20s} | 可训练:{c['t']:>10,} | 冻结:{c['f']:>12,} | {tag}")

assert status["embedding"]["t"] == 0, "FAIL: embedding应冻结"
assert status["norm_layer"]["f"] == 0, "FAIL: norm_layer应全可训练"
assert status["output_layer"]["f"] == 0, "FAIL: output_layer应全可训练"

tl = set(); fl = set()
for n, p in model.named_parameters():
    if n.startswith("transformers"):
        (tl if p.requires_grad else fl).add(n.split(".")[1])
last_idx = str(config.model.num_layers - 1)
assert tl == {last_idx}, f"FAIL: 仅{last_idx}层应可训练，实际:{tl}"
print(f"\n  [PASS] embedding冻结 / norm_layer+output_layer可训练")
print(f"  [PASS] transformers[{last_idx}]可训练, 其余{config.model.num_layers-1}层冻结")

# ---- 3. 前向传播 ----
print("\n--- 验证3: 前向传播形状 ---")
model.eval()
x = torch.randint(0, config.model.vocab_size, (2, 32))
with torch.no_grad():
    out = model(x)
    cls_out = out[:, -1, :]
assert out.shape == (2, 32, 2) and cls_out.shape == (2, 2)
print(f"  [PASS] 输出:{out.shape} -> 分类logits:{cls_out.shape}")

# ---- 4. 参数比例 ----
tp = sum(p.numel() for p in model.parameters() if p.requires_grad)
tot = sum(p.numel() for p in model.parameters())
pct = tp / tot * 100
print(f"\n--- 验证4: 参数统计 ---")
print(f"  总参数:{tot:,} | 可训练:{tp:,} ({pct:.2f}%)")
assert pct < 10.0, f"FAIL: 可训练比例过高 {pct:.2f}%"
print("  [PASS] 比例合理 (<10%)")

print("\n" + "=" * 55)
print("[ALL PASSED] 全部验证通过 ✓")
print("=" * 55)
