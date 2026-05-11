# -*- coding: utf-8 -*-
"""
分类模型推理验证脚本

功能模块:
    1. load_classification_model()   — 加载已保存的分类模型 checkpoint
    2. predict_text()                 — 单条文本推理（输入文本 → spam / not spam）
    3. demo_eval()                    — 预设测试用例批量验证
    4. interactive_eval()             — 交互式模式，持续读取用户输入文本并分类
"""

import torch
import torch.nn.functional as F
from model.self_gpt import SelfGPTModel


# ==================== 默认配置 ====================

DEFAULT_CHECKPOINT_PATH = "model/fine-tune/classfication/classification_model.pth"
LABEL_MAP = {0: "ham (not spam)", 1: "spam"}
DEMO_TEXTS = [
    ("Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005.", "spam"),
    ("I'm going to be late today, see you tomorrow.", "ham"),
    ("WINNER!! As a valued network customer selected receive a £900 prize reward!", "spam"),
    ("Can we reschedule our meeting to 3pm?", "ham"),
    ("URGENT! Your Mobile No. was awarded £2000 Bonus Caller Prize call 09071512345", "spam"),
    ("Hey, are we still on for lunch?", "ham"),
    ("Congratulations ur awarded $1000 Walmart gift card go to walmart.com", "spam"),
    ("Don't forget to pick up some milk on your way home.", "ham"),
]


# ==================== 核心函数 ====================

def load_classification_model(
    checkpoint_path: str = DEFAULT_CHECKPOINT_PATH,
    device: str = "cpu",
) -> tuple[SelfGPTModel, object, dict]:
    """
    加载分类模型 checkpoint 并重建模型。

    从保存的 .pt 文件中恢复:
        - 完整模型权重（含替换后的分类头）
        - 训练时的完整配置
        - 标签映射

    Args:
        checkpoint_path: 模型 checkpoint 路径
        device:          推理设备，默认 "cpu"

    Returns:
        (model, config, label_map) 元组
            model:     已加载权重、设为 eval 模式的 SelfGPTModel
            config:    配置对象
            label_map: 标签映射字典
    """
    from config.config import Config

    device_obj = torch.device(device)
    print(f"[load] 正在加载模型: {checkpoint_path}")
    print(f"[load] 设备: {device}")

    # ---- 1. 加载 checkpoint ----
    checkpoint = torch.load(checkpoint_path, map_location=device_obj, weights_only=False)
    state_dict = checkpoint["model_state_dict"]
    config_dict = checkpoint["config_dict"]
    num_classes = checkpoint["num_classes"]
    label_map = checkpoint["label_map"]

    # ---- 2. 重建 Config 对象 ----
    config = Config(**config_dict)

    # ---- 3. 初始化模型结构（与训练时一致）----
    model = SelfGPTModel(config.model)

    # 替换输出头为分类头（与训练时的 setup_classification_model 一致）
    model.output_layer = torch.nn.Linear(
        config.model.hidden_size, num_classes
    )

    # ---- 4. 加载权重 ----
    model.load_state_dict(state_dict)

    # ---- 5. 设置为推理模式 ----
    model.to(device_obj)
    model.eval()

    print(f"[load] 模型加载成功!")
    print(f"[load] hidden_size={config.model.hidden_size}, "
          f"num_layers={config.model.num_layers}, "
          f"num_classes={num_classes}")
    print(f"[load] 标签映射: {label_map}\n")

    return model, config, label_map


def predict_text(
    text: str,
    model: SelfGPTModel,
    tokenizer,
    device: torch.device,
    label_map: dict | None = None,
    max_length: int = 128,
) -> tuple[str, float, torch.Tensor]:
    """
    单条文本分类推理。

    流程:
        输入文本 → tiktoken encode → pad/truncate → tensor → model forward
        → 取最后一个 token logits → softmax → argmax → label_map 映射

    Args:
        text:       待分类的原始文本
        model:      已加载的分类模型
        tokenizer:  tiktoken 编码器实例
        device:     推理设备
        label_map:  标签映射，默认 {0: "ham (not spam)", 1: "spam"}
        max_length: 最大序列长度

    Returns:
        (predicted_label, confidence, logits) 元组
            predicted_label: 预测的标签字符串（如 "spam" / "ham (not spam)"）
            confidence:       置信度 (0~1 的 softmax 概率值)
            logits:           原始 logit 向量（用于调试）
    """
    if label_map is None:
        label_map = LABEL_MAP

    pad_token_id = 50256  # <EOS>

    # ---- 编码 ----
    encoded = tokenizer.encode(text)

    # 截断或填充
    if len(encoded) > max_length:
        encoded = encoded[:max_length]
    else:
        encoded = encoded + [pad_token_id] * (max_length - len(encoded))

    input_ids = torch.tensor([encoded], dtype=torch.long).to(device)

    # ---- 推理 ----
    with torch.no_grad():
        logits = model(input_ids)
        # 取最后一个 token 的 logits（与训练逻辑一致）
        cls_logits = logits[:, -1, :]
        probs = F.softmax(cls_logits, dim=-1)
        pred_idx = torch.argmax(probs, dim=-1).item()
        confidence = probs[0, pred_idx].item()

    predicted_label = label_map.get(pred_idx, f"unknown({pred_idx})")

    return predicted_label, confidence, cls_logits[0]


# ==================== Demo 与交互模式 ====================

def demo_eval(
    model: SelfGPTModel,
    tokenizer,
    device: torch.device,
    label_map: dict,
):
    """
    使用预设测试用例进行批量验证，打印结果表格。
    """
    print("=" * 70)
    print("=== 分类模型 Demo 测试 ===")
    print("=" * 70)

    correct = 0
    total = len(DEMO_TEXTS)

    for i, (text, expected_label) in enumerate(DEMO_TEXTS, start=1):
        pred_label, confidence, _ = predict_text(
            text, model, tokenizer, device, label_map
        )

        is_correct = (
            ("spam" in expected_label.lower() and "spam" == pred_label.lower()) or
            ("ham" in expected_label.lower() and "ham" in pred_label.lower())
        )
        status = "✓" if is_correct else "✗"
        if is_correct:
            correct += 1

        display_text = text[:55] + ("..." if len(text) > 55 else "")
        short_pred = "SPAM" if pred_label.lower() == "spam" else "NOT SPAM"
        print(f"\n[{i}] \"{display_text}\"")
        print(f"    预测: {short_pred} (置信度: {confidence:.4f}) | 期望: {expected_label} {status}")

    print("\n" + "-" * 70)
    print(f"Demo 结果: {correct}/{total} 正确 (准确率: {correct / total * 100:.1f}%)")
    print("-" * 70)


def interactive_eval(
    model: SelfGPTModel,
    tokenizer,
    device: torch.device,
    label_map: dict,
):
    """
    交互式验证循环。

    持续等待用户输入文本，输出预测结果。输入 quit / exit / q 退出。
    """
    print("\n" + "=" * 50)
    print("=== 交互模式 (输入 quit / exit / q 退出) ===")
    print("=" * 50)

    while True:
        try:
            text = input("\n请输入待分类文本 > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n检测到中断信号，退出交互模式。")
            break

        if text.lower() in ("quit", "exit", "q"):
            print("再见！")
            break

        if not text:
            continue

        pred_label, confidence, _ = predict_text(
            text, model, tokenizer, device, label_map
        )

        short_pred = "SPAM" if pred_label.lower() == "spam" else "NOT SPAM"
        print(f"  → 预测: {short_pred} (置信度: {confidence:.4f})")


# ==================== 主入口 ====================

if __name__ == "__main__":
    import tiktoken

    # ---- 加载模型 ----
    tokenizer = tiktoken.get_encoding("gpt2")
    device = torch.device("cuda")
    model, config, label_map = load_classification_model(
        checkpoint_path=DEFAULT_CHECKPOINT_PATH,
        device="cuda",
    )

    # ---- 运行 demo 测试 ----
    demo_eval(model, tokenizer, device, label_map)

    # ---- 进入交互模式 ----
    interactive_eval(model, tokenizer, device, label_map)
