# -*- coding: utf-8 -*-
"""
Kaggle Gradio 交互预测界面。

前提：已经在 /kaggle/working 下生成以下模型：
- huanxisha_model.pt
- shuidiaogetou_model.pt
- zhegutian_model.pt
- pusaman_model.pt
- dielianhua_model.pt
- merge_medium_model.pt

Kaggle Notebook 运行：
!pip -q install gradio
!python kaggle_gradio_predict.py
"""
from __future__ import annotations

import os
import shutil
import sys
from functools import lru_cache
from pathlib import Path


DATASET_SLUG = os.environ.get("DATASET_SLUG", "pj9-cipai-v3")
WORK_DIR = Path("/kaggle/working") if Path("/kaggle").exists() else Path.cwd()

CIPAI_OPTIONS = {
    "浣溪沙": "huanxisha",
    "水调歌头": "shuidiaogetou",
    "鹧鸪天": "zhegutian",
    "菩萨蛮": "pusaman",
    "蝶恋花": "dielianhua",
    "其他小词牌": "merge_medium",
}


def find_dataset_dir() -> Path | None:
    input_root = Path("/kaggle/input")
    if not input_root.exists():
        return None

    candidates = [
        input_root / DATASET_SLUG / "pj9-cipai-v3",
        input_root / DATASET_SLUG,
    ]
    datasets_root = input_root / "datasets"
    if datasets_root.is_dir():
        for user_dir in datasets_root.iterdir():
            if user_dir.is_dir():
                candidates.extend([
                    user_dir / DATASET_SLUG / "pj9-cipai-v3",
                    user_dir / DATASET_SLUG,
                ])

    for candidate in candidates:
        if (candidate / "predict.py").is_file():
            return candidate

    for path in input_root.rglob("predict.py"):
        return path.parent
    return None


def copy_runtime_files() -> None:
    """如果工作目录缺少代码/词表，则从 Kaggle Input 复制。"""
    src = find_dataset_dir()
    if src is None:
        return
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    for item in src.iterdir():
        target = WORK_DIR / item.name
        if item.is_file() and not target.exists():
            shutil.copy2(item, target)


copy_runtime_files()
os.chdir(WORK_DIR)
sys.path.insert(0, str(WORK_DIR))

import gradio as gr  # noqa: E402
import torch  # noqa: E402

from dataset import load_vocab_json  # noqa: E402
from predict import first_in_vocab_char, generate_one, load_for_generate  # noqa: E402
from train import get_device  # noqa: E402


device = get_device()


@lru_cache(maxsize=8)
def load_model_and_vocab(poetry_name: str):
    ckpt = WORK_DIR / f"{poetry_name}_model.pt"
    vocab = WORK_DIR / f"{poetry_name}_vocab.json"
    if not ckpt.is_file():
        raise FileNotFoundError(
            f"没有找到模型文件 {ckpt.name}。请先运行：!python kaggle_train_six_models.py"
        )
    if not vocab.is_file():
        raise FileNotFoundError(f"没有找到词表文件 {vocab.name}")

    stoi, itos, _ = load_vocab_json(str(vocab))
    model, _ = load_for_generate(str(ckpt), device)
    return model, stoi, itos


def predict(cipai_label: str, prompt: str, max_new: int, temperature: float, stop_newline: bool) -> str:
    poetry_name = CIPAI_OPTIONS[cipai_label]
    model, stoi, itos = load_model_and_vocab(poetry_name)

    raw = (prompt or "春").strip() or "春"
    try:
        seed = first_in_vocab_char(raw, stoi)
        text = generate_one(
            model=model,
            stoi=stoi,
            itos=itos,
            device=device,
            prompt=seed,
            max_new=int(max_new),
            temperature=float(temperature),
            stop_newline=bool(stop_newline),
        )
    except Exception as exc:  # noqa: BLE001
        return f"生成失败：{exc}"

    return text.strip()


def model_status() -> str:
    lines = [f"当前设备：{device}", "模型文件检查："]
    for label, name in CIPAI_OPTIONS.items():
        ok = (WORK_DIR / f"{name}_model.pt").is_file()
        lines.append(f"- {label}: {'已找到' if ok else '未找到'} {name}_model.pt")
    return "\n".join(lines)


with gr.Blocks(title="词牌生成交互预测") as demo:
    gr.Markdown(
        "# 词牌生成交互预测\n"
        "选择词牌风格，输入一个起笔字，模型会按对应风格续写。\n\n"
        "可选风格：浣溪沙、水调歌头、鹧鸪天、菩萨蛮、蝶恋花、其他小词牌。"
    )
    gr.Markdown(model_status())

    with gr.Row():
        cipai = gr.Dropdown(
            choices=list(CIPAI_OPTIONS.keys()),
            value="浣溪沙",
            label="词牌风格",
        )
        prompt = gr.Textbox(value="春", label="起笔字/起始文本", max_lines=1)

    with gr.Row():
        max_new = gr.Slider(20, 300, value=120, step=10, label="生成长度")
        temperature = gr.Slider(0.2, 1.5, value=0.9, step=0.05, label="随机性 temperature")
        stop_newline = gr.Checkbox(value=False, label="遇到换行提前停止")

    btn = gr.Button("生成")
    output = gr.Textbox(label="生成结果", lines=10)

    btn.click(
        fn=predict,
        inputs=[cipai, prompt, max_new, temperature, stop_newline],
        outputs=output,
    )


if __name__ == "__main__":
    demo.launch(share=True, debug=True)
