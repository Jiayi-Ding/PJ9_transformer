# -*- coding: utf-8 -*-
"""
在 Kaggle 上批量训练 6 个词牌/合并组模型，生成对应的 *_model.pt 文件。

用法：
1. Kaggle Notebook 添加本 Dataset 作为 Input。
2. 在 Notebook 里运行：
   !python kaggle_train_six_models.py

默认训练目标：
- huanxisha      浣溪沙
- shuidiaogetou  水调歌头
- zhegutian      鹧鸪天
- pusaman        菩萨蛮
- dielianhua     蝶恋花
- merge_medium   其他小词牌/小样本合并组

说明：
- 本脚本会把 /kaggle/input 里的数据复制到 /kaggle/working，再依次调用 train.py。
- 为了“模拟用户预测”快速得到 pt，默认参数偏轻量；如果追求效果，可增大 EPOCHS、TRAIN_NUM_SAMPLES。
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path


DATASET_SLUG = os.environ.get("DATASET_SLUG", "pj9-cipai-v3")
WORK_DIR = Path("/kaggle/working") if Path("/kaggle").exists() else Path.cwd()

# 用户交互界面中展示的 6 个选项
CIPAI_OPTIONS = {
    "huanxisha": "浣溪沙",
    "shuidiaogetou": "水调歌头",
    "zhegutian": "鹧鸪天",
    "pusaman": "菩萨蛮",
    "dielianhua": "蝶恋花",
    "merge_medium": "其他小词牌（小样本合并组）",
}

# 快速生成可预测模型的默认训练参数。需要更好效果时可以通过环境变量覆盖。
EPOCHS = int(os.environ.get("EPOCHS", "3"))
TRAIN_NUM_SAMPLES = int(os.environ.get("TRAIN_NUM_SAMPLES", "30000"))
VAL_BATCHES = int(os.environ.get("VAL_BATCHES", "80"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "32"))
MAX_TRAIN_BATCHES = int(os.environ.get("MAX_TRAIN_BATCHES", "0"))


def find_dataset_dir() -> Path:
    """自动在 /kaggle/input 中寻找包含本项目文件的 Dataset 目录。"""
    input_root = Path("/kaggle/input")
    required = "train.py"
    train_file = "huanxisha_train.pt"

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
        if (candidate / required).is_file() and (candidate / train_file).is_file():
            return candidate

    if input_root.exists():
        for path in input_root.rglob(required):
            parent = path.parent
            if (parent / train_file).is_file():
                return parent

    raise FileNotFoundError(
        "没有找到 Kaggle Dataset 目录。请确认右侧 Add Input 已添加 pj9-cipai-v3，"
        "且 Dataset 内包含 train.py 和 huanxisha_train.pt。"
    )


def copy_dataset_to_working(src: Path, dst: Path) -> None:
    """复制 Dataset 文件到 /kaggle/working，避免在只读 input 目录训练。"""
    dst.mkdir(parents=True, exist_ok=True)
    for item in src.iterdir():
        target = dst / item.name
        if item.is_file():
            shutil.copy2(item, target)
    print(f"已复制 Dataset: {src} -> {dst}")


def ensure_required_files(work_dir: Path) -> None:
    missing: list[str] = []
    for name in CIPAI_OPTIONS:
        for suffix in ("train.pt", "val.pt", "vocab.json"):
            file_name = f"{name}_{suffix}"
            if not (work_dir / file_name).is_file():
                missing.append(file_name)
    if missing:
        raise FileNotFoundError("缺少以下训练文件：\n" + "\n".join(missing))


def train_one(name: str, label: str) -> None:
    model_path = WORK_DIR / f"{name}_model.pt"
    if model_path.is_file():
        print(f"\n=== 跳过 {label}：已存在 {model_path.name} ===")
        return

    print(f"\n=== 开始训练 {label} ({name}) ===")
    cmd = [
        sys.executable,
        "train.py",
        "--poetry_name",
        name,
        "--epochs",
        str(EPOCHS),
        "--train_num_samples",
        str(TRAIN_NUM_SAMPLES),
        "--val_batches",
        str(VAL_BATCHES),
        "--batch_size",
        str(BATCH_SIZE),
        "--max_train_batches",
        str(MAX_TRAIN_BATCHES),
        "--log_plot",
        f"loss_{name}.png",
    ]
    print("运行命令:", " ".join(cmd))
    subprocess.run(cmd, cwd=WORK_DIR, check=True)

    if not model_path.is_file():
        raise RuntimeError(f"训练结束后没有找到模型文件: {model_path}")
    print(f"=== 完成 {label}: {model_path} ===")


def main() -> None:
    if Path("/kaggle/input").exists():
        src = find_dataset_dir()
        copy_dataset_to_working(src, WORK_DIR)
    else:
        print("未检测到 Kaggle 环境，将在当前目录运行。")

    os.chdir(WORK_DIR)
    sys.path.insert(0, str(WORK_DIR))
    ensure_required_files(WORK_DIR)

    print("训练配置:")
    print("  EPOCHS =", EPOCHS)
    print("  TRAIN_NUM_SAMPLES =", TRAIN_NUM_SAMPLES)
    print("  VAL_BATCHES =", VAL_BATCHES)
    print("  BATCH_SIZE =", BATCH_SIZE)
    print("  MAX_TRAIN_BATCHES =", MAX_TRAIN_BATCHES)

    for name, label in CIPAI_OPTIONS.items():
        train_one(name, label)

    print("\n全部模型已生成：")
    for name, label in CIPAI_OPTIONS.items():
        print(f"  {label}: {WORK_DIR / (name + '_model.pt')}")

    print("\n现在可以运行交互预测脚本：")
    print("  !python kaggle_gradio_predict.py")


if __name__ == "__main__":
    main()
