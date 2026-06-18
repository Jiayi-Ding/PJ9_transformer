#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
一键批量训练脚本。

用途：
- 依次训练通用古诗模型：5 / 7 / ci
- 依次训练词牌模型：huanxisha / shuidiaogetou / zhegutian / pusaman / dielianhua / merge_medium
- 已存在的模型文件会自动跳过，方便中断后续跑

用法：
    python train_all.py

可选环境变量：
    EPOCHS=3               # 覆盖单模型训练轮数
    TRAIN_NUM_SAMPLES=50000
    USE_TOPIC=0            # 若你的 train.py 支持主题，可自行扩展
    REPEAT_PENALTY=0       # 若你的 train.py 支持重复项惩罚，可自行扩展
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


GENRES = ["5", "7", "ci"]
POETRY_NAMES = [
    "huanxisha",
    "shuidiaogetou",
    "zhegutian",
    "pusaman",
    "dielianhua",
    "merge_medium",
]


def run_cmd(cmd: list[str]) -> int:
    print("\n" + "=" * 80)
    print("执行命令：", " ".join(cmd))
    print("=" * 80)
    return subprocess.call(cmd)


def build_env_args() -> list[str]:
    args: list[str] = []
    epochs = os.environ.get("EPOCHS")
    train_num_samples = os.environ.get("TRAIN_NUM_SAMPLES")
    if epochs:
        args.extend(["--epochs", epochs])
    if train_num_samples:
        args.extend(["--train_num_samples", train_num_samples])
    return args


def main() -> None:
    here = Path(__file__).resolve().parent
    os.chdir(here)

    if not Path("train.py").is_file():
        print("错误：当前目录下找不到 train.py，请确认你已切换到 v10 目录。", file=sys.stderr)
        sys.exit(1)

    extra_args = build_env_args()
    tasks: list[list[str]] = []

    # 先训练通用古诗
    for genre in GENRES:
        ckpt = Path(f"ckpt_best_{genre}.pt")
        if ckpt.is_file():
            print(f"跳过 genre={genre}，已存在 {ckpt.name}")
            continue
        tasks.append([sys.executable, "train.py", "--genre", genre, *extra_args])

    # 再训练词牌
    for name in POETRY_NAMES:
        ckpt = Path(f"{name}_model.pt")
        if ckpt.is_file():
            print(f"跳过 poetry_name={name}，已存在 {ckpt.name}")
            continue
        tasks.append([sys.executable, "train.py", "--poetry_name", name, *extra_args])

    if not tasks:
        print("没有需要训练的任务，所有模型似乎都已存在。")
        return

    print(f"共发现 {len(tasks)} 个待训练任务。")
    failed = []
    for i, cmd in enumerate(tasks, start=1):
        print(f"\n[{i}/{len(tasks)}] 开始训练...")
        code = run_cmd(cmd)
        if code != 0:
            failed.append((i, code, cmd))
            print(f"任务失败，退出码：{code}", file=sys.stderr)
            break
        print(f"[{i}/{len(tasks)}] 完成。")

    if failed:
        print("\n以下任务失败：", file=sys.stderr)
        for idx, code, cmd in failed:
            print(f"- 第 {idx} 个任务，退出码 {code}: {' '.join(cmd)}", file=sys.stderr)
        sys.exit(failed[0][1])

    print("\n全部训练任务已完成。")


if __name__ == "__main__":
    main()
