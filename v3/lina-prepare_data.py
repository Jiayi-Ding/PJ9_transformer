#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
古诗数据集下载与预处理脚本
- 从 Hugging Face 下载真实唐宋古诗数据
- 构建字符级词表并保存 vocab.json
- 划分 9:1 训练/验证集，编码后保存为 train_data_*.pt / val_data_*.pt

改进点：
1.划分为5言、7言、词 三类样本
2.为避免训练数据不足问题，对每类样本进行补齐，保证每样取样 100000 条
3.使用 opencc 将繁体字转换为简体字，增加诗句的可读性。
"""

import argparse
import json
import os
import re
import sys

import torch

from poetry_config import (
    POETRY_LABEL,
    POETRY_NAME,
    poetry_label,
    poetry_txt,
    train_pt,
    val_pt,
    vocab_json,
)

try:
    from opencc import OpenCC

    cc = OpenCC("t2s")
except ImportError:
    cc = None

GENRES = ("5", "7", "ci")

# 方案 2：按类别目标数量补齐
TARGET_PER_GENRE = {"5": 100000, "7": 100000, "ci": 100000}
MAX_PASS = 5
PASS_SAMPLE_SIZE = 100000


def normalize_poem_text(text: str) -> str:
    text = str(text).strip()
    text = text.replace("\r", "")
    if cc is not None:
        text = cc.convert(text)
    return text


def split_poem_lines(text: str) -> list[str]:
    if not text:
        return []
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    if len(lines) > 1:
        return lines
    # 若无换行，则按常见标点分句
    segments = re.split(r"[，。；？！、]+", text)
    return [seg.strip() for seg in segments if seg.strip()]


def count_chinese_chars(text: str) -> int:
    return len(re.findall(r"[\u4e00-\u9fff]", text))


def classify_poem(text: str) -> str:
    """
    对一首诗进行分类：5 言、7 言或词/杂言。
    规则：按句长比例判断，若 5 言句占比 >= 0.6 则为 5 言，7 言句占比 >= 0.6 则为 7 言，否则为词。
    注意：取消了原有的词牌名判断，避免误判。
    """
    text = normalize_poem_text(text)
    lines = split_poem_lines(text)
    if not lines:
        return "ci"

    counts = [count_chinese_chars(line) for line in lines]
    counts = [c for c in counts if c > 0]
    if not counts:
        return "ci"

    # 计算五言和七言的比例
    n5 = sum(1 for c in counts if c == 5)
    n7 = sum(1 for c in counts if c == 7)
    total = len(counts)

    if n5 / total >= 0.6:
        return "5"
    if n7 / total >= 0.6:
        return "7"
    return "ci"


def write_text_file(path: str, poems: list[str]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n\n".join(poems))


def encode_and_save(text: str, stoi: dict, out_path: str) -> None:
    ids = [stoi[c] for c in text]
    data = torch.tensor(ids, dtype=torch.long)
    torch.save(data, out_path)


def load_poems_from_txt(path: str) -> list[str]:
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read()
    poems = [normalize_poem_text(p) for p in raw.split("\n\n") if p.strip()]
    return [p for p in poems if p]


def prepare_cipai_data(poetry_name: str | None = None) -> None:
    """
    按单一词牌 txt 预处理：构建词表并保存 {name}_train.pt / {name}_val.pt / {name}_vocab.json。
    数据处理流程与体裁分支一致，仅输入输出文件名不同。
    """
    name = poetry_name or POETRY_NAME
    label = poetry_label(name)
    filename = poetry_txt(name)
    out_train = train_pt(name)
    out_val = val_pt(name)
    out_vocab = vocab_json(name)

    print("=" * 50)
    print(f"词牌预处理：{label}（{name}）")
    print(f"当前词牌：{label}")
    print("=" * 50)

    if not os.path.isfile(filename):
        raise FileNotFoundError(
            f"缺少 {filename}，请先运行：python split_cipai.py --cipai {label}"
        )

    poems = load_poems_from_txt(filename)
    if not poems:
        raise ValueError(f"{filename} 中未找到有效词作。")

    print(f"读取 {filename}：{len(poems)} 首")

    all_text = "\n\n".join(poems)
    chars = sorted(list(set(all_text)))
    vocab_size = len(chars)
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    vocab = {"vocab_size": vocab_size, "stoi": stoi, "itos": itos}

    with open(out_vocab, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)
    print(f"词表大小: {vocab_size}，已保存 {out_vocab}")

    n_train = int(len(poems) * 0.9)
    train_poems = poems[:n_train]
    val_poems = poems[n_train:]
    train_text = "\n\n".join(train_poems)
    val_text = "\n\n".join(val_poems)

    encode_and_save(train_text, stoi, out_train)
    encode_and_save(val_text, stoi, out_val)

    print(f"训练集: {len(train_poems)} 首, token 数: {len(train_text)} -> {out_train}")
    print(f"验证集: {len(val_poems)} 首, token 数: {len(val_text)} -> {out_val}")
    print(f"\n词牌「{label}」预处理完成。")


def main():
    if cc is None:
        print("请先安装 opencc：pip install opencc-python-reimplemented")
        sys.exit(1)

    # 使用国内镜像加速下载
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    from datasets import load_dataset

    print("=" * 50)
    print("步骤 1：下载真实唐宋古诗数据集")
    print("=" * 50)
    dataset = load_dataset("Lifan-Z/Chinese-poetries-txt", split="train")
    print(f"数据集总条数: {len(dataset)}")
    print(f"列名: {dataset.column_names}")

    dataset = dataset.shuffle(seed=42)
    text_col = "text" if "text" in dataset.column_names else dataset.column_names[0]
    print(f"使用列: {text_col}")

    print("\n" + "=" * 50)
    print("步骤 2：按体裁目标数量补齐样本（繁体已转简体）")
    print("=" * 50)

    poems_by_genre = {g: [] for g in GENRES}
    all_texts = []
    seen_texts = set()
    targets = TARGET_PER_GENRE.copy()
    print(f"目标每类样本数: {targets}")

    added = True
    for pass_i in range(MAX_PASS):
        if all(len(poems_by_genre[g]) >= targets[g] for g in GENRES):
            break
        if not added and pass_i > 0:
            break
        added = False

        print(f"Pass {pass_i + 1}/{MAX_PASS}：随机抽样最多 {PASS_SAMPLE_SIZE} 条，按体裁补齐")
        shuffled = dataset.shuffle(seed=42 + pass_i)
        if len(shuffled) > PASS_SAMPLE_SIZE:
            shuffled = shuffled.select(range(PASS_SAMPLE_SIZE))

        for item in shuffled:
            text = item[text_col]
            if text is None:
                continue
            text = normalize_poem_text(text)  # 此处已含繁简转换
            if not text or text in seen_texts:
                continue
            seen_texts.add(text)
            genre = classify_poem(text)       # 此处不再进行词牌名判断
            if len(poems_by_genre[genre]) < targets[genre]:
                poems_by_genre[genre].append(text)
                all_texts.append(text)
                added = True
            if all(len(poems_by_genre[g]) >= targets[g] for g in GENRES):
                break

    all_text = "\n\n".join(all_texts)
    with open("poetry.txt", "w", encoding="utf-8") as f:
        f.write(all_text)

    print(f"实际抽样条数: {len(all_texts)}")
    print(f"已保存 poetry.txt，总字符数约: {len(all_text)}")
    for genre in GENRES:
        print(f"  体裁 {genre}：{len(poems_by_genre[genre])} 条（目标 {targets[genre]}）")

    print("\n" + "=" * 50)
    print("步骤 3：构建字符级词表并保存 vocab.json")
    print("=" * 50)
    chars = sorted(list(set(all_text)))
    vocab_size = len(chars)
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}

    vocab = {
        "vocab_size": vocab_size,
        "stoi": stoi,
        "itos": itos,
    }

    with open("vocab.json", "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)

    print(f"词表大小: {vocab_size}")

    print("\n" + "=" * 50)
    print("步骤 4：按体裁划分 9:1 训练/验证集并保存 .pt 文件")
    print("=" * 50)
    for genre in GENRES:
        poems = poems_by_genre[genre]
        print(f"体裁 {genre}：{len(poems)} 首")
        if not poems:
            print(f"  跳过体裁 {genre}，未找到数据。")
            continue
        write_text_file(f"poetry_{genre}.txt", poems)

        n_train = int(len(poems) * 0.9)
        train_poems = poems[:n_train]
        val_poems = poems[n_train:]

        train_text = "\n\n".join(train_poems)
        val_text = "\n\n".join(val_poems)

        encode_and_save(train_text, stoi, f"train_data_{genre}.pt")
        encode_and_save(val_text, stoi, f"val_data_{genre}.pt")

        print(f"  保存: poetry_{genre}.txt")
        print(f"  训练集: {len(train_poems)} 首, token 数: {len(train_text)}")
        print(f"  验证集: {len(val_poems)} 首, token 数: {len(val_text)}")

    print("\n预处理全部完成。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="古诗/词牌数据预处理")
    parser.add_argument(
        "--cipai",
        action="store_true",
        help="按 poetry_config 中 POETRY_NAME 对应的词牌 txt 单独预处理",
    )
    parser.add_argument(
        "--poetry_name",
        type=str,
        default=None,
        help="词牌文件前缀，如 rumengling；默认使用 poetry_config.POETRY_NAME",
    )
    cli = parser.parse_args()

    if cli.cipai or cli.poetry_name:
        prepare_cipai_data(cli.poetry_name)
    else:
        main()
