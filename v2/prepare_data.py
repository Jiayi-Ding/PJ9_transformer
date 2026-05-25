#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
古诗数据集下载与预处理脚本
- 从 Hugging Face 下载真实唐宋古诗数据
- 随机抽样最多 100000 条并按体裁分类为 5言、7言、词
- 构建字符级词表并保存 vocab.json
- 划分 9:1 训练/验证集，编码后保存为 train_data_*.pt / val_data_*.pt
"""

import os
import re


# 使用国内镜像加速下载
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import json
import torch
from datasets import load_dataset


GENRES = ("5", "7", "ci")
CI_TITLES = [
    "念奴娇",
    "水调歌头",
    "苏幕遮",
    "浣溪沙",
    "如梦令",
    "清平乐",
    "虞美人",
    "菩萨蛮",
    "钗头凤",
    "青玉案",
    "临江仙",
    "诉衷情",
    "一剪梅",
    "长相思",
    "满江红",
    "贺新郎",
    "蝶恋花",
    "江城子",
    "八声甘州",
    "鹧鸪天",
]


def normalize_poem_text(text: str) -> str:
    text = str(text).strip()
    text = text.replace("\r", "")
    return text


def split_poem_lines(text: str) -> list[str]:
    if not text:
        return []
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    if len(lines) > 1:
        return lines
    segments = re.split(r"[，。；？！、]+", text)
    return [seg.strip() for seg in segments if seg.strip()]


def count_chinese_chars(text: str) -> int:
    return len(re.findall(r"[\u4e00-\u9fff]", text))


def classify_poem(text: str) -> str:
    text = normalize_poem_text(text)
    lines = split_poem_lines(text)
    if not lines:
        return "ci"
    if any(title in text for title in CI_TITLES):
        return "ci"
    counts = [count_chinese_chars(line) for line in lines]
    counts = [c for c in counts if c > 0]
    if not counts:
        return "ci"
    if all(c == 5 for c in counts):
        return "5"
    if all(c == 7 for c in counts):
        return "7"
    return "ci"


def write_text_file(path: str, poems: list[str]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n\n".join(poems))


def encode_and_save(text: str, stoi: dict, out_path: str) -> None:
    ids = [stoi[c] for c in text]
    data = torch.tensor(ids, dtype=torch.long)
    torch.save(data, out_path)


def main():
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
    print("步骤 2：随机抽样最多 100000 条并按体裁分类")
    print("=" * 50)
    n_samples = 100000
    actual_samples = min(n_samples, len(dataset))
    subset = dataset.select(range(actual_samples))

    poems_by_genre = {g: [] for g in GENRES}
    all_texts = []
    for item in subset:
        text = item[text_col]
        if text is None:
            continue
        text = normalize_poem_text(text)
        if not text:
            continue
        genre = classify_poem(text)
        poems_by_genre[genre].append(text)
        all_texts.append(text)

    all_text = "\n\n".join(all_texts)
    with open("poetry.txt", "w", encoding="utf-8") as f:
        f.write(all_text)

    print(f"实际抽样条数: {len(all_texts)}")
    print(f"已保存 poetry.txt，总字符数约: {len(all_text)}")

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
    main()
