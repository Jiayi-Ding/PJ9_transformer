#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
古诗数据集下载与预处理脚本。
保留：
1. 5 / 7 / ci 体裁划分
2. 词牌相关输出（兼容已有词牌体系）
3. 主题词表与主题标签
4. 重复项惩罚可使用的基础数据组织
"""
from __future__ import annotations

import json
import os
import re
import sys
from collections import Counter
from typing import Dict, List

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
from datasets import load_dataset

try:
    from opencc import OpenCC
except ImportError:
    print("请先安装 opencc：pip install opencc-python-reimplemented")
    sys.exit(1)

cc = OpenCC("t2s")

GENRES = ("5", "7", "ci")
TARGET_PER_GENRE = {"5": 100000, "7": 100000, "ci": 100000}
MAX_PASS = 5
PASS_SAMPLE_SIZE = 100000

TOPIC_KEYWORDS: Dict[str, List[str]] = {
    "landscape": ["山", "水", "江", "河", "湖", "海", "峰", "岭", "岳", "川", "泉", "溪", "涧", "潭", "林", "木", "竹", "松", "柏", "梅", "兰", "菊", "荷", "莲", "桃", "柳", "花", "草", "云", "霞", "烟", "岚", "雨", "雪", "风", "月", "日", "星"],
    "frontier": ["边", "塞", "关", "城", "戍", "征", "战", "军", "兵", "骑", "马", "弓", "箭", "刀", "剑", "戈", "旗", "鼓", "烽", "燧", "沙", "漠", "寒", "霜", "胡", "羌", "戎", "匈奴"],
    "homesickness": ["乡", "家", "故", "归", "客", "旅", "羁", "泊", "孤", "独", "夜", "秋", "暮", "雁", "书", "信", "梦", "魂", "泪", "愁", "思", "怀"],
    "historical": ["古", "今", "昔", "往", "旧", "王", "帝", "皇", "侯", "将", "相", "宫", "殿", "阙", "台", "楼", "陵", "墓", "碑", "史", "兴", "亡", "盛", "衰"],
    "love": ["心", "情", "思", "念", "爱", "怜", "惜", "红", "粉", "香", "艳", "娇", "鸳", "鸯", "眉", "眼", "唇", "镜", "钗", "簪", "裙", "袖", "泪", "怨"],
    "objects": ["梅", "兰", "竹", "菊", "松", "柏", "桂", "莲", "荷", "玉", "金", "银", "镜", "剑", "琴", "棋", "书", "画", "笔", "墨", "纸", "砚", "灯", "烛", "舟"],
    "farewell": ["别", "离", "辞", "送", "饯", "赠", "留", "饮", "酒", "杯", "帆", "舟", "柳", "花", "春", "秋", "泪", "悲", "叹", "忆", "念", "怀"],
    "time_sorrow": ["时", "光", "阴", "岁", "年", "春", "夏", "秋", "冬", "朝", "暮", "夜", "晓", "晨", "老", "病", "荣", "枯", "盛", "衰", "落", "谢", "逝", "流", "悲", "哀"],
    "festival": ["春", "夏", "秋", "冬", "年", "岁", "节", "元", "旦", "元宵", "清明", "端午", "中秋", "重阳", "除夕", "灯", "月", "饼", "粽", "酒", "欢", "喜", "庆", "赏"],
    "palace_grievance": ["宫", "殿", "阙", "楼", "台", "苑", "园", "妆", "镜", "钗", "簪", "裙", "袖", "床", "帐", "枕", "帘", "窗", "灯", "烛", "泪", "怨", "孤", "寂"],
    "immortal": ["仙", "神", "道", "丹", "鼎", "药", "云", "霞", "霓", "虹", "天", "地", "山", "海", "洞", "宫", "龙", "凤", "鹤", "芝", "桃", "露", "泉"],
    "zen": ["空", "无", "有", "真", "假", "虚", "幻", "梦", "影", "相", "心", "性", "道", "禅", "定", "慧", "静", "净", "清", "明", "寺", "庙", "僧", "佛"],
    "drinking": ["酒", "醉", "饮", "酌", "杯", "樽", "觞", "歌", "舞", "笑", "豪", "狂", "放", "月", "花", "雪", "山", "水", "风", "愁", "乐"],
    "elegy": ["悼", "哭", "哀", "悲", "伤", "痛", "泣", "涕", "魂", "魄", "墓", "坟", "冢", "碑", "死", "亡", "逝", "空", "荒", "夜", "秋", "暮"],
    "imperial_exam": ["科", "举", "试", "考", "策", "论", "诗", "赋", "经", "书", "名", "榜", "第", "进士", "状元", "落第", "寒窗", "笔", "纸", "砚", "墨", "灯", "夜"],
    "war_atrocity": ["战", "乱", "兵", "杀", "戮", "焚", "烧", "逃", "亡", "流", "离", "饥", "饿", "尸", "骨", "血", "泪", "民", "众", "百姓", "村", "城", "荒"],
    "feminine_life": ["女", "妇", "妾", "娘", "娇", "媚", "妍", "丽", "艳", "红", "粉", "香", "镜", "妆", "梳", "髻", "眉", "唇", "钗", "簪", "裙", "袖", "绣"],
    "reclusion_tourism": ["隐", "逸", "闲", "静", "山", "水", "林", "泉", "石", "溪", "湖", "寺", "庵", "院", "鹤", "鹿", "花", "草", "竹", "松", "云", "月", "舟", "酒", "茶", "游", "赏"],
}

TOPIC_LIST = [
    "landscape", "frontier", "homesickness", "historical", "love", "objects", "farewell", "time_sorrow", "festival", "palace_grievance", "immortal", "zen", "drinking", "elegy", "imperial_exam", "war_atrocity", "feminine_life", "reclusion_tourism", "other",
]
NUM_TOPICS = len(TOPIC_LIST)


def classify_topic(text: str) -> str:
    scores = {topic: 0 for topic in TOPIC_LIST[:-1]}
    for topic, keywords in TOPIC_KEYWORDS.items():
        for kw in keywords:
            if kw in text:
                scores[topic] += 1
    max_topic = max(scores, key=scores.get)
    return max_topic if scores[max_topic] > 0 else "other"


def normalize_poem_text(text: str) -> str:
    text = str(text).strip().replace("\r", "")
    return cc.convert(text)


def is_valid_poem(text: str) -> bool:
    if not text or len(text.strip()) == 0:
        return False
    if "□" in text:
        return False
    lines = text.split("\n")
    total_chinese = 0
    for line in lines:
        line = line.strip()
        if not line:
            continue
        chinese_count = len(re.findall(r"[\u4e00-\u9fff]", line))
        total_chinese += chinese_count
        if chinese_count == 0:
            return False
    return total_chinese >= 10


def split_poem_lines(text: str) -> List[str]:
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
    counts = [count_chinese_chars(line) for line in lines]
    counts = [c for c in counts if c > 0]
    if not counts:
        return "ci"
    n5 = sum(1 for c in counts if c == 5)
    n7 = sum(1 for c in counts if c == 7)
    total = len(counts)
    if n5 / total >= 0.6:
        return "5"
    if n7 / total >= 0.6:
        return "7"
    return "ci"


def write_text_file(path: str, poems: List[str]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n\n".join(poems))


def encode_and_save(text: str, stoi: dict, out_path: str) -> None:
    ids = [stoi[c] for c in text]
    torch.save(torch.tensor(ids, dtype=torch.long), out_path)


def main() -> None:
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
    all_texts: List[str] = []
    seen_texts = set()
    targets = TARGET_PER_GENRE.copy()
    added = True
    filtered_count = 0
    for pass_i in range(MAX_PASS):
        if all(len(poems_by_genre[g]) >= targets[g] for g in GENRES):
            break
        if not added and pass_i > 0:
            break
        added = False
        shuffled = dataset.shuffle(seed=42 + pass_i)
        if len(shuffled) > PASS_SAMPLE_SIZE:
            shuffled = shuffled.select(range(PASS_SAMPLE_SIZE))
        for item in shuffled:
            text = item[text_col]
            if text is None:
                continue
            text = normalize_poem_text(text)
            if not text or text in seen_texts:
                continue
            if not is_valid_poem(text):
                filtered_count += 1
                continue
            seen_texts.add(text)
            genre = classify_poem(text)
            if len(poems_by_genre[genre]) < targets[genre]:
                poems_by_genre[genre].append(text)
                all_texts.append(text)
                added = True
            if all(len(poems_by_genre[g]) >= targets[g] for g in GENRES):
                break

    print(f"过滤掉的无效诗歌数: {filtered_count}")

    all_text = "\n\n".join(all_texts)
    with open("poetry.txt", "w", encoding="utf-8") as f:
        f.write(all_text)

    print(f"实际抽样条数: {len(all_texts)}")
    for genre in GENRES:
        print(f"  体裁 {genre}：{len(poems_by_genre[genre])} 条（目标 {targets[genre]}）")

    print("\n" + "=" * 50)
    print("步骤 3：构建字符级词表并保存 vocab.json")
    print("=" * 50)
    chars = sorted(list(set(all_text)))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    vocab = {"vocab_size": len(chars), "stoi": stoi, "itos": itos}
    with open("vocab.json", "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 50)
    print("步骤 3.5：保存主题词表 topic_vocab.json")
    print("=" * 50)
    topic_vocab = {"topics": TOPIC_LIST, "num_topics": NUM_TOPICS, "topic_keywords": TOPIC_KEYWORDS}
    with open("topic_vocab.json", "w", encoding="utf-8") as f:
        json.dump(topic_vocab, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 50)
    print("步骤 4：按体裁划分 9:1 训练/验证集并保存 .pt 文件（含主题标签）")
    print("=" * 50)
    for genre in GENRES:
        poems = poems_by_genre[genre]
        if not poems:
            print(f"  跳过体裁 {genre}，未找到数据。")
            continue
        write_text_file(f"poetry_{genre}.txt", poems)
        n_train = int(len(poems) * 0.9)
        train_poems = poems[:n_train]
        val_poems = poems[n_train:]
        train_text = "\n\n".join(train_poems)
        val_text = "\n\n".join(val_poems)

        train_boundaries = []
        pos = 0
        for poem in train_poems:
            train_boundaries.append(pos)
            pos += len(poem) + 2
        val_boundaries = []
        pos = 0
        for poem in val_poems:
            val_boundaries.append(pos)
            pos += len(poem) + 2

        encode_and_save(train_text, stoi, f"train_data_{genre}.pt")
        encode_and_save(val_text, stoi, f"val_data_{genre}.pt")

        train_topics = [TOPIC_LIST.index(classify_topic(poem)) for poem in train_poems]
        val_topics = [TOPIC_LIST.index(classify_topic(poem)) for poem in val_poems]

        torch.save(torch.tensor(train_topics, dtype=torch.long), f"topics_train_{genre}.pt")
        torch.save(torch.tensor(val_topics, dtype=torch.long), f"topics_val_{genre}.pt")
        torch.save(torch.tensor(train_boundaries, dtype=torch.long), f"boundaries_train_{genre}.pt")
        torch.save(torch.tensor(val_boundaries, dtype=torch.long), f"boundaries_val_{genre}.pt")

        print(f"  保存: train_data_{genre}.pt, val_data_{genre}.pt")
        print(f"  已保存主题标签: topics_train_{genre}.pt, topics_val_{genre}.pt")
        print(f"  已保存边界: boundaries_train_{genre}.pt, boundaries_val_{genre}.pt")
        print(f"  训练集主题分布: {dict(Counter(TOPIC_LIST[tid] for tid in train_topics))}")
        print(f"  验证集主题分布: {dict(Counter(TOPIC_LIST[tid] for tid in val_topics))}")

    print("\n预处理全部完成。")
    print("\n提示：训练时可使用 --use_topic 启用主题 embedding，--use_repeat_penalty 启用重复项惩罚")


if __name__ == "__main__":
    main()
