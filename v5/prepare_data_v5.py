#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
古诗数据集下载与预处理脚本
- 从 Hugging Face 下载真实唐宋古诗数据
- 构建字符级词表并保存 vocab.json
- 划分 9:1 训练/验证集，编码后保存为 train_data_*.pt / val_data_*.pt
- 改进：为每首诗提取主题标签（边塞/山水/离别/其他），保存主题映射文件

【v5 主题 embedding 改动 1/4】
修改位置：
1. 新增 TOPIC_KEYWORDS 主题关键词库
2. 新增 classify_topic() 主题分类函数
3. 新增 save_topic_mapping() 保存主题标签
4. main() 中调用主题分类并保存 topics_train_{genre}.pt / topics_val_{genre}.pt
5. 新增 topic_vocab.json 保存主题词表

优点：每首诗一个主题标签，训练时整首诗共享同一主题 embedding
"""

import os
import re
from collections import Counter

# 使用国内镜像加速下载
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import json
import sys
import torch
from datasets import load_dataset

try:
    from opencc import OpenCC
except ImportError:
    print("请先安装 opencc：pip install opencc-python-reimplemented")
    sys.exit(1)

cc = OpenCC('t2s')  # 繁体到简体

GENRES = ("5", "7", "ci")

# 方案 2：按类别目标数量补齐
TARGET_PER_GENRE = {"5": 100000, "7": 100000, "ci": 100000}
MAX_PASS = 5
PASS_SAMPLE_SIZE = 100000

# ========== 新增：主题关键词库 ==========
TOPIC_KEYWORDS = {
    "frontier": [  # 边塞
        "边", "塞", "关", "征", "戍", "烽火", "狼烟", "铁马", "金戈", "胡", "羌",
        "玉门", "阳关", "凉州", "陇西", "雁门", "剑", "弓", "马", "战", "军",
        "沙场", "战场", "敌", "虏", "匈奴", "单于", "戈", "戟"
    ],
    "landscape": [  # 山水田园
        "山", "水", "江", "河", "湖", "海", "峰", "岭", "泉", "溪", "林", "竹",
        "松", "梅", "兰", "菊", "桃", "柳", "花", "草", "鸟", "鱼", "鹤", "鹿",
        "云", "雾", "雨", "雪", "风", "月", "日", "星", "天", "地", "田", "园",
        "村", "野", "寺", "庙", "道", "僧", "青", "绿", "碧", "翠"
    ],
    "farewell": [  # 离别
        "别", "离", "送", "辞", "去", "行", "远", "孤", "客", "游", "旅", "泊",
        "舟", "帆", "驿", "桥", "亭", "柳", "酒", "泪", "愁", "恨", "怨", "悲",
        "忆", "念", "思", "怀", "望", "盼", "归", "家", "乡", "故", "友", "君"
    ],
}

# 顺序固定，用于 id 映射（训练时 NUM_TOPICS 会从这里读取）
TOPIC_LIST = ["frontier", "landscape", "farewell", "other"]
NUM_TOPICS = len(TOPIC_LIST)


def classify_topic(text: str) -> str:
    """
    根据关键词匹配，判断诗歌主题。
    返回: frontier / landscape / farewell / other
    """
    # 统计各主题关键词出现次数
    scores = {topic: 0 for topic in TOPIC_LIST[:-1]}  # other 不参与计数
    for topic, keywords in TOPIC_KEYWORDS.items():
        for kw in keywords:
            # 简单匹配：关键词出现在文本中
            if kw in text:
                scores[topic] += 1
    
    # 找出得分最高的主题（且得分>0），否则返回 other
    max_topic = max(scores, key=scores.get)
    if scores[max_topic] > 0:
        return max_topic
    return "other"


def normalize_poem_text(text: str) -> str:
    text = str(text).strip()
    text = text.replace("\r", "")
    # 繁体转简体
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
    """
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
            text = normalize_poem_text(text)
            if not text or text in seen_texts:
                continue
            seen_texts.add(text)
            genre = classify_poem(text)
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

    # 新增：保存主题词表
    print("\n" + "=" * 50)
    print("步骤 3.5：保存主题词表 topic_vocab.json")
    print("=" * 50)
    topic_vocab = {
        "topics": TOPIC_LIST,
        "num_topics": NUM_TOPICS,
        "topic_keywords": TOPIC_KEYWORDS  # 保存关键词供参考
    }
    with open("topic_vocab.json", "w", encoding="utf-8") as f:
        json.dump(topic_vocab, f, ensure_ascii=False, indent=2)
    print(f"已保存主题词表: topic_vocab.json (主题数: {NUM_TOPICS}, 主题: {TOPIC_LIST})")

    print("\n" + "=" * 50)
    print("步骤 4：按体裁划分 9:1 训练/验证集并保存 .pt 文件（含主题标签）")
    print("=" * 50)
    
    for genre in GENRES:
        poems = poems_by_genre[genre]
        print(f"体裁 {genre}：{len(poems)} 首")
        if not poems:
            print(f"  跳过体裁 {genre}，未找到数据。")
            continue
        
        # 保存纯文本
        write_text_file(f"poetry_{genre}.txt", poems)

        # 划分训练/验证集
        n_train = int(len(poems) * 0.9)
        train_poems = poems[:n_train]
        val_poems = poems[n_train:]

        train_text = "\n\n".join(train_poems)
        val_text = "\n\n".join(val_poems)

        # 编码并保存字符序列
        encode_and_save(train_text, stoi, f"train_data_{genre}.pt")
        encode_and_save(val_text, stoi, f"val_data_{genre}.pt")

        print(f"  保存: train_data_{genre}.pt, val_data_{genre}.pt")
        print(f"  训练集: {len(train_poems)} 首, token 数: {len(train_text)}")
        print(f"  验证集: {len(val_poems)} 首, token 数: {len(val_text)}")
        
        # ========== 新增：生成并保存主题标签 ==========
        # 为每首诗计算主题标签（诗级别）
        train_topics = []
        for poem in train_poems:
            topic = classify_topic(poem)
            topic_id = TOPIC_LIST.index(topic)
            train_topics.append(topic_id)
        
        val_topics = []
        for poem in val_poems:
            topic = classify_topic(poem)
            topic_id = TOPIC_LIST.index(topic)
            val_topics.append(topic_id)
        
        # 保存为 .pt 文件（训练时 DataLoader 会按"首"读取）
        train_topics_tensor = torch.tensor(train_topics, dtype=torch.long)
        val_topics_tensor = torch.tensor(val_topics, dtype=torch.long)
        
        torch.save(train_topics_tensor, f"topics_train_{genre}.pt")
        torch.save(val_topics_tensor, f"topics_val_{genre}.pt")
        
        # 打印主题分布统计
        train_topic_names = [TOPIC_LIST[tid] for tid in train_topics]
        train_dist = Counter(train_topic_names)
        val_topic_names = [TOPIC_LIST[tid] for tid in val_topics]
        val_dist = Counter(val_topic_names)
        
        print(f"  已保存主题标签: topics_train_{genre}.pt, topics_val_{genre}.pt")
        print(f"  训练集主题分布: {dict(train_dist)}")
        print(f"  验证集主题分布: {dict(val_dist)}")

    print("\n预处理全部完成。")
    print("\n提示：已生成主题标签文件，训练时请使用 --use_topic 参数启用主题 embedding")


if __name__ == "__main__":
    main()