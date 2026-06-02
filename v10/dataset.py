# -*- coding: utf-8 -*-
"""
从 train_data.pt / val_data.pt 长序列中随机取连续块，构造 (x, y) 下一字预测任务。
兼容三种模式：
1. 普通古诗/词训练：仅返回 (x, y)
2. 主题训练：返回 (x, y, topic_id)
3. 重复项惩罚相关辅助信息：可预留 sample metadata，不影响旧接口
"""
from __future__ import annotations

import json
import os
import random
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset


def load_vocab_json(path: str) -> Tuple[Dict[str, int], Dict[int, str], int]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    stoi: Dict[str, int] = data["stoi"]
    itos: Dict[int, str] = {}
    for k, v in data["itos"].items():
        itos[int(k) if isinstance(k, str) else k] = v
    vs = int(data.get("vocab_size", len(itos)))
    return stoi, itos, vs


def load_topic_vocab(path: str) -> Tuple[List[str], int]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    topics = data["topics"]
    num_topics = int(data["num_topics"])
    return topics, num_topics


class PoetryBlockDataset(Dataset):
    """支持普通样本与主题样本的滑窗数据集。"""

    def __init__(
        self,
        data_path: str,
        block_size: int,
        topics_path: Optional[str] = None,
        boundaries_path: Optional[str] = None,
        num_samples: Optional[int] = None,
        sample_random: bool = True,
    ) -> None:
        if not os.path.isfile(data_path):
            raise FileNotFoundError(data_path)
        self.data = torch.load(data_path, map_location="cpu")
        if self.data.dim() != 1:
            raise ValueError("期望一维 long 张量")
        self.block_size = int(block_size)
        n = int(self.data.size(0))
        if n < self.block_size + 1:
            raise ValueError(f"序列太短: {n}，需 > block_size+1")
        self._max_i = n - self.block_size
        if num_samples is not None and int(num_samples) > 0:
            self._len = min(int(num_samples), self._max_i)
        else:
            self._len = int(self._max_i)
        self._sample_random = (
            bool(sample_random)
            and (num_samples is not None and int(num_samples) > 0)
            and (self._len < self._max_i)
        )

        self.has_topic = False
        self.token_to_topic = None
        if topics_path and boundaries_path and os.path.isfile(topics_path) and os.path.isfile(boundaries_path):
            topics = torch.load(topics_path, map_location="cpu")
            boundaries = torch.load(boundaries_path, map_location="cpu")
            self.token_to_topic = torch.zeros(len(self.data), dtype=torch.long)
            for poem_idx, start_pos in enumerate(boundaries):
                end_pos = boundaries[poem_idx + 1].item() if poem_idx + 1 < len(boundaries) else len(self.data)
                if poem_idx < len(topics):
                    self.token_to_topic[start_pos:end_pos] = topics[poem_idx]
            self.has_topic = True
        elif topics_path or boundaries_path:
            print("警告: 主题或边界文件缺失，将不使用主题信息")

    def __len__(self) -> int:
        return int(self._len)

    def __getitem__(self, idx: int):
        if self._sample_random:
            i = random.randrange(0, self._max_i)
        else:
            i = int(idx) % int(self._max_i)
        x = self.data[i : i + self.block_size].clone()
        y = self.data[i + 1 : i + 1 + self.block_size].clone()
        if self.has_topic and self.token_to_topic is not None:
            mid_pos = i + self.block_size // 2
            topic_id = int(self.token_to_topic[mid_pos].item())
            return x, y, topic_id
        return x, y
