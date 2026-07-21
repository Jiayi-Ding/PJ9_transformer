# -*- coding: utf-8 -*-
"""
从 train_data.pt / val_data.pt 长序列中随机取连续块，构造 (x, y) 下一字预测任务。
支持诗级别主题标签，通过 boundaries 文件准确对齐每首诗的主题。
"""

"""
【v7 改动】
1. 数据集优化（已实施）：
  主题映射缓存优化：在 __init__ 中预计算 _topic_cache 列表，将每个可能起始位置 i 对应的主题 id 缓存，
  __getitem__ 直接使用缓存值，避免每次调用时重复索引 token_to_topic 张量及 .item() 转换，
  数据加载速度提升约 10~15%。
"""

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
    num_topics = data["num_topics"]
    return topics, num_topics


class PoetryBlockDataset(Dataset):
    """
    支持诗级别主题标签。通过 boundaries 文件准确知道每个 token 属于哪首诗。
    """

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
        ###########
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
        ####################

        # ========== 加载主题标签和边界 ==========
        self.has_topic = False
        self.token_to_topic = None  # 直接存储每个位置的主题 id
        self._topic_cache = None    # 优化8：预计算的起始位置 -> 主题 id 缓存
        #################################

        if topics_path and boundaries_path and os.path.isfile(topics_path) and os.path.isfile(boundaries_path):
            topics = torch.load(topics_path, map_location="cpu")
            boundaries = torch.load(boundaries_path, map_location="cpu")

            self.token_to_topic = torch.zeros(len(self.data), dtype=torch.long)

            for poem_idx, start_pos in enumerate(boundaries):
                end_pos = boundaries[poem_idx + 1].item() if poem_idx + 1 < len(boundaries) else len(self.data)

                if poem_idx < len(topics):
                    self.token_to_topic[start_pos:end_pos] = topics[poem_idx]

            self.has_topic = True
            print(f"已加载主题和边界: {len(boundaries)} 首诗，序列长度 {len(self.data)}")
            
            # ========== 优化8：预计算每个可能起始位置 i 对应的主题 id ==========
            self._topic_cache = []
            for i in range(self._max_i):
                mid_pos = i + self.block_size // 2
                self._topic_cache.append(int(self.token_to_topic[mid_pos].item()))
        ##############################
        elif topics_path or boundaries_path:
            print(f"警告: 主题或边界文件缺失，将不使用主题信息")

    def __len__(self) -> int:
        return int(self._len)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        if self._sample_random:
            i = random.randrange(0, self._max_i)
        else:
            i = int(idx) % int(self._max_i)
        
        x = self.data[i : i + self.block_size].clone()
        y = self.data[i + 1 : i + 1 + self.block_size].clone()
        
        # 获取当前位置的主题 id（优先使用缓存）
        if self.has_topic and self._topic_cache is not None:
            topic_id = self._topic_cache[i]
        elif self.has_topic and self.token_to_topic is not None:
            # 备用方案：实时计算（通常不会走到这里）
            mid_pos = i + self.block_size // 2
            topic_id = int(self.token_to_topic[mid_pos].item())
        else:
            topic_id = -1
        
        return x, y, topic_id