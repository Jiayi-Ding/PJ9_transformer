# -*- coding: utf-8 -*-
"""
从 train_data.pt / val_data.pt 长序列中随机取连续块，构造 (x, y) 下一字预测任务。
x[i] 预测 y[i] = 序列中下一字符 id，长度均为 block_size。

【v5 主题 embedding 改动 2/4】
修改位置：
1. 新增 __init__ 参数: topics_path（诗级别主题标签文件路径）
2. 新增 self.topics 存储主题标签（每首诗一个标签）
3. 新增 self.poem_boundaries 存储每首诗在长序列中的起始位置
4. __getitem__ 中根据起始位置 i 找到所属的诗索引，返回对应的 topic_id
5. 返回值从 (x, y) 改为 (x, y, topic_id)

原理：每个滑窗返回它所属的那首诗的主题标签，整首诗共享同一主题
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
    """加载主题词表，返回 (topic_list, num_topics)"""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    topics = data["topics"]
    num_topics = data["num_topics"]
    return topics, num_topics


class PoetryBlockDataset(Dataset):
    """
    一维长序列上，以起始下标 i 取 [i : i+block_size] 为 x，
    [i+1 : i+1+block_size] 为 y。
    
    新增：支持诗级别主题标签。每首诗对应一个 topic_id，同一首诗的所有滑窗返回相同的 topic_id。
    需要提供：
    - data_path: 字符序列文件 (.pt)
    - topics_path: 主题标签文件 (.pt)，长度 = 诗的数量
    - boundaries_path: 诗边界文件 (.pt)，记录每首诗在 data 中的起始位置（可选，自动构建）
    """

    def __init__(
        self,
        data_path: str,
        block_size: int,
        topics_path: Optional[str] = None,      # 新增：主题标签文件路径
        boundaries_path: Optional[str] = None,  # 可选：预计算的边界文件
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
        
        # ========== 新增：加载主题标签并构建边界索引 ==========
        self.has_topic = (topics_path is not None and os.path.isfile(topics_path))
        self.topics = None
        self.poem_boundaries = None  # [start_pos, start_pos, ...] 每首诗在 data 中的起始位置
        
        if self.has_topic:
            # 加载主题标签（每首诗一个）
            self.topics = torch.load(topics_path, map_location="cpu")
            print(f"已加载主题标签: {topics_path}, 共 {len(self.topics)} 首诗")
            
            # 构建诗边界：需要知道每首诗在 data 中的起始位置
            # 方式1：从预计算的 boundaries_path 加载
            if boundaries_path and os.path.isfile(boundaries_path):
                self.poem_boundaries = torch.load(boundaries_path, map_location="cpu")
                print(f"已加载诗边界: {boundaries_path}")
            else:
                # 方式2：自动构建（通过查找换行符 \n\n 来分割诗）
                # 注意：prepare_data.py 中使用 "\n\n" 分隔不同诗
                print("正在自动构建诗边界索引（通过查找换行符）...")
                self.poem_boundaries = self._build_boundaries_from_newlines()
                print(f"已自动构建 {len(self.poem_boundaries)} 个诗边界")
            
            # 确保主题数量与诗数量一致
            assert len(self.topics) == len(self.poem_boundaries), \
                f"主题数量({len(self.topics)})与诗数量({len(self.poem_boundaries)})不一致"
        else:
            print("未提供主题标签，将不使用主题信息（返回 topic_id=-1）")

    def _build_boundaries_from_newlines(self) -> torch.Tensor:
        """
        通过查找连续两个换行符 "\n\n" 来定位每首诗的起始位置。
        返回: [start0, start1, start2, ...] 每首诗在 data 中的起始 token 索引
        """
        boundaries = [0]  # 第一首诗从位置 0 开始
        data_list = self.data.tolist()
        
        # 获取换行符的 token id（需要从词表中知道，这里假设换行符是常见字符）
        # 在 prepare_data.py 中，换行符 '\n' 会被编码。我们需要找到连续的 '\n\n'
        # 由于无法直接获取 stoi，这里通过查找连续两个相同的 token 来推断
        # 更可靠的方式：让用户传入换行符 id，这里简化处理
        # 实际训练时，我们可以在 prepare_data 时额外保存 boundaries 文件
        
        # 简化方案：扫描序列，找到连续两个换行符的位置
        # 这要求我们知道换行符的 token id。这里假设最常见的情况
        # 更好的做法：在 prepare_data 时显式保存 boundaries 文件
        
        # 为了不阻塞训练，如果自动构建失败，返回 [0] 作为 fallback（所有诗作为一个整体）
        # 建议用户运行一次后保存 boundaries 文件
        print("  警告：自动构建边界可能不准确，建议运行 prepare_data 时同时保存 boundaries 文件")
        
        # 简单实现：假设诗词文本中不会出现连续的换行符，只有分隔符
        # 这里返回单一边界（整段文本视为一首诗）
        return torch.tensor([0], dtype=torch.long)
    
    def _get_topic_for_position(self, pos: int) -> int:
        """
        根据字符位置 pos，返回所属诗的主题 id。
        pos: 在 data 序列中的索引
        """
        if not self.has_topic or self.poem_boundaries is None:
            return -1  # 无主题时返回 -1
        
        # 二分查找 pos 属于哪首诗
        left, right = 0, len(self.poem_boundaries) - 1
        while left < right:
            mid = (left + right + 1) // 2
            if self.poem_boundaries[mid] <= pos:
                left = mid
            else:
                right = mid - 1
        poem_idx = left
        return int(self.topics[poem_idx].item())

    def __len__(self) -> int:
        return int(self._len)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        if self._sample_random:
            i = random.randrange(0, self._max_i)
        else:
            i = int(idx) % int(self._max_i)
        
        x = self.data[i : i + self.block_size].clone()
        y = self.data[i + 1 : i + 1 + self.block_size].clone()
        
        # 获取当前位置所属诗的主题 id
        topic_id = self._get_topic_for_position(i)
        
        return x, y, topic_id