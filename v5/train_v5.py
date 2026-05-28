# -*- coding: utf-8 -*-
"""
字符级 Transformer 语言模型训练。
依赖：同目录下已运行 prepare_data.py 生成数据；model.py、dataset.py。

【v5 主题 embedding 改动 4/4】
修改位置：
1. 新增命令行参数: --use_topic（是否启用主题 embedding）
2. 新增命令行参数: --topic_vocab（主题词表路径）
3. 加载训练/验证集时，同时加载对应的主题标签文件
4. 修改 DataLoader 的 collate_fn 或直接处理 DataSet 返回的 (x, y, topic_id)
5. 训练循环中将 topic_ids 传给 model.forward()
6. 保存 checkpoint 时增加 num_topics 字段
"""
import argparse
import os
import random
import sys
from typing import List, Optional

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import PoetryBlockDataset, load_vocab_json, load_topic_vocab
from model import CharGPT

# =============================================================================
# 默认可调超参：优先在此修改。下方 argparse 的 default 会引用这些常量。
# =============================================================================

# ---- 数据与文件路径 ----
TRAIN_PT = "train_data.pt"
VAL_PT = "val_data.pt"
VOCAB = "vocab.json"
TOPIC_VOCAB = "topic_vocab.json"      # 新增：主题词表
CKPT_OUT = "ckpt_best.pt"
LOSS_PLOT = "loss_curve.png"

# ---- 模型结构 ----
BLOCK_SIZE = 128
D_MODEL = 256
N_HEAD = 8
N_LAYER = 4
D_FF = 1024
DROPOUT = 0.1

# ---- 训练过程 ----
BATCH_SIZE = 32
LR = 3e-4
EPOCHS = 3
SEED = 42
VAL_BATCHES = 200
TRAIN_NUM_SAMPLES = 50_000
MAX_TRAIN_BATCHES = 0
LOG_INTERVAL = 50

# ---- 优化与数值稳定 ----
WEIGHT_DECAY = 0.01
GRAD_CLIP_NORM = 1.0

# ---- DataLoader 行为 ----
DATALOADER_NUM_WORKERS = 0
DATALOADER_DROP_LAST = True

# =============================================================================


def set_seed(s: int) -> None:
    random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def collate_fn_with_topic(batch):
    """
    自定义 collate 函数，处理 (x, y, topic_id) 三元组
    """
    x_list, y_list, topic_list = zip(*batch)
    x = torch.stack(x_list, dim=0)
    y = torch.stack(y_list, dim=0)
    topic_ids = torch.tensor(topic_list, dtype=torch.long)
    return x, y, topic_ids


def collate_fn_without_topic(batch):
    """
    兼容无主题的情况
    """
    # 如果 batch 是三元组，取前两个
    if len(batch[0]) == 3:
        x_list, y_list, _ = zip(*batch)
    else:
        x_list, y_list = zip(*batch)
    x = torch.stack(x_list, dim=0)
    y = torch.stack(y_list, dim=0)
    return x, y, None


@torch.no_grad()
def eval_loss(
    model: