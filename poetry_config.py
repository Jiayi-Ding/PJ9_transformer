# -*- coding: utf-8 -*-
"""
词牌训练统一配置。
切换词牌时只需修改 POETRY_NAME / POETRY_LABEL，或运行 split_cipai.py 生成对应 txt。
"""

# 当前词牌文件前缀（用于自动生成 train / val / vocab / model 文件名）
POETRY_NAME = "rumengling"

# 当前词牌中文名（用于日志展示）
POETRY_LABEL = "如梦令"

# 词牌中文名 -> 文件前缀（扩展新词牌时在此追加）
CIPAI_TO_NAME = {
    "如梦令": "rumengling",
    "水调歌头": "shuidiaogetou",
    "沁园春": "qinyuanchun",
}

NAME_TO_CIPAI = {v: k for k, v in CIPAI_TO_NAME.items()}


def poetry_txt(name: str | None = None) -> str:
    return f"{name or POETRY_NAME}.txt"


def train_pt(name: str | None = None) -> str:
    return f"{name or POETRY_NAME}_train.pt"


def val_pt(name: str | None = None) -> str:
    return f"{name or POETRY_NAME}_val.pt"


def vocab_json(name: str | None = None) -> str:
    return f"{name or POETRY_NAME}_vocab.json"


def model_pt(name: str | None = None) -> str:
    return f"{name or POETRY_NAME}_model.pt"


def poetry_label(name: str | None = None) -> str:
    key = name or POETRY_NAME
    return NAME_TO_CIPAI.get(key, POETRY_LABEL)
