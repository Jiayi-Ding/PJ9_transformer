# -*- coding: utf-8 -*-
"""
统一词牌配置。
保留词牌划分能力，同时兼容通用古诗 / 主题风格 / 重复项惩罚版本。
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

POETRY_NAME = "huanxisha"

POETRY_LABELS: Dict[str, str] = {
    "huanxisha": "浣溪沙",
    "shuidiaogetou": "水调歌头",
    "zhegutian": "鹧鸪天",
    "pusaman": "菩萨蛮",
    "dielianhua": "蝶恋花",
    "merge_medium": "其他小词牌",
}


def poetry_label(name: str) -> str:
    return POETRY_LABELS.get(name, name)


def train_pt(name: str) -> str:
    return f"{name}_train.pt"


def val_pt(name: str) -> str:
    return f"{name}_val.pt"


def vocab_json(name: str) -> str:
    return f"{name}_vocab.json"


def model_pt(name: str) -> str:
    return f"{name}_model.pt"


def list_trainable_from_plan() -> List[dict]:
    plan_path = Path("cipai_plan.json")
    if not plan_path.is_file():
        return []
    import json

    with open(plan_path, "r", encoding="utf-8") as f:
        plan = json.load(f)
    items: List[dict] = []
    for x in plan.get("solo", []):
        items.append({"tier": "solo", "label": x.get("cipai", x.get("poetry_name", "")), "poetry_name": x.get("poetry_name", "")})
    for g in plan.get("merge_groups", []):
        items.append({"tier": "merge", "label": g.get("label", g.get("poetry_name", "")), "poetry_name": g.get("poetry_name", "")})
    return items
