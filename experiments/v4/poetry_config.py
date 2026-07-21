# -*- coding: utf-8 -*-
"""
词牌训练统一配置。
推荐流程：analyze_cipai.py -> split_cipai_plan.py -> prepare_cipai_batch.py -> train_cipai_batch.py
"""
from __future__ import annotations

import json
import os

from cipai_utils import CIPAI_TO_NAME, NAME_TO_CIPAI, slugify_cipai

# 单独训练阈值（与 analyze_cipai.py 默认一致）
SOLO_THRESHOLD = 350

# 当前默认词牌（手动切换时用）
POETRY_NAME = "huanxisha"
POETRY_LABEL = "浣溪沙"

CIPAI_PLAN_PATH = "cipai_plan.json"


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
    if key in NAME_TO_CIPAI:
        return NAME_TO_CIPAI[key]
    if os.path.isfile(CIPAI_PLAN_PATH):
        try:
            with open(CIPAI_PLAN_PATH, "r", encoding="utf-8") as f:
                plan = json.load(f)
            for x in plan.get("solo", []):
                if x.get("poetry_name") == key:
                    return x.get("cipai", key)
            for g in plan.get("merge_groups", []):
                if g.get("poetry_name") == key:
                    return g.get("label", key)
        except Exception:
            pass
    return POETRY_LABEL


def list_trainable_from_plan(plan_path: str = CIPAI_PLAN_PATH) -> list[dict]:
    if not os.path.isfile(plan_path):
        return []
    with open(plan_path, "r", encoding="utf-8") as f:
        plan = json.load(f)
    out = []
    for x in plan.get("solo", []):
        out.append({"poetry_name": x["poetry_name"], "label": x["cipai"], "tier": "solo"})
    for g in plan.get("merge_groups", []):
        out.append({"poetry_name": g["poetry_name"], "label": g.get("label", g["poetry_name"]), "tier": "merge"})
    return out
