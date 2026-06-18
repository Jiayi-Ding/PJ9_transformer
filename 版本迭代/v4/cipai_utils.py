# -*- coding: utf-8 -*-
"""词牌数据处理公共工具。"""
from __future__ import annotations

import json
import os
import re
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Optional, Tuple

try:
    from opencc import OpenCC

    _cc = OpenCC("t2s")
except ImportError:
    _cc = None

LOCAL_CI_DIRS = ("ci_json", "data/ci_json")
CI_JSON_BASE = (
    "https://cdn.jsdelivr.net/gh/chinese-poetry/chinese-poetry@master/%E5%AE%8B%E8%AF%8D/"
)

# 常见词牌 -> 文件前缀（拼音）
CIPAI_TO_NAME: Dict[str, str] = {
    "浣溪沙": "huanxisha",
    "水调歌头": "shuidiaogetou",
    "鹧鸪天": "zhegutian",
    "菩萨蛮": "pusaman",
    "满江红": "manjianghong",
    "西江月": "xijiangyue",
    "临江仙": "linjiangxian",
    "念奴娇": "niannujiao",
    "减字木兰花": "jianzimulanhua",
    "沁园春": "qinyuanchun",
    "蝶恋花": "dielianhua",
    "失调名": "shidiaoming",
    "贺新郎": "hexinlang",
    "清平乐": "qingpingle",
    "满庭芳": "mantingfang",
    "虞美人": "yumeiren",
    "水龙吟": "shuilongyin",
    "好事近": "haoshijin",
    "渔家傲": "yujiaao",
    "卜算子": "busuanzi",
    "如梦令": "rumengling",
    "浪淘沙": "langtaosha",
    "忆秦娥": "yiqine",
    "南乡子": "nanxiangzi",
    "生查子": "shengchazi",
    "醉花阴": "zuihuayin",
    "定风波": "dingfengbo",
    "江城子": "jiangchengzi",
    "青玉案": "qingyuan",
    "摸鱼儿": "moeryu",
}

NAME_TO_CIPAI = {v: k for k, v in CIPAI_TO_NAME.items()}


def normalize(text: str) -> str:
    text = str(text).strip().replace("\r", "")
    if _cc is not None:
        text = _cc.convert(text)
    return text


def slugify_cipai(name: str) -> str:
    if name in CIPAI_TO_NAME:
        return CIPAI_TO_NAME[name]
    s = re.sub(r"[^\w\u4e00-\u9fff]+", "", name)
    if re.search(r"[\u4e00-\u9fff]", s):
        return f"cipai_{abs(hash(name)) % 100000:05d}"
    return s.lower() or "unknown"


def count_chinese_chars(text: str) -> int:
    return len(re.findall(r"[\u4e00-\u9fff]", text))


def split_poem_lines(text: str) -> List[str]:
    text = normalize(text)
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    if len(lines) > 1:
        return lines
    segments = re.split(r"[，。；？！、]+", text)
    return [seg.strip() for seg in segments if seg.strip()]


def poem_signature(text: str) -> Tuple[int, ...]:
    """句长序列，用于合并结构相近的小词牌。"""
    lines = split_poem_lines(text)
    if not lines:
        return ()
    return tuple(count_chinese_chars(ln) for ln in lines)


def dominant_signature(poems: Iterable[str]) -> Tuple[int, ...]:
    sigs = [poem_signature(p) for p in poems if p.strip()]
    sigs = [s for s in sigs if s]
    if not sigs:
        return ()
    return Counter(sigs).most_common(1)[0][0]


def signature_bucket(sig: Tuple[int, ...]) -> str:
    """把句长模式粗分到若干桶，便于合并。"""
    if not sig:
        return "unknown"
    avg = sum(sig) / len(sig)
    if len(sig) <= 6 and avg <= 5:
        return "short"
    if len(sig) <= 10 and avg <= 6.5:
        return "medium"
    if avg >= 7:
        return "long"
    return "mixed"


def poem_to_text(item: dict) -> str:
    paragraphs = item.get("paragraphs") or []
    lines = [normalize(p) for p in paragraphs if str(p).strip()]
    return "\n".join(lines)


def load_local_ci_json() -> List[dict]:
    poems: List[dict] = []
    for folder in LOCAL_CI_DIRS:
        if not os.path.isdir(folder):
            continue
        for fn in sorted(os.listdir(folder)):
            if not fn.startswith("ci.song") or not fn.endswith(".json"):
                continue
            path = os.path.join(folder, fn)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    batch = json.load(f)
            except Exception:
                continue
            if isinstance(batch, list):
                poems.extend(batch)
    return poems


def load_all_ci_json(use_remote: bool = False) -> List[dict]:
    poems = load_local_ci_json()
    if poems or not use_remote:
        return poems
    for i in range(30):
        fn = f"ci.song.{i}.json"
        url = CI_JSON_BASE + urllib.parse.quote(fn)
        try:
            with urllib.request.urlopen(url, timeout=20) as resp:
                batch = json.loads(resp.read().decode("utf-8"))
            if isinstance(batch, list):
                poems.extend(batch)
        except Exception:
            continue
    return poems


def count_by_cipai(poems: List[dict]) -> Counter:
    c = Counter()
    for item in poems:
        name = normalize(item.get("rhythmic", ""))
        if name:
            c[name] += 1
    return c


def poems_by_cipai(poems: List[dict]) -> Dict[str, List[str]]:
    grouped: Dict[str, List[str]] = defaultdict(list)
    for item in poems:
        name = normalize(item.get("rhythmic", ""))
        if not name:
            continue
        text = poem_to_text(item)
        if text:
            grouped[name].append(text)
    return dict(grouped)


def poetry_paths(name: str) -> dict:
    return {
        "txt": f"{name}.txt",
        "train": f"{name}_train.pt",
        "val": f"{name}_val.pt",
        "vocab": f"{name}_vocab.json",
        "model": f"{name}_model.pt",
    }
