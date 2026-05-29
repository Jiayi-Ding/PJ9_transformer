#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从 chinese-poetry 宋词 JSON 按词牌名（rhythmic）拆分，生成 {poetry_name}.txt。
示例：python split_cipai.py --cipai 如梦令
      python split_cipai.py --all
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.parse
import urllib.request

try:
    from opencc import OpenCC

    cc = OpenCC("t2s")
except ImportError:
    cc = None

from poetry_config import CIPAI_TO_NAME, NAME_TO_CIPAI

CI_JSON_BASE = (
    "https://cdn.jsdelivr.net/gh/chinese-poetry/chinese-poetry@master/%E5%AE%8B%E8%AF%8D/"
)
LOCAL_CI_DIRS = ("ci_json", "data/ci_json", os.path.join("..", "chinese-poetry", "宋词"))

# 常见词牌按句长模式匹配（JSON 缺失时的兜底，仅作补充）
CIPAI_LINE_PATTERNS: dict[str, tuple[int, ...]] = {
    "如梦令": (7, 4, 7, 4, 7, 4, 7, 4),
    "水调歌头": (5, 5, 7, 5, 5, 7, 5, 5, 7, 5, 5, 7, 5, 5, 7, 5, 5, 7, 5, 5, 7),
    "沁园春": (4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4),
}


def count_chinese_chars(text: str) -> int:
    import re

    return len(re.findall(r"[\u4e00-\u9fff]", text))


def split_poem_lines(text: str) -> list[str]:
    import re

    text = normalize(text)
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    if len(lines) > 1:
        return lines
    segments = re.split(r"[，。；？！、]+", text)
    return [seg.strip() for seg in segments if seg.strip()]


def match_line_pattern(text: str, pattern: tuple[int, ...]) -> bool:
    lines = split_poem_lines(text)
    if len(lines) != len(pattern):
        return False
    for line, need in zip(lines, pattern):
        if count_chinese_chars(line) != need:
            return False
    return True


def download_ci_json(out_dir: str = "ci_json") -> None:
    """首次使用时下载 chinese-poetry 宋词 JSON 到 ci_json/。"""
    import zipfile
    from pathlib import Path

    zip_path = Path("ci_json_repo.zip")
    url = "https://github.com/chinese-poetry/chinese-poetry/archive/refs/heads/master.zip"
    out = Path(out_dir)
    out.mkdir(exist_ok=True)
    if not zip_path.exists():
        print("正在下载 chinese-poetry 仓库（约 90MB）…")
        urllib.request.urlretrieve(url, zip_path)
    prefix = "chinese-poetry-master/\u5b8b\u8bcd/"
    with zipfile.ZipFile(zip_path) as zf:
        n = 0
        for name in zf.namelist():
            if name.startswith(prefix) and name.endswith(".json") and "ci.song" in name:
                fn = os.path.basename(name)
                (out / fn).write_bytes(zf.read(name))
                n += 1
    print(f"已解压 {n} 个 ci.song*.json 到 {out_dir}/")


def load_local_ci_json() -> list[dict]:
    poems: list[dict] = []
    for folder in LOCAL_CI_DIRS:
        if not os.path.isdir(folder):
            continue
        for fn in sorted(os.listdir(folder)):
            if not fn.endswith(".json"):
                continue
            if not fn.startswith("ci.song"):
                continue
            path = os.path.join(folder, fn)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    batch = json.load(f)
            except Exception:
                continue
            if isinstance(batch, list):
                poems.extend(batch)
                print(f"  已加载本地 {path}：{len(batch)} 首")
    return poems


def normalize(text: str) -> str:
    text = str(text).strip()
    if cc is not None:
        text = cc.convert(text)
    return text


def poem_to_text(item: dict) -> str:
    paragraphs = item.get("paragraphs") or []
    lines = [normalize(p) for p in paragraphs if str(p).strip()]
    return "\n".join(lines)


def load_all_ci_json() -> list[dict]:
    poems: list[dict] = []
    poems.extend(load_local_ci_json())
    for i in range(0, 30):
        fn = f"ci.song.{i}.json"
        url = CI_JSON_BASE + urllib.parse.quote(fn)
        try:
            with urllib.request.urlopen(url, timeout=30) as resp:
                batch = json.loads(resp.read().decode("utf-8"))
        except Exception:
            continue
        if not isinstance(batch, list):
            continue
        poems.extend(batch)
        print(f"  已加载远程 {fn}：{len(batch)} 首")
    return poems


def fallback_from_poetry_ci(cipai: str) -> list[str]:
    src = "poetry_ci.txt"
    if not os.path.isfile(src):
        return []
    pattern = CIPAI_LINE_PATTERNS.get(cipai)
    if pattern is None:
        return []
    with open(src, "r", encoding="utf-8") as f:
        raw = f.read()
    poems = [normalize(p) for p in raw.split("\n\n") if p.strip()]
    matched = [p for p in poems if match_line_pattern(p, pattern)]
    if matched:
        print(f"  从 {src} 按句长模式匹配「{cipai}」：{len(matched)} 首")
    return matched


def split_one(cipai: str, poetry_name: str, poems: list[dict], out_dir: str) -> int:
    matched = [p for p in poems if normalize(p.get("rhythmic", "")) == cipai]
    texts = [poem_to_text(p) for p in matched]
    texts = [t for t in texts if t]
    if not texts:
        texts = fallback_from_poetry_ci(cipai)
    out_path = os.path.join(out_dir, f"{poetry_name}.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n\n".join(texts))
    print(f"词牌「{cipai}」-> {out_path}，共 {len(texts)} 首")
    return len(texts)


def main() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    os.chdir(here)

    p = argparse.ArgumentParser(description="按词牌拆分宋词 JSON 为独立 txt")
    p.add_argument("--cipai", type=str, default=None, help="词牌中文名，如 如梦令")
    p.add_argument(
        "--poetry_name",
        type=str,
        default=None,
        help="输出文件前缀，默认从 CIPAI_TO_NAME 查找",
    )
    p.add_argument(
        "--all",
        action="store_true",
        help="拆分 CIPAI_TO_NAME 中全部词牌",
    )
    p.add_argument(
        "--download-ci-json",
        action="store_true",
        help="下载 chinese-poetry 宋词 JSON 到 ci_json/（首次拆分前建议执行）",
    )
    args = p.parse_args()

    if args.download_ci_json:
        download_ci_json()

    if args.poetry_name and not args.cipai:
        args.cipai = NAME_TO_CIPAI.get(args.poetry_name)
    elif not args.all and not args.cipai:
        args.cipai = NAME_TO_CIPAI.get("rumengling", "如梦令")

    print("正在下载/读取宋词 JSON …")
    poems = load_all_ci_json()
    if not poems and not os.path.isfile("poetry_ci.txt"):
        print("未获取到宋词数据，请检查网络或手动放置 ci.song.*.json 到 ci_json/", file=sys.stderr)
        sys.exit(1)
    if poems:
        print(f"宋词 JSON 总数：{len(poems)}")

    if args.all:
        for cipai, name in CIPAI_TO_NAME.items():
            split_one(cipai, name, poems, here)
        return

    cipai = args.cipai
    poetry_name = args.poetry_name or CIPAI_TO_NAME.get(cipai)
    if not poetry_name:
        print(f"未知词牌「{cipai}」，请在 poetry_config.py 的 CIPAI_TO_NAME 中注册。", file=sys.stderr)
        sys.exit(1)
    n = split_one(cipai, poetry_name, poems, here)
    if n == 0:
        print(f"警告：未找到词牌「{cipai}」的样本。", file=sys.stderr)


if __name__ == "__main__":
    main()
