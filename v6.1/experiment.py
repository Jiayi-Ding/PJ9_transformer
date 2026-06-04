# -*- coding: utf-8 -*-
"""
v6.1 熵阈值实验 —— 批量生成脚本

5 组阈值参数 × 10 首诗，共生成 50 首五言诗。
固定条件：genre=5, prompt=春, temperature=0.8, stop_newline

输出结构：
  experiment_results/
  ├── poems/                  # 按组存放的原始诗作
  │   ├── G1_original/        # 第1组 (原始阈值)
  │   ├── G2_mild_stretch/    # 第2组 (微拉伸)
  │   ├── G3_mid_stretch/     # 第3组 (中拉伸)
  │   ├── G4_strong_stretch/  # 第4组 (强拉伸)
  │   └── G5_no_adaptive/     # 第5组 (对照：无自适应)
  ├── scoring_table.csv       # 匿名化评分表（随机排序，含 code 列）
  └── code_mapping.json       # code → 组别 的映射（评分结束后查看）
"""

import os
import sys
import json
import csv
import random
import subprocess
from datetime import datetime

# =============================================================================
# 实验配置
# =============================================================================

# 5 组阈值参数 (entropy_high, entropy_mid, entropy_low, 标签, 文件夹名)
GROUPS = [
    {   # 第1组：原始 v6 参数
        "id": "G1",
        "name": "原始阈值",
        "folder": "G1_original",
        "entropy_high": 2.5,
        "entropy_mid": 1.5,
        "entropy_low": 0.8,
        "use_adaptive": True,
    },
    {   # 第2组：微拉伸
        "id": "G2",
        "name": "微拉伸",
        "folder": "G2_mild_stretch",
        "entropy_high": 3.0,
        "entropy_mid": 1.8,
        "entropy_low": 0.5,
        "use_adaptive": True,
    },
    {   # 第3组：中拉伸
        "id": "G3",
        "name": "中拉伸",
        "folder": "G3_mid_stretch",
        "entropy_high": 3.5,
        "entropy_mid": 2.0,
        "entropy_low": 0.3,
        "use_adaptive": True,
    },
    {   # 第4组：强拉伸
        "id": "G4",
        "name": "强拉伸",
        "folder": "G4_strong_stretch",
        "entropy_high": 4.5,
        "entropy_mid": 2.5,
        "entropy_low": 0.1,
        "use_adaptive": True,
    },
    {   # 第5组：无自适应对照
        "id": "G5",
        "name": "无自适应对照",
        "folder": "G5_no_adaptive",
        "entropy_high": None,
        "entropy_mid": None,
        "entropy_low": None,
        "use_adaptive": False,
    },
]

# 固定参数
PROMPT = "春"
TEMPERATURE = 0.8
MAX_NEW = 200
POEMS_PER_GROUP = 10
SEED_BASE = 42  # 每组用不同 seed 保证可复现但组间可区分


def main():
    import argparse
    ap = argparse.ArgumentParser(description="v6.1 熵阈值批量实验")
    ap.add_argument("--genre", type=str, choices=["5", "7", "ci"], default="5",
                    help="体裁: 5(五言) / 7(七言) / ci(词)")
    args = ap.parse_args()
    genre = args.genre

    here = os.path.dirname(os.path.abspath(__file__))
    os.chdir(here)

    out_dir = os.path.join(here, "experiment_results", genre)
    poems_dir = os.path.join(out_dir, "poems")
    os.makedirs(poems_dir, exist_ok=True)

    all_poems = []  # [{code, group_id, group_name, poem_num, text, params}]

    genre_label = {"5": "五言", "7": "七言", "ci": "词"}.get(genre, genre)
    print("=" * 60)
    print("v6.1 熵阈值实验 —— 批量生成")
    print(f"体裁: {genre_label}  起笔字: {PROMPT}  温度: {TEMPERATURE}")
    print(f"共 {len(GROUPS)} 组 × {POEMS_PER_GROUP} 首 = {len(GROUPS) * POEMS_PER_GROUP} 首")
    print("=" * 60)

    for group in GROUPS:
        group_folder = os.path.join(poems_dir, group["folder"])
        os.makedirs(group_folder, exist_ok=True)

        print(f"\n{'─' * 40}")
        print(f"正在生成: [{group['id']}] {group['name']}")

        if group["use_adaptive"]:
            print(f"  阈值: high={group['entropy_high']}, mid={group['entropy_mid']}, low={group['entropy_low']}")
            normal_width = group["entropy_high"] - group["entropy_low"]
            print(f"  正常区宽度: {normal_width:.1f}")
        else:
            print(f"  自适应: 已关闭（固定温度 {TEMPERATURE}）")

        group_poems = []

        for i in range(1, POEMS_PER_GROUP + 1):
            # 唯一编码
            code = f"{group['id']}_{i:02d}"

            # 构建命令行参数
            cmd = [
                sys.executable, "predict.py",
                "--genre", genre,
                "--prompt", PROMPT,
                "--temperature", str(TEMPERATURE),
                "--max_new", str(MAX_NEW),
                "--stop_newline",
            ]

            if group["use_adaptive"]:
                cmd += [
                    "--entropy_high", str(group["entropy_high"]),
                    "--entropy_mid", str(group["entropy_mid"]),
                    "--entropy_low", str(group["entropy_low"]),
                ]
            else:
                cmd.append("--no_adaptive")

            # 运行 predict.py
            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    timeout=120,
                    cwd=here,
                )
            except subprocess.TimeoutExpired:
                print(f"  [{code}] 超时，跳过")
                continue

            if result.returncode != 0:
                stderr_text = result.stderr.decode("gbk", errors="replace")
                print(f"  [{code}] 出错: {stderr_text[:200]}")
                continue

            # 解码 stdout（Windows 中文系统优先尝试 gbk）
            output = None
            for enc in ["gbk", "utf-8", "gb18030"]:
                try:
                    output = result.stdout.decode(enc)
                    break
                except UnicodeDecodeError:
                    continue
            if output is None:
                output = result.stdout.decode("gbk", errors="replace")

            poem_text = extract_poem(output)

            if not poem_text or len(poem_text) < 5:
                print(f"  [{code}] 生成内容过短，跳过")
                continue

            # 保存单首诗
            poem_file = os.path.join(group_folder, f"{code}.txt")
            with open(poem_file, "w", encoding="utf-8") as f:
                f.write(f"# {code} | {group['name']}\n")
                f.write(f"# 阈值: high={group['entropy_high']}, mid={group['entropy_mid']}, low={group['entropy_low']}\n")
                f.write(f"# use_adaptive={group['use_adaptive']}\n")
                f.write(f"# 生成时间: {datetime.now().isoformat()}\n")
                f.write("\n")
                f.write(poem_text)

            group_poems.append({
                "code": code,
                "group_id": group["id"],
                "group_name": group["name"],
                "poem_num": i,
                "text": poem_text,
                "params": {
                    "entropy_high": group["entropy_high"],
                    "entropy_mid": group["entropy_mid"],
                    "entropy_low": group["entropy_low"],
                    "use_adaptive": group["use_adaptive"],
                },
            })

            print(f"  [{code}] ✓ ({len(poem_text)} 字)")

        all_poems.extend(group_poems)
        print(f"  完成: {len(group_poems)}/{POEMS_PER_GROUP} 首")

        # 保存组内汇总
        summary_file = os.path.join(group_folder, "_summary.txt")
        with open(summary_file, "w", encoding="utf-8") as f:
            f.write(f"组: [{group['id']}] {group['name']}\n")
            f.write(f"阈值: high={group['entropy_high']}, mid={group['entropy_mid']}, low={group['entropy_low']}\n")
            f.write(f"use_adaptive={group['use_adaptive']}\n")
            f.write(f"共 {len(group_poems)} 首\n")
            f.write("=" * 40 + "\n\n")
            for p in group_poems:
                f.write(f"--- {p['code']} ---\n{p['text']}\n\n")

    # =========================================================================
    # 生成匿名化评分表（随机排序，避免组别偏见）
    # =========================================================================
    print(f"\n{'=' * 60}")
    print("生成匿名化评分表...")

    # 随机打乱
    shuffled = all_poems.copy()
    random.shuffle(shuffled)

    # 重新分配匿名序号
    blind_codes = {}
    for i, poem in enumerate(shuffled):
        blind_code = f"P{i+1:02d}"
        blind_codes[blind_code] = poem["code"]
        poem["blind_code"] = blind_code

    # 写入 CSV 评分表
    scoring_csv = os.path.join(out_dir, "scoring_table.csv")
    with open(scoring_csv, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["匿名编号", "诗作内容", "流畅度(1-10)", "主题贴合度(1-10)", "押韵(1-10)", "备注"])
        for poem in shuffled:
            # 把诗中的换行替换为 " / " 方便 CSV 显示
            text_one_line = poem["text"].replace("\n", " / ").replace("\r", "")
            writer.writerow([
                poem["blind_code"],
                text_one_line,
                "",   # 留空供人工填写
                "",
                "",
                "",
            ])

    # 写入 code 映射表（评分后再查看）
    mapping_json = os.path.join(out_dir, "code_mapping.json")
    with open(mapping_json, "w", encoding="utf-8") as f:
        json.dump({
            "description": "评分结束后用此文件解码匿名编号 → 实验组别",
            "mapping": {p["blind_code"]: {
                "original_code": p["code"],
                "group_id": p["group_id"],
                "group_name": p["group_name"],
                "entropy_high": p["params"]["entropy_high"],
                "entropy_mid": p["params"]["entropy_mid"],
                "entropy_low": p["params"]["entropy_low"],
                "use_adaptive": p["params"]["use_adaptive"],
            } for p in shuffled},
        }, f, ensure_ascii=False, indent=2)

    # 统计
    print(f"\n生成完成:")
    print(f"  总诗作数: {len(all_poems)}")
    print(f"  评分表:   {scoring_csv}")
    print(f"  映射表:   {mapping_json}")
    print(f"  诗作目录: {poems_dir}")
    print(f"\n评分步骤:")
    print(f"  1. 打开 scoring_table.csv，对每首诗填写三个维度 1-10 分")
    print(f"  2. 评分完成后，用 code_mapping.json 解码组别")
    print(f"  3. 按组汇总，比较各组均值")


def extract_poem(output: str) -> str:
    """从 predict.py 的 stdout 中提取生成的诗作内容"""
    lines = output.strip().split("\n")
    poem_lines = []
    in_poem = False
    for line in lines:
        stripped = line.strip()
        if stripped == "=" * 40:
            if not in_poem:
                in_poem = True
            else:
                break  # 结束分隔线
            continue
        if in_poem and stripped:
            poem_lines.append(stripped)
    return "\n".join(poem_lines)


if __name__ == "__main__":
    main()
