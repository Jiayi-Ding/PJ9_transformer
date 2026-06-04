# -*- coding: utf-8 -*-
"""
v6.1 实验结果分析与可视化
用法: python analyze.py --genre 5    (五言)
      python analyze.py --genre 7    (七言)
"""

import argparse
import csv
import json
import os
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np


def setup_chinese_font():
    """尝试多种方式设置中文字体"""
    for name in ["Microsoft YaHei", "SimHei", "SimSun", "KaiTi", "FangSong",
                  "Noto Sans CJK SC", "WenQuanYi Micro Hei", "STSong"]:
        for f in fm.fontManager.ttflist:
            if f.name == name:
                plt.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
                plt.rcParams["axes.unicode_minus"] = False
                print(f"使用字体: {name}")
                return
    common_paths = [
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/simhei.ttf",
        "C:/Windows/Fonts/simsun.ttc",
        "C:/Windows/Fonts/simkai.ttf",
    ]
    for path in common_paths:
        if os.path.exists(path):
            try:
                prop = fm.FontProperties(fname=path)
                font_name = prop.get_name()
                plt.rcParams["font.sans-serif"] = [font_name, "DejaVu Sans"]
                plt.rcParams["axes.unicode_minus"] = False
                print(f"使用字体: {font_name} ({path})")
                return
            except Exception:
                continue
    print("警告: 未找到中文字体，图表中的中文可能显示为方框")


def load_scores(csv_path):
    """读取评分表，自动识别列名，返回 {blind_code: {fluency, rhyme, topic}}"""
    scores = {}
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames

        # 自动识别列名
        fluency_col = None
        rhyme_col = None
        topic_col = None
        for h in headers:
            h_clean = h.strip()
            if "流畅" in h_clean:
                fluency_col = h
            elif "押韵" in h_clean or "韵" in h_clean:
                rhyme_col = h
            elif "主题" in h_clean:
                topic_col = h

        for row in reader:
            code = row["匿名编号"].strip()
            fluency = row.get(fluency_col, "").strip() if fluency_col else ""
            rhyme = row.get(rhyme_col, "").strip() if rhyme_col else ""
            topic = row.get(topic_col, "").strip() if topic_col else ""

            if fluency:
                scores[code] = {
                    "fluency": int(fluency),
                    "rhyme": int(rhyme) if rhyme else None,
                    "topic": int(topic) if topic else None,
                }
    return scores


def load_mapping(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["mapping"]


def merge_data(scores, mapping):
    groups = defaultdict(lambda: {
        "group_name": "",
        "normal_width": None,
        "use_adaptive": True,
        "fluency": [],
        "rhyme": [],
        "topic": [],
    })

    for blind_code, info in mapping.items():
        if blind_code not in scores:
            continue
        gid = info["group_id"]
        score = scores[blind_code]

        if groups[gid]["group_name"] == "":
            groups[gid]["group_name"] = info["group_name"]
            if info["use_adaptive"] and info["entropy_high"] is not None:
                groups[gid]["normal_width"] = info["entropy_high"] - info["entropy_low"]
            else:
                groups[gid]["normal_width"] = float("inf")
            groups[gid]["use_adaptive"] = info["use_adaptive"]

        groups[gid]["fluency"].append(score["fluency"])
        if score["rhyme"] is not None:
            groups[gid]["rhyme"].append(score["rhyme"])
        if score["topic"] is not None:
            groups[gid]["topic"].append(score["topic"])

    return groups


def print_summary(groups, genre_label):
    sorted_keys = sorted(groups.keys())

    print("\n" + "=" * 60)
    print("  %s 实验结果" % genre_label)
    print("=" * 60)

    for gid in sorted_keys:
        g = groups[gid]
        f_arr = np.array(g["fluency"])
        r_arr = np.array(g["rhyme"]) if g["rhyme"] else None

        w = g["normal_width"]
        width_str = "%.1f" % w if w != float("inf") else "inf"
        print("\n  [%s] %-12s  normal_width=%s  n=%d" % (gid, g["group_name"], width_str, len(f_arr)))
        print("    流畅度: mean=%.2f  std=%.2f  max=%.0f  min=%.0f" % (
            f_arr.mean(), f_arr.std(), f_arr.max(), f_arr.min()))
        if r_arr is not None and len(r_arr) > 0:
            print("    押韵:   mean=%.2f  std=%.2f  max=%.0f  min=%.0f" % (
                r_arr.mean(), r_arr.std(), r_arr.max(), r_arr.min()))

    sep = "-" * 50
    print("\n" + sep)
    print("流畅度均值排名:")
    ranked = sorted(sorted_keys, key=lambda k: np.mean(groups[k]["fluency"]), reverse=True)
    for rank, gid in enumerate(ranked, 1):
        g = groups[gid]
        m = np.mean(g["fluency"])
        bar = "|" * int(m * 3)
        print("  %d. [%s] %-12s  %.2f  %s" % (rank, gid, g["group_name"], m, bar))

    if any(len(groups[gid]["rhyme"]) > 0 for gid in sorted_keys):
        print("\n押韵均值排名:")
        ranked_r = sorted(sorted_keys, key=lambda k: np.mean(groups[k]["rhyme"]) if groups[k]["rhyme"] else 0, reverse=True)
        for rank, gid in enumerate(ranked_r, 1):
            g = groups[gid]
            if g["rhyme"]:
                m = np.mean(g["rhyme"])
                bar = "|" * int(m * 3)
                print("  %d. [%s] %-12s  %.2f  %s" % (rank, gid, g["group_name"], m, bar))


def plot_results(groups, out_dir, genre_label):
    sorted_keys = sorted(groups.keys())
    labels = ["[%s]\n%s" % (gid, groups[gid]["group_name"]) for gid in sorted_keys]
    colors = ["#4472C4", "#ED7D31", "#A5A5A5", "#FFC000", "#5B9BD5"]

    # ---- 图1: 流畅度 柱状图+箱线图 ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    fluency_means = [np.mean(groups[gid]["fluency"]) for gid in sorted_keys]
    fluency_stds = [np.std(groups[gid]["fluency"]) for gid in sorted_keys]

    bars = axes[0].bar(labels, fluency_means, color=colors, edgecolor="white", linewidth=0.8)
    axes[0].errorbar(labels, fluency_means, yerr=fluency_stds, fmt="none",
                     ecolor="#333333", capsize=5, capthick=1.2, linewidth=1.2)
    axes[0].set_ylabel("fluency mean (1-10)", fontsize=12)
    axes[0].set_title("%s - fluency by group" % genre_label, fontsize=13, fontweight="bold")
    axes[0].set_ylim(0, 11)
    axes[0].grid(axis="y", alpha=0.3, linestyle="--")
    for bar, val in zip(bars, fluency_means):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                     "%.2f" % val, ha="center", va="bottom", fontsize=11, fontweight="bold")

    fluency_data = [groups[gid]["fluency"] for gid in sorted_keys]
    bp = axes[1].boxplot(fluency_data, tick_labels=labels, patch_artist=True,
                          showmeans=True, meanline=True)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    axes[1].set_ylabel("fluency (1-10)", fontsize=12)
    axes[1].set_title("%s - fluency distribution" % genre_label, fontsize=13, fontweight="bold")
    axes[1].set_ylim(0, 11)
    axes[1].grid(axis="y", alpha=0.3, linestyle="--")

    plt.tight_layout()
    path1 = os.path.join(out_dir, "fluency_comparison.png")
    fig.savefig(path1, dpi=150, bbox_inches="tight")
    print("saved: %s" % path1)
    plt.close(fig)

    # ---- 图2: 押韵柱状图 ----
    if any(len(groups[gid]["rhyme"]) > 0 for gid in sorted_keys):
        fig, ax = plt.subplots(figsize=(8, 5))
        rm = [np.mean(groups[gid]["rhyme"]) if groups[gid]["rhyme"] else 0 for gid in sorted_keys]
        rs = [np.std(groups[gid]["rhyme"]) if groups[gid]["rhyme"] else 0 for gid in sorted_keys]

        bars = ax.bar(labels, rm, color=colors, edgecolor="white", linewidth=0.8)
        ax.errorbar(labels, rm, yerr=rs, fmt="none",
                    ecolor="#333333", capsize=5, capthick=1.2, linewidth=1.2)
        ax.set_ylabel("rhyme mean (1-10)", fontsize=12)
        ax.set_title("%s - rhyme by group" % genre_label, fontsize=13, fontweight="bold")
        ax.set_ylim(0, 11)
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        for bar, val in zip(bars, rm):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                    "%.2f" % val, ha="center", va="bottom", fontsize=11, fontweight="bold")

        plt.tight_layout()
        path2 = os.path.join(out_dir, "rhyme_comparison.png")
        fig.savefig(path2, dpi=150, bbox_inches="tight")
        print("saved: %s" % path2)
        plt.close(fig)

    # ---- 图3: 流畅度 vs 正常区宽度 ----
    fig, ax = plt.subplots(figsize=(8, 5))
    x_vals, y_means, y_stds, pt_labels = [], [], [], []
    for gid in sorted_keys:
        g = groups[gid]
        w = g["normal_width"]
        if w == float("inf"):
            x_vals.append(6.0)
            pt_labels.append("[%s] no_adaptive" % gid)
        else:
            x_vals.append(w)
            pt_labels.append("[%s] %s" % (gid, g["group_name"]))
        y_means.append(np.mean(g["fluency"]))
        y_stds.append(np.std(g["fluency"]))

    ax.errorbar(x_vals, y_means, yerr=y_stds, fmt="o-", color="#4472C4",
                capsize=5, markersize=10, linewidth=2, markerfacecolor="white",
                markeredgewidth=2)
    for x, y, lbl in zip(x_vals, y_means, pt_labels):
        ax.annotate(lbl, (x, y), textcoords="offset points", xytext=(0, 12),
                    ha="center", fontsize=9)

    ax.set_xlabel("normal-zone width", fontsize=12)
    ax.set_ylabel("fluency mean", fontsize=12)
    ax.set_title("%s - fluency vs intervention threshold" % genre_label, fontsize=13, fontweight="bold")
    ax.set_ylim(0, 11)
    ax.grid(alpha=0.3, linestyle="--")

    plt.tight_layout()
    path3 = os.path.join(out_dir, "fluency_vs_width.png")
    fig.savefig(path3, dpi=150, bbox_inches="tight")
    print("saved: %s" % path3)
    plt.close(fig)

    # ---- 图4: 五言 vs 七言 对比 (仅在两个结果都存在时) ----
    # 这个在主函数里处理


def main():
    ap = argparse.ArgumentParser(description="v6.1 实验结果分析")
    ap.add_argument("--genre", type=str, choices=["5", "7", "ci"], default="5",
                    help="体裁: 5(五言) / 7(七言) / ci(词)")
    args = ap.parse_args()
    genre = args.genre

    genre_label = {"5": "五言", "7": "七言", "ci": "词"}.get(genre, genre)

    here = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(here, "experiment_results", genre)

    csv_path = os.path.join(out_dir, "scoring_table.csv")
    json_path = os.path.join(out_dir, "code_mapping.json")

    if not os.path.isfile(csv_path):
        print("错误: 找不到评分表 %s" % csv_path, file=sys.stderr)
        sys.exit(1)
    if not os.path.isfile(json_path):
        print("错误: 找不到映射表 %s" % json_path, file=sys.stderr)
        sys.exit(1)

    setup_chinese_font()

    scores = load_scores(csv_path)
    mapping = load_mapping(json_path)
    groups = merge_data(scores, mapping)

    print("加载评分: %d 条" % len(scores))
    print("加载映射: %d 条" % len(mapping))
    print("分组数:   %d" % len(groups))

    print_summary(groups, genre_label)
    plot_results(groups, out_dir, genre_label)

    # ---- 五言+七言 对比图 ----
    # 检查是否有五言和七言两个结果
    out5 = os.path.join(here, "experiment_results", "5")
    out7 = os.path.join(here, "experiment_results", "7")
    has5 = os.path.isfile(os.path.join(out5, "scoring_table.csv")) and \
           os.path.isfile(os.path.join(out5, "code_mapping.json"))
    has7 = os.path.isfile(os.path.join(out7, "scoring_table.csv")) and \
           os.path.isfile(os.path.join(out7, "code_mapping.json"))

    if has5 and has7:
        print("\n生成五言 vs 七言对比图...")
        # 加载五言
        s5 = load_scores(os.path.join(out5, "scoring_table.csv"))
        m5 = load_mapping(os.path.join(out5, "code_mapping.json"))
        g5 = merge_data(s5, m5)
        # 加载七言
        s7 = load_scores(os.path.join(out7, "scoring_table.csv"))
        m7 = load_mapping(os.path.join(out7, "code_mapping.json"))
        g7 = merge_data(s7, m7)

        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(5)
        width = 0.35
        colors_5y = "#4472C4"
        colors_7y = "#ED7D31"

        sorted_k = sorted(g5.keys())
        m5 = [np.mean(g5[k]["fluency"]) for k in sorted_k]
        s5_std = [np.std(g5[k]["fluency"]) for k in sorted_k]
        m7 = [np.mean(g7[k]["fluency"]) for k in sorted_k]
        s7_std = [np.std(g7[k]["fluency"]) for k in sorted_k]

        group_labels = ["[%s]\n%s" % (k, g5[k]["group_name"]) for k in sorted_k]

        bars1 = ax.bar(x - width/2, m5, width, yerr=s5_std, capsize=4,
                        color=colors_5y, edgecolor="white", label="五言")
        bars2 = ax.bar(x + width/2, m7, width, yerr=s7_std, capsize=4,
                        color=colors_7y, edgecolor="white", label="七言")

        ax.set_xticks(x)
        ax.set_xticklabels(group_labels, fontsize=9)
        ax.set_ylabel("fluency mean", fontsize=12)
        ax.set_title("五言 vs 七言 fluency comparison", fontsize=14, fontweight="bold")
        ax.set_ylim(0, 11)
        ax.legend(fontsize=11)
        ax.grid(axis="y", alpha=0.3, linestyle="--")

        for bar, val in zip(bars1, m5):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.15,
                    "%.2f" % val, ha="center", va="bottom", fontsize=9, fontweight="bold")
        for bar, val in zip(bars2, m7):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.15,
                    "%.2f" % val, ha="center", va="bottom", fontsize=9, fontweight="bold")

        plt.tight_layout()
        path4 = os.path.join(here, "experiment_results", "compare_5vs7.png")
        fig.savefig(path4, dpi=150, bbox_inches="tight")
        print("saved: %s" % path4)
        plt.close(fig)

    print("\n分析完成。图表保存在: %s" % out_dir)


if __name__ == "__main__":
    main()
