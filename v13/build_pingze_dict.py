#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
构建汉字 → 平仄 映射表（0=平，1=仄，2=其他），使用三级回退策略确保平仄分类：
    1、基于平水韵库 pingshui_rhyme，利用中古音韵进行平仄分类。
    2、对于简体字无法识别的情况，先转换为繁体字再尝试分类；
    3、若仍然失败，则使用拼音法（普通话声调）作为最终兜底。

运行方式：python build_pingze_dict.py
输出：pingze_dict.json

依赖：pip install pingshui_rhyme opencc-python-reimplemented pypinyin
注意：需在 prepare_data.py 生成 vocab.json 之后运行。
"""

import json
import sys

try:
    from pingshui_rhyme import PingZeClassifier
except ImportError:
    print("错误：请先安装 pingshui_rhyme：pip install pingshui_rhyme", file=sys.stderr)
    sys.exit(1)

try:
    from opencc import OpenCC
except ImportError:
    print("错误：请先安装 opencc-python-reimplemented：pip install opencc-python-reimplemented", file=sys.stderr)
    sys.exit(1)

try:
    from pypinyin import pinyin, Style
except ImportError:
    print("错误：请先安装 pypinyin：pip install pypinyin", file=sys.stderr)
    sys.exit(1)

# 初始化繁简转换器（简→繁）
cc_s2t = OpenCC('s2t')


def get_pingze_by_pinyin(char: str) -> int:
    """
    使用拼音法判断平仄（普通话声调）：1/2声→平(0)，3/4声→仄(1)，其他→2
    """
    if not ('\u4e00' <= char <= '\u9fff'):
        return 2
    try:
        py_list = pinyin(char, style=Style.TONE3)
        if not py_list or not py_list[0]:
            return 2
        py = py_list[0][0]
        if py and py[-1].isdigit():
            tone = int(py[-1])
            return 0 if tone in (1, 2) else 1
    except Exception:
        pass
    return 2


def get_pingze_with_fallback(ch: str, classifier: PingZeClassifier) -> int:
    """
    三级回退：
    1. 直接使用平水韵分类器（简体）
    2. 转换为繁体后再次尝试
    3. 使用拼音法（普通话声调）兜底
    返回 0(平), 1(仄), 2(其他)
    """
    # 第一级：简体直接分类
    try:
        result = classifier.classify(ch)
        if result and len(result) > 0:
            if result[0] == 'ping':
                return 0
            elif result[0] == 'ze':
                return 1
            # 如果是 'unknown' 或其他，继续下一级
    except Exception:
        pass

    # 第二级：转换为繁体再分类
    try:
        trad = cc_s2t.convert(ch)
        if trad and trad != ch:
            result = classifier.classify(trad)
            if result and len(result) > 0:
                if result[0] == 'ping':
                    return 0
                elif result[0] == 'ze':
                    return 1
    except Exception:
        pass

    # 第三级：拼音法兜底
    return get_pingze_by_pinyin(ch)


def main() -> None:
    vocab_path = "vocab.json"
    output_path = "pingze_dict.json"

    # 加载词表
    try:
        with open(vocab_path, 'r', encoding='utf-8') as f:
            vocab = json.load(f)
    except FileNotFoundError:
        print(f"错误：找不到 {vocab_path}，请先运行 prepare_data.py 生成词表。", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"错误：{vocab_path} 不是有效的 JSON 文件：{e}", file=sys.stderr)
        sys.exit(1)

    stoi = vocab.get("stoi", {})
    if not stoi:
        print(f"错误：{vocab_path} 中缺少 'stoi' 字段。", file=sys.stderr)
        sys.exit(1)

    print(f"正在处理 {len(stoi)} 个字符...")

    # 初始化平水韵分类器
    classifier = PingZeClassifier()

    pingze_dict = {}
    stats = {0: 0, 1: 0, 2: 0}
    # 记录不同回退级别的统计（可选）
    fallback_stats = {"direct": 0, "trad": 0, "pinyin": 0}

    for idx, ch in enumerate(stoi.keys()):
        # 1. 先尝试直接分类（简体）
        try:
            result = classifier.classify(ch)
            if result and len(result) > 0 and result[0] in ('ping', 'ze'):
                pingze = 0 if result[0] == 'ping' else 1
                fallback_stats["direct"] += 1
                pingze_dict[ch] = pingze
                stats[pingze] += 1
                if (idx + 1) % 1000 == 0:
                    print(f"  已处理 {idx + 1} / {len(stoi)} 个字符...")
                continue
        except Exception:
            pass

        # 2. 繁体转换尝试
        try:
            trad = cc_s2t.convert(ch)
            if trad and trad != ch:
                result = classifier.classify(trad)
                if result and len(result) > 0 and result[0] in ('ping', 'ze'):
                    pingze = 0 if result[0] == 'ping' else 1
                    fallback_stats["trad"] += 1
                    pingze_dict[ch] = pingze
                    stats[pingze] += 1
                    if (idx + 1) % 1000 == 0:
                        print(f"  已处理 {idx + 1} / {len(stoi)} 个字符...")
                    continue
        except Exception:
            pass

        # 3. 拼音法兜底
        pingze = get_pingze_by_pinyin(ch)
        fallback_stats["pinyin"] += 1
        pingze_dict[ch] = pingze
        stats[pingze] += 1

        if (idx + 1) % 1000 == 0:
            print(f"  已处理 {idx + 1} / {len(stoi)} 个字符...")

    # 打印最终分类情况
    print(f"平声字: {stats[0]}, 仄声字: {stats[1]}, 其他: {stats[2]}")
    print(f"分类方式统计: 直接匹配={fallback_stats['direct']}, 繁体转换后匹配={fallback_stats['trad']}, 拼音兜底={fallback_stats['pinyin']}")

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(pingze_dict, f, ensure_ascii=False, indent=2)

    print(f"平仄映射表已保存至: {output_path}")


if __name__ == "__main__":
    main()