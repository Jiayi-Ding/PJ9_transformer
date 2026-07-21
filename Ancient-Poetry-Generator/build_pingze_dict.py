#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""生成词表汉字平仄映射表的工具模块。

本模块读取 prepare_data.py 生成的 vocab.json，通过拼音与平水韵分类器
推断每个汉字的平仄属性，并保存为 pingze_dict.json。
v13 相较于 v1 增加了繁体字 fallback 与拼音兜底机制，提升了字表覆盖率。
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

cc_s2t = OpenCC('s2t')

def get_pingze_by_pinyin(char: str) -> int:
    """通过拼音声调推断汉字平仄。

    如果字符不是汉字或拼音解析失败，则返回 2 表示未知。
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

    try:
        result = classifier.classify(ch)
        if result and len(result) > 0:
            if result[0] == 'ping':
                return 0
            elif result[0] == 'ze':
                return 1

    except Exception:
        pass

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

    return get_pingze_by_pinyin(ch)

def main() -> None:
    """从词表加载字符并生成平仄映射表。

    通过直接分类、繁体字符转换后分类、拼音推断三层 fallback，尽量覆盖词表中所有汉字。
    生成结果包括平声、仄声和未知类别统计，并写入 pingze_dict.json。
    """
    vocab_path = "vocab.json"
    output_path = "pingze_dict.json"

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

    classifier = PingZeClassifier()

    pingze_dict = {}
    stats = {0: 0, 1: 0, 2: 0}

    fallback_stats = {"direct": 0, "trad": 0, "pinyin": 0}

    for idx, ch in enumerate(stoi.keys()):

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

        pingze = get_pingze_by_pinyin(ch)
        fallback_stats["pinyin"] += 1
        pingze_dict[ch] = pingze
        stats[pingze] += 1

        if (idx + 1) % 1000 == 0:
            print(f"  已处理 {idx + 1} / {len(stoi)} 个字符...")

    print(f"平声字: {stats[0]}, 仄声字: {stats[1]}, 其他: {stats[2]}")
    print(f"分类方式统计: 直接匹配={fallback_stats['direct']}, 繁体转换后匹配={fallback_stats['trad']}, 拼音兜底={fallback_stats['pinyin']}")

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(pingze_dict, f, ensure_ascii=False, indent=2)

    print(f"平仄映射表已保存至: {output_path}")

if __name__ == "__main__":
    main()
