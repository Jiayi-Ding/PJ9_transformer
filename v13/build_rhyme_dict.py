#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""生成汉字韵母映射表的工具模块。

本模块读取 prepare_data.py 生成的 vocab.json，使用 pypinyin 计算每个汉字的带声调拼音，
提取韵母并导出 rhyme_dict.json。该映射可用于生成时的押韵约束。

v13 相较于 v1 的改进说明：
- 通过词表直接对齐词汇，提高生成时押韵可用字覆盖率。
- 输出 rhyme_dict.json 包含字符到韵母映射、韵母到字符列表和统计信息。
"""

import json
import sys
from collections import defaultdict

try:
    from pypinyin import pinyin, Style
except ImportError:
    print("请先安装 pypinyin：pip install pypinyin")
    sys.exit(1)

def get_vowel(pinyin_str: str) -> str:

    if not pinyin_str:
        return ""

    vowel = ''.join([c for c in pinyin_str if not c.isdigit()])
    return vowel

def build_rhyme_dict_from_vocab(vocab_path: str, output_path: str) -> None:

    with open(vocab_path, 'r', encoding='utf-8') as f:
        vocab = json.load(f)

    stoi = vocab.get("stoi", {})
    chars = list(stoi.keys())

    rhyme_map = {}
    vowel_to_chars = defaultdict(list)
    missing_chars = []

    print(f"正在处理 {len(chars)} 个字符...")

    for ch in chars:
        try:

            pinyin_list = pinyin(ch, style=Style.TONE3)
            if pinyin_list and pinyin_list[0]:
                py = pinyin_list[0][0]
                vowel = get_vowel(py)
                if vowel:
                    rhyme_map[ch] = vowel
                    vowel_to_chars[vowel].append(ch)
                else:
                    missing_chars.append(ch)
            else:
                missing_chars.append(ch)
        except Exception as e:
            print(f"警告: 处理 '{ch}' 失败: {e}")
            missing_chars.append(ch)

    print(f"\n成功映射: {len(rhyme_map)} 个字符")
    print(f"未能映射: {len(missing_chars)} 个字符")
    print(f"韵母种类: {len(vowel_to_chars)}")

    print("\n常见韵母统计（前10）:")
    for vowel, chars_list in sorted(vowel_to_chars.items(), key=lambda x: -len(x[1]))[:10]:
        print(f"  {vowel}: {len(chars_list)} 字")

    result = {
        "char_to_vowel": rhyme_map,
        "vowel_to_chars": {k: v for k, v in vowel_to_chars.items()},
        "missing_chars": missing_chars,
        "stats": {
            "total_chars": len(chars),
            "mapped_chars": len(rhyme_map),
            "missing_chars": len(missing_chars),
            "vowel_types": len(vowel_to_chars)
        }
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"\n韵母映射表已保存至: {output_path}")

if __name__ == "__main__":

    vocab_path = "vocab.json"
    output_path = "rhyme_dict.json"

    import os
    if not os.path.isfile(vocab_path):
        print(f"错误: 找不到 {vocab_path}，请先运行 prepare_data.py")
        sys.exit(1)

    build_rhyme_dict_from_vocab(vocab_path, output_path)
    print("\n提示: 可在预测时使用 --rhyme 或 --auto_rhyme 启用押韵功能")
