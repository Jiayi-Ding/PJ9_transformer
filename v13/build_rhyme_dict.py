#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
构建汉字 → 韵母 映射表
基于 pypinyin 库，将汉字转换为拼音后提取韵母
运行方式：python build_rhyme_dict.py
输出：rhyme_dict.json

pip install pypinyin  # 如果未安装
python build_rhyme_dict.py
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
    """
    从拼音中提取韵母（去掉声调）
    例如: 'yue3' -> 'yue', 'hua1' -> 'hua', 'an1' -> 'an'
    """
    if not pinyin_str:
        return ""
    # 去除末尾声调数字
    vowel = ''.join([c for c in pinyin_str if not c.isdigit()])
    return vowel


def build_rhyme_dict_from_vocab(vocab_path: str, output_path: str) -> None:
    """
    从词汇表构建汉字→韵母映射
    只处理词汇表中存在的汉字，避免浪费
    """
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
            # 获取拼音（不带声调）
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
    
    # 统计信息
    print(f"\n成功映射: {len(rhyme_map)} 个字符")
    print(f"未能映射: {len(missing_chars)} 个字符")
    print(f"韵母种类: {len(vowel_to_chars)}")
    
    # 显示前10个常见韵母的字数
    print("\n常见韵母统计（前10）:")
    for vowel, chars_list in sorted(vowel_to_chars.items(), key=lambda x: -len(x[1]))[:10]:
        print(f"  {vowel}: {len(chars_list)} 字")
    
    # 保存映射表
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
    # 默认路径
    vocab_path = "vocab.json"
    output_path = "rhyme_dict.json"
    
    import os
    if not os.path.isfile(vocab_path):
        print(f"错误: 找不到 {vocab_path}，请先运行 prepare_data.py")
        sys.exit(1)
    
    build_rhyme_dict_from_vocab(vocab_path, output_path)
    print("\n提示: 可在预测时使用 --rhyme 或 --auto_rhyme 启用押韵功能")