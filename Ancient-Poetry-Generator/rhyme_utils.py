# -*- coding: utf-8 -*-
"""押韵辅助工具模块。

本模块用于加载 rhyme_dict.json，提供韵母查询、押韵组获取、
过滤可押字符及押韵友好度判断等辅助功能，支持生成阶段的押韵约束。
"""

import json
import os
from typing import Dict, List, Optional, Set, Tuple

class RhymeHelper:
    """押韵辅助类，封装韵母映射与押韵查询逻辑。"""

    def __init__(self, rhyme_dict_path: str = "rhyme_dict.json"):

        self.char_to_vowel: Dict[str, str] = {}
        self.vowel_to_chars: Dict[str, List[str]] = {}
        self.missing_chars: Set[str] = set()

        if os.path.isfile(rhyme_dict_path):
            with open(rhyme_dict_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.char_to_vowel = data.get("char_to_vowel", {})
                self.vowel_to_chars = data.get("vowel_to_chars", {})
                self.missing_chars = set(data.get("missing_chars", []))
            print(f"已加载韵母映射表: {len(self.char_to_vowel)} 字, {len(self.vowel_to_chars)} 个韵母")
        else:
            print(f"警告: 找不到 {rhyme_dict_path}，押韵功能不可用")

    def get_vowel(self, char: str) -> Optional[str]:
        """返回字符对应的韵母，若无法找到则返回 None。"""
        return self.char_to_vowel.get(char)

    def is_rhyme(self, char1: str, char2: str) -> bool:
        """判断两个字符是否押同一个韵母。"""
        v1 = self.get_vowel(char1)
        v2 = self.get_vowel(char2)
        if v1 is None or v2 is None:
            return False
        return v1 == v2

    def get_rhyme_group(self, vowel: str) -> List[str]:
        """返回指定韵母对应的所有字符列表。"""
        return self.vowel_to_chars.get(vowel, [])

    def filter_by_vowel(self, token_ids: List[int], itos: Dict[int, str],
                        target_vowel: str) -> Set[int]:
        """从 token id 列表中过滤出可押 target_vowel 的 token。"""
        rhyme_chars = set(self.get_rhyme_group(target_vowel))
        result = set()
        for tid in token_ids:
            char = itos.get(tid)
            if char and char in rhyme_chars:
                result.add(tid)
        return result

    def is_rhyme_friendly(self, vowel: str, min_chars: int = 30) -> bool:
        """判断某个韵母是否在韵母集合中足够友好（候选字数量足够多）。"""
        chars = self.get_rhyme_group(vowel)
        return len(chars) >= min_chars

    def get_friendly_vowels(self, min_chars: int = 30) -> List[Tuple[str, int]]:
        """返回候选字符数大于等于 min_chars 的韵母及其数量。"""
        result = []
        for vowel, chars in self.vowel_to_chars.items():
            if len(chars) >= min_chars:
                result.append((vowel, len(chars)))
        result.sort(key=lambda x: -x[1])
        return result

    def explain(self, char: str) -> str:
        """返回字符的韵母解释信息，包含同韵字数量。"""
        vowel = self.get_vowel(char)
        if vowel:
            count = len(self.get_rhyme_group(vowel))
            return f"'{char}' → 韵母 '{vowel}' (共 {count} 个可押字)"
        else:
            return f"'{char}' → 无法获取韵母"
