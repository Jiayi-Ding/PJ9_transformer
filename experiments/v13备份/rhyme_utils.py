# -*- coding: utf-8 -*-
"""
韵母处理工具模块
提供汉字→韵母映射、押韵判断、韵母过滤等功能
"""

import json
import os
from typing import Dict, List, Optional, Set, Tuple


class RhymeHelper:
    """韵母帮助类，管理押韵相关操作"""
    
    def __init__(self, rhyme_dict_path: str = "rhyme_dict.json"):
        """
        初始化，加载韵母映射表
        
        Args:
            rhyme_dict_path: 韵母映射表路径
        """
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
        """
        获取单个汉字的韵母
        
        Args:
            char: 汉字
            
        Returns:
            韵母字符串，若无法获取则返回 None
        """
        return self.char_to_vowel.get(char)
    
    def is_rhyme(self, char1: str, char2: str) -> bool:
        """
        判断两个字是否押韵（韵母相同）
        
        Args:
            char1: 第一个汉字
            char2: 第二个汉字
            
        Returns:
            是否押韵
        """
        v1 = self.get_vowel(char1)
        v2 = self.get_vowel(char2)
        if v1 is None or v2 is None:
            return False
        return v1 == v2
    
    def get_rhyme_group(self, vowel: str) -> List[str]:
        """
        获取指定韵母的所有汉字列表
        
        Args:
            vowel: 韵母
            
        Returns:
            该韵母对应的汉字列表
        """
        return self.vowel_to_chars.get(vowel, [])
    
    def filter_by_vowel(self, token_ids: List[int], itos: Dict[int, str], 
                        target_vowel: str) -> Set[int]:
        """
        根据韵母过滤 token id 集合
        
        Args:
            token_ids: 候选 token id 列表
            itos: id → 字符 映射
            target_vowel: 目标韵母
            
        Returns:
            符合韵母条件的 token id 集合
        """
        rhyme_chars = set(self.get_rhyme_group(target_vowel))
        result = set()
        for tid in token_ids:
            char = itos.get(tid)
            if char and char in rhyme_chars:
                result.add(tid)
        return result
    
    def is_rhyme_friendly(self, vowel: str, min_chars: int = 30) -> bool:
        """
        判断韵母是否"友好"（是否有足够多的可押字）
        
        Args:
            vowel: 韵母
            min_chars: 最小字数阈值
            
        Returns:
            是否友好
        """
        chars = self.get_rhyme_group(vowel)
        return len(chars) >= min_chars
    
    def get_friendly_vowels(self, min_chars: int = 30) -> List[Tuple[str, int]]:
        """
        获取所有友好的韵母列表（按字数降序）
        
        Args:
            min_chars: 最小字数阈值
            
        Returns:
            [(韵母, 字数), ...]
        """
        result = []
        for vowel, chars in self.vowel_to_chars.items():
            if len(chars) >= min_chars:
                result.append((vowel, len(chars)))
        result.sort(key=lambda x: -x[1])
        return result
    
    def explain(self, char: str) -> str:
        """调试用：返回字符的韵母信息"""
        vowel = self.get_vowel(char)
        if vowel:
            count = len(self.get_rhyme_group(vowel))
            return f"'{char}' → 韵母 '{vowel}' (共 {count} 个可押字)"
        else:
            return f"'{char}' → 无法获取韵母"