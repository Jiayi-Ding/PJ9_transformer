# -*- coding: utf-8 -*-
"""
自回归续写 / 生成 - 支持主题选择版本

使用示例：
    python predict.py --genre 5 --topic landscape --prompt 春
    python predict.py --genre 7 --topic frontier --prompt 月 --max_new 100
    python predict.py --genre 5 --prompt 春 --auto_rhyme --lines 4
    python predict.py --genre 7 --prompt 月 --rhyme ang --lines 8
    python predict.py --genre 5 --prompt 春 --use_pingze --lines 4
"""
"""【v7: 修改 predict.py 中 load_for_generate 函数以解决 torch.compile 导致的 state_dict 键名前缀问题】"""
"""【v9】
 新增重复惩罚 
 次数递增、
 跳过特殊token、
 性能优化（Counter）
"""
"""【v10】
 新增押韵功能：
 --rhyme: 指定韵脚
 --auto_rhyme: 自动押韵
 --no_rhyme: 禁用押韵
"""
"""【v11】
 新增平仄约束功能：
 --use_pingze: 启用平仄约束（基于标准绝句/律诗格式）
 支持五言/七言、绝句/律诗、平起/仄起自动选择
 与押韵功能协同工作（取交集，若无交集则放弃押韵）
 修复自动定韵时平仄约束失效及标点被选为韵脚的问题
"""

import argparse
import os
import sys
from typing import Dict, List, Tuple, Optional
from collections import Counter

import torch
import torch.nn.functional as F

from dataset import load_vocab_json, load_topic_vocab
from model import CharGPT
from train import get_device

# 导入韵母工具
from rhyme_utils import RhymeHelper


# ========== 平仄模板定义 ==========
# 五言绝句（4句）平起式（首句不押韵）
PINGZE_5_4_PINGQI = [
    [0, 0, 0, 1, 1],  # 平平平仄仄
    [1, 1, 1, 0, 0],  # 仄仄仄平平
    [1, 1, 0, 0, 1],  # 仄仄平平仄
    [0, 0, 1, 1, 0],  # 平平仄仄平
]
# 五言绝句（4句）仄起式（首句不押韵）
PINGZE_5_4_ZEQI = [
    [1, 1, 0, 0, 1],  # 仄仄平平仄
    [0, 0, 1, 1, 0],  # 平平仄仄平
    [0, 0, 0, 1, 1],  # 平平平仄仄
    [1, 1, 1, 0, 0],  # 仄仄仄平平
]

# 五言律诗（8句）仄起式（首句不押韵）
PINGZE_5_8_ZEQI = [
    [1, 1, 0, 0, 1],  # 仄仄平平仄
    [0, 0, 1, 1, 0],  # 平平仄仄平
    [0, 0, 0, 1, 1],  # 平平平仄仄
    [1, 1, 1, 0, 0],  # 仄仄仄平平
    [1, 1, 0, 0, 1],  # 仄仄平平仄
    [0, 0, 1, 1, 0],  # 平平仄仄平
    [0, 0, 0, 1, 1],  # 平平平仄仄
    [1, 1, 1, 0, 0],  # 仄仄仄平平
]
# 五言律诗（8句）平起式（首句不押韵）
PINGZE_5_8_PINGQI = [
    [0, 0, 0, 1, 1],  # 平平平仄仄
    [1, 1, 1, 0, 0],  # 仄仄仄平平
    [1, 1, 0, 0, 1],  # 仄仄平平仄
    [0, 0, 1, 1, 0],  # 平平仄仄平
    [0, 0, 0, 1, 1],  # 平平平仄仄
    [1, 1, 1, 0, 0],  # 仄仄仄平平
    [1, 1, 0, 0, 1],  # 仄仄平平仄
    [0, 0, 1, 1, 0],  # 平平仄仄平
]

# 七言绝句（4句）平起式（首句不押韵）
PINGZE_7_4_PINGQI = [
    [0, 0, 1, 1, 0, 0, 1],  # 平平仄仄平平仄
    [1, 1, 0, 0, 1, 1, 0],  # 仄仄平平仄仄平
    [1, 1, 0, 0, 0, 1, 1],  # 仄仄平平平仄仄
    [0, 0, 1, 1, 1, 0, 0],  # 平平仄仄仄平平
]
# 七言绝句（4句）仄起式（首句不押韵）
PINGZE_7_4_ZEQI = [
    [1, 1, 0, 0, 0, 1, 1],  # 仄仄平平平仄仄
    [0, 0, 1, 1, 1, 0, 0],  # 平平仄仄仄平平
    [0, 0, 1, 1, 0, 0, 1],  # 平平仄仄平平仄
    [1, 1, 0, 0, 1, 1, 0],  # 仄仄平平仄仄平
]

# 七言律诗（8句）仄起式（首句不押韵）
PINGZE_7_8_ZEQI = [
    [1, 1, 0, 0, 0, 1, 1],  # 仄仄平平平仄仄
    [0, 0, 1, 1, 1, 0, 0],  # 平平仄仄仄平平
    [0, 0, 1, 1, 0, 0, 1],  # 平平仄仄平平仄
    [1, 1, 0, 0, 1, 1, 0],  # 仄仄平平仄仄平
    [1, 1, 0, 0, 0, 1, 1],  # 仄仄平平平仄仄
    [0, 0, 1, 1, 1, 0, 0],  # 平平仄仄仄平平
    [0, 0, 1, 1, 0, 0, 1],  # 平平仄仄平平仄
    [1, 1, 0, 0, 1, 1, 0],  # 仄仄平平仄仄平
]
# 七言律诗（8句）平起式（首句不押韵）
PINGZE_7_8_PINGQI = [
    [0, 0, 1, 1, 0, 0, 1],  # 平平仄仄平平仄
    [1, 1, 0, 0, 1, 1, 0],  # 仄仄平平仄仄平
    [1, 1, 0, 0, 0, 1, 1],  # 仄仄平平平仄仄
    [0, 0, 1, 1, 1, 0, 0],  # 平平仄仄仄平平
    [0, 0, 1, 1, 0, 0, 1],  # 平平仄仄平平仄
    [1, 1, 0, 0, 1, 1, 0],  # 仄仄平平仄仄平
    [1, 1, 0, 0, 0, 1, 1],  # 仄仄平平平仄仄
    [0, 0, 1, 1, 1, 0, 0],  # 平平仄仄仄平平
]

# 模板映射字典
PINGZE_TEMPLATES = {
    ("5", 4, "pingqi"): PINGZE_5_4_PINGQI,
    ("5", 4, "zeqi"): PINGZE_5_4_ZEQI,
    ("5", 8, "pingqi"): PINGZE_5_8_PINGQI,
    ("5", 8, "zeqi"): PINGZE_5_8_ZEQI,
    ("7", 4, "pingqi"): PINGZE_7_4_PINGQI,
    ("7", 4, "zeqi"): PINGZE_7_4_ZEQI,
    ("7", 8, "pingqi"): PINGZE_7_8_PINGQI,
    ("7", 8, "zeqi"): PINGZE_7_8_ZEQI,
}


# ========== 平仄帮助类 ==========
class PingzeHelper:
    """平仄帮助类，管理平仄字典和 token 集合"""
    def __init__(self, dict_path: str = "pingze_dict.json"):
        import json
        with open(dict_path, 'r', encoding='utf-8') as f:
            self.char_to_pingze = json.load(f)
        # 预计算平仄 token 集合（稍后在加载 stoi 后初始化）
        self.ping_tokens = None
        self.ze_tokens = None

    def set_stoi(self, stoi: Dict[str, int]):
        """根据词表构建平仄 token 集合"""
        self.ping_tokens = []
        self.ze_tokens = []
        for ch, tid in stoi.items():
            pz = self.char_to_pingze.get(ch, 2)
            if pz == 0:
                self.ping_tokens.append(tid)
            elif pz == 1:
                self.ze_tokens.append(tid)

    def get_tokens_by_pingze(self, expected: int) -> List[int]:
        """返回期望平仄（0或1）对应的 token id 列表（浅拷贝）"""
        if expected == 0:
            return self.ping_tokens.copy()
        elif expected == 1:
            return self.ze_tokens.copy()
        else:
            return []


@torch.no_grad()
def sample_next(
    model: CharGPT, 
    idx: torch.Tensor, 
    topic_id: Optional[int],
    temperature: float, 
    device: torch.device,
    token_counter: Dict[int, int],
    rep_penalty: float = 1.2,
    skip_tokens: Optional[set] = None,
    use_adaptive: bool = False,
    base_temp: float = 0.8,
    temp_factor: float = 1.0,
) -> int:
    """采样下一个 token，支持主题和重复惩罚（次数递增，跳过特殊 token）"""
    model.eval()
    if idx.size(1) == 0:
        raise ValueError("序列为空")
    t = min(idx.size(1), model.block_size)
    x = idx[:, -t:].contiguous()
    
    # 如果有主题，构造 topic_ids 张量
    topic_tensor = None
    if topic_id is not None and model.topic_emb is not None:
        topic_tensor = torch.tensor([topic_id], device=device)
    logits, _ = model(x, topic_ids=topic_tensor)
    last_logits = logits[:, -1, :]  # [B, V]
    
    # ========== 自适应温度：先算熵 ==========
    if use_adaptive:
        # 计算当前概率分布（用基础温度）
        probs = F.softmax(last_logits / max(base_temp, 1e-6), dim=-1)
        
        # 计算熵：-Σ p * log(p)
        log_probs = torch.log(probs + 1e-8)  # 加极小值防止log(0)
        entropy = -(probs * log_probs).sum(dim=-1)  # [B]
        
        # 根据熵调整温度
        # 熵高（犹豫）-> 降温，帮模型做决定
        # 熵低（自信）-> 升温，增加随机性
        if entropy.item() > 2.5:      # 很犹豫
            temp = base_temp * 0.5
        elif entropy.item() > 1.5:    # 中等犹豫
            temp = base_temp * 0.8
        elif entropy.item() < 0.8:    # 很自信
            temp = base_temp * 1.5
        else:                          # 正常
            temp = base_temp
    else:
        temp = temperature

    temp *= temp_factor
    
    # 用调整后的温度缩放 logits
    adjusted_logits = last_logits / max(temp, 1e-6)
    
    # ========== 优化版重复惩罚：次数递增 + 跳过特殊token ==========
    if rep_penalty != 1.0 and token_counter:
        penalized_logits = adjusted_logits.clone()
        # 遍历所有已出现的 token 及其出现次数
        for token_id, count in token_counter.items():
            # 跳过特殊 token（如换行符、标点符号）
            if skip_tokens and token_id in skip_tokens:
                continue
            # 线性递增惩罚：除以 (1 + (rep_penalty-1) * count)
            # count=1 -> 除以 rep_penalty, count=2 -> 除以 1+2*(rep_penalty-1)
            penalty_factor = 1.0 + (rep_penalty - 1.0) * count
            penalized_logits[0, token_id] /= penalty_factor
        adjusted_logits = penalized_logits
    # ============================================================
    
    p = F.softmax(adjusted_logits, dim=-1)
    nxt = torch.multinomial(p, num_samples=1).item()
    return int(nxt)


@torch.no_grad()
def sample_with_allowed(
    model: CharGPT,
    out_ids: List[int],
    topic_id: Optional[int],
    temperature: float,
    device: torch.device,
    allowed_token_ids: List[int],
    temp_factor: float,
    token_counter: Counter,
    rep_penalty: float,
    skip_tokens: set,
) -> int:
    """
    从允许的 token 列表中采样（支持重复惩罚和温度）
    如果 allowed_token_ids 为空或没有有效 token，则降级为正常采样
    """
    if not allowed_token_ids:
        # 降级：正常采样
        return sample_next(model, torch.tensor([out_ids], device=device, dtype=torch.long),
                           topic_id, temperature, device, token_counter, rep_penalty,
                           skip_tokens, use_adaptive=False, base_temp=temperature, temp_factor=1.0)
    
    t_len = min(len(out_ids), model.block_size)
    x = torch.tensor([out_ids[-t_len:]], device=device, dtype=torch.long)
    topic_tensor = None
    if topic_id is not None and model.topic_emb is not None:
        topic_tensor = torch.tensor([topic_id], device=device)
    logits, _ = model(x, topic_ids=topic_tensor)
    last_logits = logits[:, -1, :]  # [1, V]
    
    # 温度缩放
    adjusted_logits = last_logits / max(temperature * temp_factor, 1e-6)
    
    # 重复惩罚
    if rep_penalty != 1.0 and token_counter:
        for token_id, count in token_counter.items():
            if skip_tokens and token_id in skip_tokens:
                continue
            penalty_factor = 1.0 + (rep_penalty - 1.0) * count
            adjusted_logits[0, token_id] /= penalty_factor
    
    # 掩码：只保留允许的 token
    mask = torch.full_like(adjusted_logits[0], float('-inf'))
    for tid in allowed_token_ids:
        if 0 <= tid < mask.size(0):
            mask[tid] = adjusted_logits[0][tid]
    
    # 如果掩码全为 -inf（没有可用 token），降级正常采样
    if not torch.any(mask > float('-inf')):
        p = F.softmax(adjusted_logits, dim=-1)
        return torch.multinomial(p, num_samples=1).item()
    
    p = F.softmax(mask.unsqueeze(0), dim=-1)
    return torch.multinomial(p, num_samples=1).item()


# 保留原有 sample_rhyme_only 作为兼容（内部调用 sample_with_allowed）
@torch.no_grad()
def sample_rhyme_only(
    model: CharGPT,
    out_ids: List[int],
    topic_id: Optional[int],
    temperature: float,
    device: torch.device,
    allowed_token_ids: List[int],
    temp_factor: float,
    token_counter: Counter,
    rep_penalty: float,
    skip_tokens: set,
) -> int:
    """仅从允许的 token 中采样（用于押韵强制约束）"""
    return sample_with_allowed(model, out_ids, topic_id, temperature, device,
                               allowed_token_ids, temp_factor, token_counter,
                               rep_penalty, skip_tokens)


def _encode_chinese_prefix(prefix: str, stoi: Dict[str, int], device: torch.device) -> torch.Tensor:
    ids: List[int] = []
    for ch in prefix:
        if ch not in stoi:
            print(f"警告: 字符不在词表，已跳过: {repr(ch)}", file=sys.stderr)
            continue
        ids.append(int(stoi[ch]))
    if not ids:
        raise ValueError("编码后无有效字，请换起笔或检查词表")
    return torch.tensor([ids], dtype=torch.long, device=device)


def first_in_vocab_char(s: str, stoi: Dict[str, int]) -> str:
    t = s.strip()
    if not t:
        raise ValueError("起笔为空")
    for ch in t:
        if ch in stoi:
            if len(t) > 1:
                print(f"（以首字「{ch}」为起笔；其余字不作为上文）", flush=True)
            return ch
    raise ValueError("输入中无在词表内的字，请另试")


@torch.no_grad()
def generate_one(
    model: CharGPT,
    stoi: Dict[str, int],
    itos: Dict[int, str],
    device: torch.device,
    prompt: str,
    topic_id: Optional[int],
    max_new: int,
    temperature: float,
    stop_newline: bool,
    rep_penalty: float = 1.2,
    skip_newline_penalty: bool = True,
    target_lines: int = 4,
    rhyme_helper: Optional[RhymeHelper] = None,
    rhyme_vowel: Optional[str] = None,
    auto_rhyme: bool = False,
    genre: str = "5",
    use_pingze: bool = False,
    pingze_helper: Optional[PingzeHelper] = None,
) -> str:
    """
    生成古诗，支持押韵约束和平仄约束
    
    新增参数：
        rhyme_helper: 韵母帮助类
        rhyme_vowel: 指定的韵脚（如 "ang"），优先级最高
        auto_rhyme: 自动押韵（根据第一个偶句末字确定韵脚）
        genre: 体裁，"5" 或 "7"
        use_pingze: 是否启用平仄约束
        pingze_helper: 平仄帮助类
    """
    # 根据体裁确定每行字数
    line_len = 5 if genre == "5" else 7
    
    # 根据起笔字选择平仄模板（平起或仄起）
    if use_pingze and pingze_helper:
        first_char = prompt[0] if prompt else "春"
        first_pingze = pingze_helper.char_to_pingze.get(first_char, 2)
        # 选择模板类型：平声为平起，仄声或其他为仄起
        template_type = "pingqi" if first_pingze == 0 else "zeqi"
        template_key = (genre, target_lines, template_type)
        pingze_template = PINGZE_TEMPLATES.get(template_key)
        if pingze_template is None:
            print(f"警告: 未找到平仄模板 (genre={genre}, lines={target_lines}, type={template_type})，将禁用平仄约束")
            use_pingze = False
    else:
        pingze_template = None

    idx = _encode_chinese_prefix(prompt, stoi, device)
    newline_id = stoi.get("\n", None)
    if len(prompt) == 1 and newline_id is not None:
        if idx.size(1) == 0 or int(idx[0, 0].item()) != int(newline_id):
            idx = torch.cat([torch.tensor([[newline_id]], device=device, dtype=torch.long), idx], dim=1)
    
    sentence_end_chars = "，。！？"
    end_token_ids = [stoi.get(ch) for ch in sentence_end_chars if ch in stoi]
    end_token_ids = [tid for tid in end_token_ids if tid is not None]  # 过滤 None

    out_ids: List[int] = idx[0].tolist()

    # 当前已经生成了多少行
    line_count = 0
    # 当前行已生成的字数（不包括标点）
    chars_in_line = 0

    # 初始化：计算起笔字已经占用的字数
    for ch_id in out_ids:
        if ch_id in end_token_ids:
            chars_in_line = 0  # 遇到标点说明上一行结束
        else:
            chars_in_line += 1

    # 统计已生成的中文字符总数（用于平仄位置计算）
    chinese_count = sum(1 for tid in out_ids if '\u4e00' <= itos.get(tid, '') <= '\u9fff')

    # 押韵相关状态
    determined_rhyme_vowel = rhyme_vowel  # 用户指定的韵脚
    first_rhyme_char = None               # 第一个偶句末字
    
    # 当启用自动押韵但未指定韵脚时，等待第一个偶句末字
    need_determine_rhyme = auto_rhyme and determined_rhyme_vowel is None

    # 性能优化：使用 Counter 记录每个 token 出现次数
    token_counter = Counter(out_ids)
    
    # 确定要跳过的特殊 token（换行符 + 标点符号）
    skip_tokens = set()
    if skip_newline_penalty and newline_id is not None:
        skip_tokens.add(newline_id)
    
    # 定义常见标点符号（中英文标点）
    punctuation_chars = "，。！？；：、“”‘’《》【】（）．,.;:?!\"'`~@#$%^&*_+=-—…"
    for ch in punctuation_chars:
        if ch in stoi:
            skip_tokens.add(stoi[ch])
    
    for step in range(max_new):
        # 体裁/首颔颈尾的温度调整
        temp_factor = 1.0
        if target_lines == 8:
            # 颔联
            if line_count in [2, 3]:
                temp_factor = 0.8
            # 颈联
            elif line_count in [4, 5]:
                temp_factor = 1.1
        elif target_lines == 4:
            if line_count in [0, 1]:
                temp_factor = 0.8
            elif line_count in [2, 3]:
                temp_factor = 1.1
        
        # 判断是否需要押韵：当前行最后一个字（且是偶数句）
        is_last_char_of_line = (chars_in_line == line_len - 1)
        need_rhyme = is_last_char_of_line and (line_count % 2 == 1)
        
        # ========== 计算平仄约束 ==========
        expected_pingze = -1  # -1 表示不约束
        if use_pingze and pingze_template is not None:
            # 计算当前行号和句内位置
            line_idx = chinese_count // line_len
            pos_in_line = chinese_count % line_len
            if line_idx < len(pingze_template) and pos_in_line < line_len:
                expected_pingze = pingze_template[line_idx][pos_in_line]
        
        # ========== 构建允许的 token 集合 ==========
        allowed_tokens = None
        
        # 1. 平仄约束（如果有期望平仄）
        if expected_pingze != -1 and pingze_helper is not None:
            pingze_tokens = pingze_helper.get_tokens_by_pingze(expected_pingze)
            allowed_tokens = set(pingze_tokens)
        
        # 2. 押韵约束（仅当有韵脚且需要押韵时）
        if need_rhyme and determined_rhyme_vowel is not None and rhyme_helper is not None:
            rhyme_chars = rhyme_helper.get_rhyme_group(determined_rhyme_vowel)
            rhyme_token_ids = [stoi.get(ch) for ch in rhyme_chars if ch in stoi]
            rhyme_token_ids = [tid for tid in rhyme_token_ids if tid is not None]
            if allowed_tokens is None:
                allowed_tokens = set(rhyme_token_ids)
            else:
                allowed_tokens &= set(rhyme_token_ids)  # 取交集
        
        # ========== 特殊处理：自动确定韵脚阶段 ==========
        if auto_rhyme and need_determine_rhyme and need_rhyme:
            # 构建平仄约束集合（确保不包含标点）
            constraint_tokens = None
            if expected_pingze != -1 and pingze_helper:
                constraint_tokens = set(pingze_helper.get_tokens_by_pingze(expected_pingze))
            else:
                # 如果没有平仄约束，则禁止标点：允许所有平仄为0或1的字符
                if pingze_helper:
                    constraint_tokens = set(pingze_helper.get_tokens_by_pingze(0) + pingze_helper.get_tokens_by_pingze(1))
            
            # 定义约束采样函数
            def sample_with_constraint():
                if constraint_tokens:
                    return sample_with_allowed(
                        model, out_ids, topic_id, temperature, device,
                        list(constraint_tokens), temp_factor, token_counter,
                        rep_penalty, skip_tokens
                    )
                else:
                    return sample_next(
                        model, torch.tensor([out_ids], device=device, dtype=torch.long),
                        topic_id, temperature, device, token_counter, rep_penalty,
                        skip_tokens, use_adaptive=True, base_temp=temperature, temp_factor=temp_factor
                    )
            
            # 第一次采样
            nxt = sample_with_constraint()
            nxt_char = itos.get(nxt, "")
            vowel = rhyme_helper.get_vowel(nxt_char)
            
            # 重试循环（最多3次）
            for retry in range(3):
                if vowel and rhyme_helper.is_rhyme_friendly(vowel, min_chars=30):
                    determined_rhyme_vowel = vowel
                    first_rhyme_char = nxt_char
                    need_determine_rhyme = False
                    print(f"  [押韵] 自动确定韵脚: '{nxt_char}' → {vowel}")
                    break
                else:
                    nxt = sample_with_constraint()
                    nxt_char = itos.get(nxt, "")
                    vowel = rhyme_helper.get_vowel(nxt_char)
            else:
                # 重试失败，使用最后一个采样结果（即使不友好）
                if vowel:
                    determined_rhyme_vowel = vowel
                    first_rhyme_char = nxt_char
                    need_determine_rhyme = False
                    print(f"  [押韵] 警告: 韵脚 '{vowel}' 可押字较少")
                else:
                    print(f"  [押韵] 警告: 无法获取韵母，跳过押韵")
                    need_determine_rhyme = False
            
            # 已经通过采样得到了 nxt，直接跳过后面的采样逻辑
            # 注意：需要更新计数器和字符状态，然后 continue
            out_ids.append(nxt)
            token_counter[nxt] += 1
            if nxt in end_token_ids:
                line_count += 1
                chars_in_line = 0
            else:
                chars_in_line += 1
                if '\u4e00' <= itos.get(nxt, '') <= '\u9fff':
                    chinese_count += 1
            if nxt in end_token_ids and line_count >= target_lines:
                break
            continue  # 跳过后面的普通采样
        
        # ========== 普通采样（非自动定韵阶段） ==========
        if allowed_tokens is not None and len(allowed_tokens) > 0:
            nxt = sample_with_allowed(
                model, out_ids, topic_id, temperature, device,
                list(allowed_tokens), temp_factor, token_counter,
                rep_penalty, skip_tokens
            )
        else:
            # 无约束，正常采样
            nxt = sample_next(
                model, 
                torch.tensor([out_ids], device=device, dtype=torch.long), 
                topic_id, 
                temperature, 
                device,
                token_counter=token_counter,
                rep_penalty=rep_penalty,
                skip_tokens=skip_tokens,
                use_adaptive=True,
                base_temp=temperature,
                temp_factor=temp_factor,
            )
        
        # ========== 押韵修正（非自动定韵阶段） ==========
        if rhyme_helper is not None and need_rhyme and not (auto_rhyme and need_determine_rhyme):
            nxt_char = itos.get(nxt, "")
            # 已有韵脚，检查是否押韵
            if determined_rhyme_vowel is not None and not rhyme_helper.is_rhyme(nxt_char, first_rhyme_char):
                # 不押韵，从押韵候选池中重新采样（同时考虑平仄约束）
                rhyme_chars = rhyme_helper.get_rhyme_group(determined_rhyme_vowel)
                rhyme_token_ids = [stoi.get(ch) for ch in rhyme_chars if ch in stoi]
                rhyme_token_ids = [tid for tid in rhyme_token_ids if tid is not None]
                # 如果有平仄约束，取交集
                if expected_pingze != -1 and pingze_helper:
                    pingze_tokens = set(pingze_helper.get_tokens_by_pingze(expected_pingze))
                    final_allowed = [tid for tid in rhyme_token_ids if tid in pingze_tokens]
                else:
                    final_allowed = rhyme_token_ids
                if final_allowed:
                    nxt = sample_with_allowed(
                        model, out_ids, topic_id, temperature, device,
                        final_allowed, temp_factor, token_counter,
                        rep_penalty, skip_tokens
                    )
                    print(f"  [押韵] 修正为: '{itos.get(nxt)}' (韵母 {determined_rhyme_vowel})")
                else:
                    print(f"  [押韵] 警告: 无法同时满足平仄和押韵，放弃押韵")
        
        out_ids.append(nxt)
        token_counter[nxt] += 1
        
        # 更新字符计数
        if nxt in end_token_ids:
            line_count += 1
            chars_in_line = 0
        else:
            chars_in_line += 1
            if '\u4e00' <= itos.get(nxt, '') <= '\u9fff':
                chinese_count += 1
        
        # 自动结束：检测到结束标点且达到目标句数
        if nxt in end_token_ids and line_count >= target_lines:
            break
    
    return "".join(itos.get(i, "?") for i in out_ids)


def load_for_generate(ckpt_path: str, device: torch.device) -> Tuple[CharGPT, dict]:
    pack = torch.load(ckpt_path, map_location=device)
    sd = pack["model"]
    hp = pack["hparams"]
    m = CharGPT(
        vocab_size=hp["vocab_size"],
        block_size=hp["block_size"],
        d_model=hp["d_model"],
        n_head=hp["n_head"],
        n_layer=hp["n_layer"],
        d_ff=hp["d_ff"],
        dropout=float(hp.get("dropout", 0.1)),
        num_topics=hp.get("num_topics", 0),
    ).to(device)
    
    # 处理 torch.compile 产生的 _orig_mod. 前缀
    new_sd = {}
    for k, v in sd.items():
        if k.startswith('_orig_mod.'):
            new_sd[k[10:]] = v
        else:
            new_sd[k] = v
    
    m.load_state_dict(new_sd, strict=True)
    m.eval()
    return m, hp


def auto_select_topic(prompt: str, topic_vocab_path: str) -> str:
    """
    根据起笔字的第一个字符，自动选择主题。
    如果字符在某个主题的关键词中 → 返回该主题
    如果字符在多个主题的关键词中 → 返回匹配最多的主题
    如果字符不在任何主题的关键词中 → 随机选择
    """
    import json
    import random
    
    with open(topic_vocab_path, 'r', encoding='utf-8') as f:
        topic_data = json.load(f)
    
    keywords = topic_data.get("topic_keywords", {})

    topic_list = topic_data.get("topics", [
        "landscape", "frontier", "homesickness", "historical", "love",
        "objects", "farewell", "time_sorrow", "festival", "palace_grievance",
        "immortal", "zen", "drinking", "elegy", "imperial_exam",
        "war_atrocity", "feminine_life", "reclusion_tourism", "other"
    ])
        
    if not prompt:
        return random.choice(topic_list[:-1])  # 排除 other，随机选一个具体主题
    
    first_char = prompt[0]
    
    # 统计这个字符出现在哪些主题的关键词中
    matched_topics = []
    for topic, kw_list in keywords.items():
        if first_char in kw_list:
            matched_topics.append(topic)
    
    if matched_topics:
        # 有匹配的主题 → 随机选一个匹配的（如果有多个）
        return random.choice(matched_topics)
    else:
        # 没有匹配 → 随机选一个具体主题（排除 other）
        return random.choice(topic_list[:-1])


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    os.chdir(here)

    p = argparse.ArgumentParser(description="古诗生成 - 支持主题选择和押韵")
    
    # 押韵参数
    p.add_argument("--rhyme", type=str, default=None,
                   help="指定韵脚（如 an, ang, ong），启用押韵约束")
    p.add_argument("--auto_rhyme", action="store_true",
                   help="自动押韵（根据第一个偶数句末字确定韵脚）")
    p.add_argument("--no_rhyme", action="store_true",
                   help="禁用押韵（覆盖其他押韵参数）")
    
    # 平仄参数
    p.add_argument("--use_pingze", action="store_true",
                   help="启用平仄约束（基于标准绝句/律诗格式）")
    
    # 模型和体裁参数
    p.add_argument("--ckpt", type=str, default=None)
    p.add_argument("--genre", type=str, required=True, 
                   help="体裁，如：5,7")
    
    # 主题参数
    topic_choices = [
        "landscape", "frontier", "homesickness", "historical", "love",
        "objects", "farewell", "time_sorrow", "festival", "palace_grievance",
        "immortal", "zen", "drinking", "elegy", "imperial_exam",
        "war_atrocity", "feminine_life", "reclusion_tourism", "other"
    ]
    p.add_argument("--topic", type=str, choices=topic_choices, default="other",
                   help=f"主题: {', '.join(topic_choices)}")
    
    # 文件路径参数
    p.add_argument("--vocab", type=str, default="vocab.json")
    p.add_argument("--topic_vocab", type=str, default="topic_vocab.json")
    
    # 生成参数
    p.add_argument("--prompt", type=str, default="春", help="起笔字")
    p.add_argument("--max_new", type=int, default=200, help="新增长度")
    p.add_argument("--lines", type=int, default=4, choices=[4, 8],
                   help="4=绝句，8=律诗")
    p.add_argument("--temperature", type=float, default=0.8, help="温度参数")
    p.add_argument("--stop_newline", action="store_true", help="遇到换行符停止")
    p.add_argument("--rep_penalty", type=float, default=1.2, 
                   help="重复惩罚系数（>1 惩罚已出现字，1 表示无惩罚）")
    p.add_argument("--skip_newline_penalty", action="store_true", default=True, 
                   help="跳过对换行符的惩罚")
    
    args = p.parse_args()

    # 设置 checkpoint 路径
    if args.ckpt is None:
        args.ckpt = f"ckpt_best_{args.genre}.pt"

    if not os.path.isfile(args.ckpt):
        print(f"错误: 未找到模型文件 {args.ckpt}", file=sys.stderr)
        print(f"请先运行: python train.py --genre {args.genre} [--use_topic]", file=sys.stderr)
        sys.exit(1)

    # 加载词表
    stoi, itos, _ = load_vocab_json(args.vocab)
    device = get_device()
    model, hp = load_for_generate(args.ckpt, device)

    # 初始化押韵帮助类
    rhyme_helper = None
    rhyme_vowel = None
    auto_rhyme = False

    # 检查是否禁用押韵
    if not args.no_rhyme:
        rhyme_dict_path = "rhyme_dict.json"
        if os.path.isfile(rhyme_dict_path):
            rhyme_helper = RhymeHelper(rhyme_dict_path)
            
            if args.rhyme:
                rhyme_vowel = args.rhyme
                print(f"押韵模式: 使用指定韵脚 '{rhyme_vowel}'")
            elif args.auto_rhyme:
                auto_rhyme = True
                print("押韵模式: 自动押韵（根据第一个偶句末字确定韵脚）")
            else:
                print("押韵模式: 未启用（使用 --rhyme 或 --auto_rhyme 启用）")
        else:
            if args.rhyme or args.auto_rhyme:
                print(f"警告: 找不到 {rhyme_dict_path}，无法启用押韵。请先运行 python build_rhyme_dict.py")

    # 初始化平仄帮助类
    pingze_helper = None
    if args.use_pingze:
        pingze_dict_path = "pingze_dict.json"
        if os.path.isfile(pingze_dict_path):
            pingze_helper = PingzeHelper(pingze_dict_path)
            pingze_helper.set_stoi(stoi)  # 预计算平仄 token 集合
            print("平仄模式: 已启用（基于标准绝句/律诗格式）")
        else:
            print(f"警告: 找不到 {pingze_dict_path}，无法启用平仄。请先运行 python build_pingze_dict.py")

    # 获取主题 id
    topic_id = None
    if model.topic_emb is not None:
        try:
            topic_list, _ = load_topic_vocab(args.topic_vocab)
            
            # 检查用户是否在命令行中显式指定了 --topic
            user_specified_topic = False
            for i, arg in enumerate(sys.argv):
                if arg == '--topic' and i + 1 < len(sys.argv):
                    user_specified_topic = True
                    break
            
            if user_specified_topic:
                target_topic = args.topic
            else:
                target_topic = auto_select_topic(args.prompt, args.topic_vocab)
                print(f"自动选择主题: {target_topic}")
            
            if target_topic in topic_list:
                topic_id = topic_list.index(target_topic)
        except Exception as e:
            print(f"警告: 主题选择失败: {e}")

    # 生成
    print(f"体裁: {args.genre}, 起笔字: {args.prompt}")
    try:
        text = generate_one(
            model, stoi, itos, device,
            args.prompt,
            topic_id,
            args.max_new,
            args.temperature,
            args.stop_newline,
            rep_penalty=args.rep_penalty,
            skip_newline_penalty=args.skip_newline_penalty,
            target_lines=args.lines,
            rhyme_helper=rhyme_helper,
            rhyme_vowel=rhyme_vowel,
            auto_rhyme=auto_rhyme,
            genre=args.genre,
            use_pingze=args.use_pingze,
            pingze_helper=pingze_helper,
        )
        print("\n" + "=" * 40)
        print(text)
        print("=" * 40)
    except Exception as e:
        print(f"生成失败: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()