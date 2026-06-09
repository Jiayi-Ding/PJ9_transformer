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
 支持五言/七言、绝句/律诗，根据起笔字平仄自动选择平起或仄起格式
 平仄与押韵可同时生效，冲突时优先保证平仄
 修复：标点位置不施加平仄约束，自动韵脚不会选到标点
"""

import argparse
import os
import sys
import json
from typing import Dict, List, Tuple, Optional
from collections import Counter

import torch
import torch.nn.functional as F

from dataset import load_vocab_json, load_topic_vocab
from model import CharGPT
from train import get_device

# 导入韵母工具
from rhyme_utils import RhymeHelper


# ==================== 平仄约束相关定义 ====================

class PingzeHelper:
    """平仄帮助类，管理平仄映射和 token 过滤"""
    def __init__(self, dict_path: str = "pingze_dict.json"):
        with open(dict_path, 'r', encoding='utf-8') as f:
            self.char_to_pingze = json.load(f)
        # 预计算每个平仄类别对应的 token id 集合（性能优化）
        self.pingze_to_tokens = {0: [], 1: [], 2: []}
        self.stoi = None

    def set_stoi(self, stoi: Dict[str, int]):
        """设置词表映射，并预计算 token id 集合"""
        self.stoi = stoi
        self.pingze_to_tokens = {0: [], 1: [], 2: []}
        for ch, tid in stoi.items():
            pz = self.char_to_pingze.get(ch, 2)
            self.pingze_to_tokens[pz].append(tid)

    def get_pingze(self, char: str) -> int:
        return self.char_to_pingze.get(char, 2)

    def get_tokens_by_pingze(self, expected: int) -> List[int]:
        if self.stoi is None:
            return []
        return self.pingze_to_tokens.get(expected, [])


# 五言绝句平仄模板（平起式、仄起式）
PINGZE_5_4_PINGQI = [
    [0, 0, 1, 1, 0],  # 平平仄仄平
    [1, 1, 0, 0, 1],  # 仄仄平平仄
    [0, 0, 0, 1, 1],  # 平平平仄仄
    [1, 1, 1, 0, 0],  # 仄仄仄平平
]
PINGZE_5_4_ZEQI = [
    [1, 1, 0, 0, 1],  # 仄仄平平仄
    [0, 0, 1, 1, 0],  # 平平仄仄平
    [0, 0, 0, 1, 1],  # 平平平仄仄
    [1, 1, 1, 0, 0],  # 仄仄仄平平
]

# 五言律诗平仄模板（仄起式为例，遵循粘对规则）
PINGZE_5_8_ZEQI = [
    [1, 1, 0, 0, 1],  # 首联上句
    [0, 0, 1, 1, 0],  # 首联下句（押韵）
    [0, 0, 0, 1, 1],  # 颔联上句（粘）
    [1, 1, 1, 0, 0],  # 颔联下句（对）
    [1, 1, 0, 0, 1],  # 颈联上句（粘）
    [0, 0, 1, 1, 0],  # 颈联下句（对）
    [0, 0, 0, 1, 1],  # 尾联上句（粘）
    [1, 1, 1, 0, 0],  # 尾联下句（对）
]
# 五言律诗平起式（简单复用仄起式，实际应调整，但为了完整性暂时保留）
PINGZE_5_8_PINGQI = PINGZE_5_8_ZEQI

# 七言绝句平仄模板（平起式、仄起式）
PINGZE_7_4_PINGQI = [
    [0, 0, 1, 1, 0, 0, 1],  # 平平仄仄平平仄
    [1, 1, 0, 0, 1, 1, 0],  # 仄仄平平仄仄平
    [1, 1, 0, 0, 0, 1, 1],  # 仄仄平平平仄仄
    [0, 0, 1, 1, 1, 0, 0],  # 平平仄仄仄平平
]
PINGZE_7_4_ZEQI = [
    [1, 1, 0, 0, 0, 1, 1],  # 仄仄平平平仄仄
    [0, 0, 1, 1, 1, 0, 0],  # 平平仄仄仄平平
    [0, 0, 1, 1, 0, 0, 1],  # 平平仄仄平平仄
    [1, 1, 0, 0, 1, 1, 0],  # 仄仄平平仄仄平
]

# 七言律诗平仄模板（仄起式为例）
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
PINGZE_7_8_PINGQI = PINGZE_7_8_ZEQI


def get_pingze_template(genre: str, target_lines: int, start_pingze: int = 1) -> List[List[int]]:
    """
    根据体裁、行数、首字平仄返回平仄模板。
    genre: "5" 或 "7"
    target_lines: 4 或 8
    start_pingze: 0=平起, 1=仄起（默认）
    """
    if genre == "5":
        if target_lines == 4:
            if start_pingze == 0:
                return PINGZE_5_4_PINGQI
            else:
                return PINGZE_5_4_ZEQI
        else:  # 8句
            if start_pingze == 0:
                return PINGZE_5_8_PINGQI
            else:
                return PINGZE_5_8_ZEQI
    else:  # genre == "7"
        if target_lines == 4:
            if start_pingze == 0:
                return PINGZE_7_4_PINGQI
            else:
                return PINGZE_7_4_ZEQI
        else:
            if start_pingze == 0:
                return PINGZE_7_8_PINGQI
            else:
                return PINGZE_7_8_ZEQI


# ==================== 通用约束采样函数 ====================

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
    used_rhyme_chars: set = None,  # 新增：记录已使用的韵脚字
) -> int:
    """
    从 allowed_token_ids 中采样下一个 token
    used_rhyme_chars: 已经作为韵脚使用过的字符集合（禁止重复）
    """
    if not allowed_token_ids:
        return sample_next(model, torch.tensor([out_ids], device=device, dtype=torch.long),
                           topic_id, temperature, device, token_counter, rep_penalty,
                           skip_tokens, use_adaptive=False, base_temp=temperature, temp_factor=1.0)

    t_len = min(len(out_ids), model.block_size)
    x = torch.tensor([out_ids[-t_len:]], device=device, dtype=torch.long)
    topic_tensor = None
    if topic_id is not None and model.topic_emb is not None:
        topic_tensor = torch.tensor([topic_id], device=device)
    logits, _ = model(x, topic_ids=topic_tensor)
    last_logits = logits[:, -1, :]

    adjusted_logits = last_logits / max(temperature * temp_factor, 1e-6)

    # 重复惩罚
    if rep_penalty != 1.0 and token_counter:
        for token_id, count in token_counter.items():
            if skip_tokens and token_id in skip_tokens:
                continue
            penalty_factor = 1.0 + (rep_penalty - 1.0) * count
            adjusted_logits[0, token_id] /= penalty_factor

    # 过滤允许的 token
    mask = torch.full_like(adjusted_logits[0], float('-inf'))
    for tid in allowed_token_ids:
        # 【关键修改】如果是韵脚字且已经用过了，跳过
        if used_rhyme_chars and tid in used_rhyme_chars:
            continue
        if 0 <= tid < mask.size(0):
            mask[tid] = adjusted_logits[0][tid]

    if not torch.any(mask > float('-inf')):
        # 降级：忽略 used_rhyme_chars 限制
        mask2 = torch.full_like(adjusted_logits[0], float('-inf'))
        for tid in allowed_token_ids:
            if 0 <= tid < mask2.size(0):
                mask2[tid] = adjusted_logits[0][tid]
        p = F.softmax(mask2.unsqueeze(0), dim=-1)
        return torch.multinomial(p, num_samples=1).item()

    p = F.softmax(mask.unsqueeze(0), dim=-1)
    return torch.multinomial(p, num_samples=1).item()


# 保留原有接口，内部调用通用函数
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
    return sample_with_allowed(model, out_ids, topic_id, temperature, device,
                               allowed_token_ids, temp_factor, token_counter,
                               rep_penalty, skip_tokens)


# ==================== 原有采样函数 ====================

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


# ==================== 辅助函数 ====================

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


# ==================== 生成主函数 ====================

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
        use_pingze: 是否启用平仄约束
        pingze_helper: 平仄帮助类
        genre: 体裁 "5" 或 "7"
    """


    line_length = 5 if genre == "5" else 7

    idx = _encode_chinese_prefix(prompt, stoi, device)
    newline_id = stoi.get("\n", None)
    if len(prompt) == 1 and newline_id is not None:
        if idx.size(1) == 0 or int(idx[0, 0].item()) != int(newline_id):
            idx = torch.cat([torch.tensor([[newline_id]], device=device, dtype=torch.long), idx], dim=1)
    
    sentence_end_chars = "，。！？"
    end_token_ids = [stoi.get(ch) for ch in sentence_end_chars if ch in stoi]
    end_token_ids = [tid for tid in end_token_ids if tid is not None]

    out_ids: List[int] = idx[0].tolist()

    line_count = 0
    # 统计已生成的中文字符数（用于平仄和押韵）
    chinese_count = sum(1 for tid in out_ids if '\u4e00' <= itos.get(tid, '') <= '\u9fff')

    used_rhyme_chars = set()  # 【新增】记录已经用作韵脚的字（token id）

    # 押韵相关状态
    determined_rhyme_vowel = rhyme_vowel
    first_rhyme_char = None
    need_determine_rhyme = auto_rhyme and determined_rhyme_vowel is None

    # 平仄模板与首字平仄
    pingze_template = None
    if use_pingze and pingze_helper:
        # 根据起笔字平仄选择模板
        first_char = prompt[0] if prompt else "春"
        first_pingze = pingze_helper.get_pingze(first_char)
        # 平为0，仄为1；若无法判断默认仄起（1）
        start_tone = 0 if first_pingze == 0 else 1
        pingze_template = get_pingze_template(genre, target_lines, start_tone)
        # 预计算平仄到 token 的映射（如果还未设置）
        if pingze_helper.stoi is None:
            pingze_helper.set_stoi(stoi)

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
    
    # 用于避免重复打印警告
    warned_no_rhyme_pingze = False
    
    for _ in range(max_new):
        # 体裁/首颔颈尾的温度调整
        temp_factor = 1.0
        if target_lines == 8:
            if line_count in [2, 3]:
                temp_factor = 0.8
            elif line_count in [4, 5]:
                temp_factor = 1.1
        elif target_lines == 4:
            if line_count in [0, 1]:
                temp_factor = 0.8
            elif line_count in [2, 3]:
                temp_factor = 1.1
        
        # 判断当前即将生成的字是否是句末最后一个字（需要押韵）
        # 注意：chinese_count 是已生成的中文字符数，下一个汉字的位置索引 = chinese_count
        # 当 (chinese_count % line_length) == line_length - 1 时，下一个汉字是当前行的最后一个字
        is_last_char_of_line = (chinese_count % line_length == line_length - 1)
        need_rhyme = is_last_char_of_line and (line_count % 2 == 1)
        
        # ========== 平仄约束：计算期望平仄（仅对汉字位置，标点位置跳过） ==========
        expected_pingze = -1
        allowed_by_pingze = None
        if use_pingze and pingze_template:
            # 关键修复：如果下一个位置是标点（即当前行汉字已满），则跳过平仄约束
            # 条件：chinese_count > 0 且 chinese_count 能被 line_length 整除 => 下一位置是标点
            if chinese_count > 0 and chinese_count % line_length == 0:
                expected_pingze = -1  # 标点，不约束
            else:
                line_idx = chinese_count // line_length
                pos_in_line = chinese_count % line_length
                if line_idx < len(pingze_template) and pos_in_line < line_length:
                    expected_pingze = pingze_template[line_idx][pos_in_line]
                    allowed_by_pingze = pingze_helper.get_tokens_by_pingze(expected_pingze)
        
        # ========== 押韵逻辑 ==========
        allowed_tokens = None
        
        # 如果同时有平仄约束和押韵需求，则取交集；否则只取其中一个
        if need_rhyme and determined_rhyme_vowel is not None:
            rhyme_group = rhyme_helper.get_rhyme_group(determined_rhyme_vowel)

            # 【修改后】排除已经使用过的韵脚字
            rhyme_tokens = []
            for ch in rhyme_group:
                if ch in stoi:
                    tid = stoi[ch]
                    # 跳过已经用作韵脚的字
                    if tid not in used_rhyme_chars:
                        rhyme_tokens.append(tid)
            
            # 如果可用的韵脚字太少，给出警告
            if len(rhyme_tokens) < 3:
                print(f"  [警告] 韵母 '{determined_rhyme_vowel}' 只剩 {len(rhyme_tokens)} 个可用字")

            if allowed_by_pingze is not None:
                # 取交集
                intersection = [tid for tid in rhyme_tokens if tid in allowed_by_pingze]
                if intersection:
                    allowed_tokens = intersection
                else:
                    # 无法同时满足，放弃平仄，仅保留押韵（押韵优先级高于平仄）
                    if not warned_no_rhyme_pingze:
                        print("  [警告] 无法同时满足平仄和押韵，将放弃平仄，仅保留押韵约束")
                        warned_no_rhyme_pingze = True
                    allowed_tokens = rhyme_tokens
            else:
                allowed_tokens = rhyme_tokens
        else:
            # 只有平仄约束
            if allowed_by_pingze is not None:
                allowed_tokens = allowed_by_pingze
        
        # ========== 自动确定韵脚（auto_rhyme 且尚未确定） ==========
        if auto_rhyme and need_determine_rhyme and need_rhyme:
            # 先根据平仄约束（如果有）获取允许集合
            temp_allowed = allowed_by_pingze if allowed_by_pingze else []
            found = False
            for retry in range(5):  # 最多5次尝试
                nxt = sample_with_allowed(
                    model, out_ids, topic_id, temperature, device,
                    temp_allowed, temp_factor, token_counter, rep_penalty, skip_tokens
                )
                nxt_char = itos.get(nxt, "")
                # 必须为汉字
                if not ('\u4e00' <= nxt_char <= '\u9fff'):
                    continue
                vowel = rhyme_helper.get_vowel(nxt_char) if rhyme_helper else None
                if vowel and rhyme_helper.is_rhyme_friendly(vowel, min_chars=30):
                    determined_rhyme_vowel = vowel
                    first_rhyme_char = nxt_char
                    need_determine_rhyme = False
                    found = True
                    print(f"  [押韵] 自动确定韵脚: '{nxt_char}' → {vowel}")
                    # 将这个 token 加入序列
                    out_ids.append(nxt)
                    token_counter[nxt] += 1
                    if '\u4e00' <= itos.get(nxt, '') <= '\u9fff':
                        chinese_count += 1
                    # 检查是否结束（如果生成了标点，增加行数）
                    if nxt in end_token_ids:
                        line_count += 1
                        if line_count >= target_lines:
                            break
                    break
            if not found:
                # 无法确定韵脚，关闭自动押韵
                need_determine_rhyme = False
                print("  [押韵] 警告：无法找到合适的韵脚，将禁用自动押韵")
            continue  # 跳过下面的普通采样，直接进入下一轮循环
        
        # ========== 正常采样（非自动韵脚阶段） ==========
        if allowed_tokens:
            nxt = sample_with_allowed(
                model, out_ids, topic_id, temperature, device,
                allowed_tokens, temp_factor, token_counter,
                rep_penalty, skip_tokens,
                used_rhyme_chars=used_rhyme_chars if need_rhyme else None,  # 【新增】
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
        
        # ========== 更新状态 ==========
        out_ids.append(nxt)
        token_counter[nxt] += 1

        nxt_char = itos.get(nxt, "")  # 【新增】获取字符

        if '\u4e00' <= itos.get(nxt, '') <= '\u9fff':
            chinese_count += 1
        
        # 【新增】如果这是句末押韵位置，且确实是韵脚字，记录下来
        if need_rhyme and determined_rhyme_vowel:
            # 检查这个字是否属于当前韵母组
            rhyme_group = rhyme_helper.get_rhyme_group(determined_rhyme_vowel)
            if nxt_char in rhyme_group:
                used_rhyme_chars.add(nxt)  # 记录 token id
                # 可选：打印调试信息
                # print(f"  [押韵] 已使用韵脚字: '{nxt_char}'")

        # 检查是否结束（遇到结束标点且达到目标行数）
        if nxt in end_token_ids:
            line_count += 1
            if line_count >= target_lines:
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
    p.add_argument("--use_pingze", action="store_true", help="启用平仄约束（基于标准绝句/律诗格式）")
    
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
            print("平仄模式: 已启用（使用标准绝句/律诗格式）")
        else:
            print(f"警告: 找不到 {pingze_dict_path}，无法启用平仄。请先运行 python build_pingze_dict.py")
            args.use_pingze = False

    # 获取主题 id
    topic_id = None
    if model.topic_emb is not None:
        try:
            topic_list, _ = load_topic_vocab(args.topic_vocab)
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