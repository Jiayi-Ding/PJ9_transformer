# -*- coding: utf-8 -*-
"""古诗生成与采样策略模块。

本模块负责加载模型与词表，构建押韵/平仄约束，并执行自适应温度、重复惩罚、
起笔换行、押韵与平仄约束下的诗歌生成。

v13 相较于 v1 的功能更新：
1. 支持主题选择或自动选择主题 embedding。
2. 引入自动押韵、指定押韵、平仄约束。
3. 采样策略包含自适应温度（基于熵）、重复惩罚、跳过换行惩罚。
4. 起笔单字时在前面添加换行符以改善首句结构。
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

from rhyme_utils import RhymeHelper

class PingzeHelper:
    """平仄辅助类，用于在生成时根据字表查找平仄候选 token。"""

    def __init__(self, dict_path: str = "pingze_dict.json"):
        with open(dict_path, 'r', encoding='utf-8') as f:
            self.char_to_pingze = json.load(f)

        self.pingze_to_tokens = {0: [], 1: [], 2: []}
        self.stoi = None

    def set_stoi(self, stoi: Dict[str, int]):
        """初始化词表映射，将平仄分类映射到 token id 列表。"""
        self.stoi = stoi
        self.pingze_to_tokens = {0: [], 1: [], 2: []}
        for ch, tid in stoi.items():
            pz = self.char_to_pingze.get(ch, 2)
            self.pingze_to_tokens[pz].append(tid)

    def get_pingze(self, char: str) -> int:
        """返回单个汉字的平仄类别。"""
        return self.char_to_pingze.get(char, 2)

    def get_tokens_by_pingze(self, expected: int) -> List[int]:
        """返回指定平仄类别的所有 token id 列表。"""
        if self.stoi is None:
            return []
        return self.pingze_to_tokens.get(expected, [])

PINGZE_5_4_PINGQI = [
    [0, 0, 1, 1, 0],
    [1, 1, 0, 0, 1],
    [0, 0, 0, 1, 1],
    [1, 1, 1, 0, 0],
]
PINGZE_5_4_ZEQI = [
    [1, 1, 0, 0, 1],
    [0, 0, 1, 1, 0],
    [0, 0, 0, 1, 1],
    [1, 1, 1, 0, 0],
]

PINGZE_5_8_ZEQI = [
    [1, 1, 0, 0, 1],
    [0, 0, 1, 1, 0],
    [0, 0, 0, 1, 1],
    [1, 1, 1, 0, 0],
    [1, 1, 0, 0, 1],
    [0, 0, 1, 1, 0],
    [0, 0, 0, 1, 1],
    [1, 1, 1, 0, 0],
]

PINGZE_5_8_PINGQI = PINGZE_5_8_ZEQI

PINGZE_7_4_PINGQI = [
    [0, 0, 1, 1, 0, 0, 1],
    [1, 1, 0, 0, 1, 1, 0],
    [1, 1, 0, 0, 0, 1, 1],
    [0, 0, 1, 1, 1, 0, 0],
]
PINGZE_7_4_ZEQI = [
    [1, 1, 0, 0, 0, 1, 1],
    [0, 0, 1, 1, 1, 0, 0],
    [0, 0, 1, 1, 0, 0, 1],
    [1, 1, 0, 0, 1, 1, 0],
]

PINGZE_7_8_ZEQI = [
    [1, 1, 0, 0, 0, 1, 1],
    [0, 0, 1, 1, 1, 0, 0],
    [0, 0, 1, 1, 0, 0, 1],
    [1, 1, 0, 0, 1, 1, 0],
    [1, 1, 0, 0, 0, 1, 1],
    [0, 0, 1, 1, 1, 0, 0],
    [0, 0, 1, 1, 0, 0, 1],
    [1, 1, 0, 0, 1, 1, 0],
]
PINGZE_7_8_PINGQI = PINGZE_7_8_ZEQI

def get_pingze_template(genre: str, target_lines: int, start_pingze: int = 1) -> List[List[int]]:
    """根据体裁和起始平仄返回标准平仄模板。

    体裁包括五言/七言绝句和律诗，返回对应行列的平仄方案。
    start_pingze 控制首字的平仄类型，从而决定全诗平仄格律。
    """

    if genre == "5":
        if target_lines == 4:
            if start_pingze == 0:
                return PINGZE_5_4_PINGQI
            else:
                return PINGZE_5_4_ZEQI
        else:
            if start_pingze == 0:
                return PINGZE_5_8_PINGQI
            else:
                return PINGZE_5_8_ZEQI
    else:
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
    used_rhyme_chars: set = None,
) -> int:
    """在允许 token 集合中采样下一个 token。

    用于平仄或押韵约束场景。若 allowed_token_ids 为空，
    则退回到普通采样逻辑。
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

    if rep_penalty != 1.0 and token_counter:
        for token_id, count in token_counter.items():
            if skip_tokens and token_id in skip_tokens:
                continue
            penalty_factor = 1.0 + (rep_penalty - 1.0) * count
            adjusted_logits[0, token_id] /= penalty_factor

    mask = torch.full_like(adjusted_logits[0], float('-inf'))
    for tid in allowed_token_ids:

        if used_rhyme_chars and tid in used_rhyme_chars:
            continue
        if 0 <= tid < mask.size(0):
            mask[tid] = adjusted_logits[0][tid]

    if not torch.any(mask > float('-inf')):
        # 如果所有允许 token 都被过滤，则恢复到允许列表中的 softmax 采样
        mask2 = torch.full_like(adjusted_logits[0], float('-inf'))
        for tid in allowed_token_ids:
            if 0 <= tid < mask2.size(0):
                mask2[tid] = adjusted_logits[0][tid]
        p = F.softmax(mask2.unsqueeze(0), dim=-1)
        return torch.multinomial(p, num_samples=1).item()

    p = F.softmax(mask.unsqueeze(0), dim=-1)
    return torch.multinomial(p, num_samples=1).item()

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
    """采样下一个 token，支持自适应温度与重复惩罚。

    该函数可用于常规生成，或在无法限制 token 时作为后备采样。
    """

    model.eval()
    if idx.size(1) == 0:
        raise ValueError("序列为空")
    t = min(idx.size(1), model.block_size)
    x = idx[:, -t:].contiguous()

    topic_tensor = None
    if topic_id is not None and model.topic_emb is not None:
        topic_tensor = torch.tensor([topic_id], device=device)
    logits, _ = model(x, topic_ids=topic_tensor)
    last_logits = logits[:, -1, :]

    if use_adaptive:
        # 自适应温度：根据当前预测分布熵调整采样温度，避免过于确定或过于随机
        probs = F.softmax(last_logits / max(base_temp, 1e-6), dim=-1)
        log_probs = torch.log(probs + 1e-8)
        entropy = -(probs * log_probs).sum(dim=-1)

        if entropy.item() > 2.5:
            temp = base_temp * 0.5
        elif entropy.item() > 1.5:
            temp = base_temp * 0.8
        elif entropy.item() < 0.8:
            temp = base_temp * 1.5
        else:
            temp = base_temp
    else:
        temp = temperature

    temp *= temp_factor
    adjusted_logits = last_logits / max(temp, 1e-6)

    if rep_penalty != 1.0 and token_counter:
        penalized_logits = adjusted_logits.clone()
        for token_id, count in token_counter.items():
            if skip_tokens and token_id in skip_tokens:
                continue
            penalty_factor = 1.0 + (rep_penalty - 1.0) * count
            penalized_logits[0, token_id] /= penalty_factor
        adjusted_logits = penalized_logits

    p = F.softmax(adjusted_logits, dim=-1)
    nxt = torch.multinomial(p, num_samples=1).item()
    return int(nxt)

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
    """执行一次完整的古诗生成。

    支持单字起笔自动加换行、自动押韵、指定押韵、平仄约束、
    自适应温度、重复惩罚与行末押韵控制。
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

    chinese_count = sum(1 for tid in out_ids if '\u4e00' <= itos.get(tid, '') <= '\u9fff')

    used_rhyme_chars = set()

    determined_rhyme_vowel = rhyme_vowel
    first_rhyme_char = None
    need_determine_rhyme = auto_rhyme and determined_rhyme_vowel is None

    pingze_template = None
    if use_pingze and pingze_helper:

        first_char = prompt[0] if prompt else "春"
        first_pingze = pingze_helper.get_pingze(first_char)

        start_tone = 0 if first_pingze == 0 else 1
        pingze_template = get_pingze_template(genre, target_lines, start_tone)

        if pingze_helper.stoi is None:
            pingze_helper.set_stoi(stoi)

    token_counter = Counter(out_ids)

    skip_tokens = set()
    if skip_newline_penalty and newline_id is not None:
        skip_tokens.add(newline_id)

    punctuation_chars = "，。！？；：、“”‘’《》【】（）．,.;:?!\"'`~@#$%^&*_+=-—…"
    for ch in punctuation_chars:
        if ch in stoi:
            skip_tokens.add(stoi[ch])

    warned_no_rhyme_pingze = False

    for _ in range(max_new):

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

        is_last_char_of_line = (chinese_count % line_length == line_length - 1)
        need_rhyme = is_last_char_of_line and (line_count % 2 == 1)

        expected_pingze = -1
        allowed_by_pingze = None
        if use_pingze and pingze_template:

            if chinese_count > 0 and chinese_count % line_length == 0:
                expected_pingze = -1
            else:
                line_idx = chinese_count // line_length
                pos_in_line = chinese_count % line_length
                if line_idx < len(pingze_template) and pos_in_line < line_length:
                    expected_pingze = pingze_template[line_idx][pos_in_line]
                    allowed_by_pingze = pingze_helper.get_tokens_by_pingze(expected_pingze)

        allowed_tokens = None

        if need_rhyme and determined_rhyme_vowel is not None:
            rhyme_group = rhyme_helper.get_rhyme_group(determined_rhyme_vowel)

            rhyme_tokens = []
            for ch in rhyme_group:
                if ch in stoi:
                    tid = stoi[ch]

                    if tid not in used_rhyme_chars:
                        rhyme_tokens.append(tid)

            if len(rhyme_tokens) < 3:
                print(f"  [警告] 韵母 '{determined_rhyme_vowel}' 只剩 {len(rhyme_tokens)} 个可用字")

            if allowed_by_pingze is not None:

                intersection = [tid for tid in rhyme_tokens if tid in allowed_by_pingze]
                if intersection:
                    allowed_tokens = intersection
                else:

                    if not warned_no_rhyme_pingze:
                        print("  [警告] 无法同时满足平仄和押韵，将放弃平仄，仅保留押韵约束")
                        warned_no_rhyme_pingze = True
                    allowed_tokens = rhyme_tokens
            else:
                allowed_tokens = rhyme_tokens
        else:

            if allowed_by_pingze is not None:
                allowed_tokens = allowed_by_pingze

        if auto_rhyme and need_determine_rhyme and need_rhyme:

            temp_allowed = allowed_by_pingze if allowed_by_pingze else []
            found = False
            for retry in range(5):
                nxt = sample_with_allowed(
                    model, out_ids, topic_id, temperature, device,
                    temp_allowed, temp_factor, token_counter, rep_penalty, skip_tokens
                )
                nxt_char = itos.get(nxt, "")

                if not ('\u4e00' <= nxt_char <= '\u9fff'):
                    continue
                vowel = rhyme_helper.get_vowel(nxt_char) if rhyme_helper else None
                if vowel and rhyme_helper.is_rhyme_friendly(vowel, min_chars=30):
                    determined_rhyme_vowel = vowel
                    first_rhyme_char = nxt_char
                    need_determine_rhyme = False
                    found = True
                    print(f"  [押韵] 自动确定韵脚: '{nxt_char}' → {vowel}")

                    out_ids.append(nxt)
                    token_counter[nxt] += 1
                    if '\u4e00' <= itos.get(nxt, '') <= '\u9fff':
                        chinese_count += 1

                    if nxt in end_token_ids:
                        line_count += 1
                        if line_count >= target_lines:
                            break
                    break
            if not found:

                need_determine_rhyme = False
                print("  [押韵] 警告：无法找到合适的韵脚，将禁用自动押韵")
            continue

        if allowed_tokens:
            nxt = sample_with_allowed(
                model, out_ids, topic_id, temperature, device,
                allowed_tokens, temp_factor, token_counter,
                rep_penalty, skip_tokens,
                used_rhyme_chars=used_rhyme_chars if need_rhyme else None,
            )
        else:

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

        out_ids.append(nxt)
        token_counter[nxt] += 1

        nxt_char = itos.get(nxt, "")

        if '\u4e00' <= itos.get(nxt, '') <= '\u9fff':
            chinese_count += 1

        if need_rhyme and determined_rhyme_vowel:

            rhyme_group = rhyme_helper.get_rhyme_group(determined_rhyme_vowel)
            if nxt_char in rhyme_group:
                used_rhyme_chars.add(nxt)

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
        return random.choice(topic_list[:-1])

    first_char = prompt[0]

    matched_topics = []
    for topic, kw_list in keywords.items():
        if first_char in kw_list:
            matched_topics.append(topic)

    if matched_topics:

        return random.choice(matched_topics)
    else:

        return random.choice(topic_list[:-1])

def main():
    here = os.path.dirname(os.path.abspath(__file__))
    os.chdir(here)

    p = argparse.ArgumentParser(description="古诗生成 - 支持主题选择和押韵")

    p.add_argument("--rhyme", type=str, default=None,
                   help="指定韵脚（如 an, ang, ong），启用押韵约束")
    p.add_argument("--auto_rhyme", action="store_true",
                   help="自动押韵（根据第一个偶数句末字确定韵脚）")
    p.add_argument("--no_rhyme", action="store_true",
                   help="禁用押韵（覆盖其他押韵参数）")

    p.add_argument("--use_pingze", action="store_true", help="启用平仄约束（基于标准绝句/律诗格式）")

    p.add_argument("--ckpt", type=str, default=None)
    p.add_argument("--genre", type=str, required=True,
                   help="体裁，如：5,7")

    topic_choices = [
        "landscape", "frontier", "homesickness", "historical", "love",
        "objects", "farewell", "time_sorrow", "festival", "palace_grievance",
        "immortal", "zen", "drinking", "elegy", "imperial_exam",
        "war_atrocity", "feminine_life", "reclusion_tourism", "other"
    ]
    p.add_argument("--topic", type=str, choices=topic_choices, default="other",
                   help=f"主题: {', '.join(topic_choices)}")

    p.add_argument("--vocab", type=str, default="vocab.json")
    p.add_argument("--topic_vocab", type=str, default="topic_vocab.json")

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

    if args.ckpt is None:
        args.ckpt = f"ckpt_best_{args.genre}.pt"

    if not os.path.isfile(args.ckpt):
        print(f"错误: 未找到模型文件 {args.ckpt}", file=sys.stderr)
        print(f"请先运行: python train.py --genre {args.genre} [--use_topic]", file=sys.stderr)
        sys.exit(1)

    stoi, itos, _ = load_vocab_json(args.vocab)
    device = get_device()
    model, hp = load_for_generate(args.ckpt, device)

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

    pingze_helper = None
    if args.use_pingze:
        pingze_dict_path = "pingze_dict.json"
        if os.path.isfile(pingze_dict_path):
            pingze_helper = PingzeHelper(pingze_dict_path)
            print("平仄模式: 已启用（使用标准绝句/律诗格式）")
        else:
            print(f"警告: 找不到 {pingze_dict_path}，无法启用平仄。请先运行 python build_pingze_dict.py")
            args.use_pingze = False

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
