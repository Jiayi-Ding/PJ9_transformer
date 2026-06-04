# -*- coding: utf-8 -*-
"""
自回归续写 / 生成 - v6.1 实验版
支持命令行指定熵阈值参数，用于探究自适应温度对生成效果的影响。

使用示例：
    # 使用默认阈值（原始 v6 参数）
    python predict.py --genre 5 --prompt 春 --stop_newline

    # 自定义熵阈值（拉伸区间）
    python predict.py --genre 5 --prompt 春 --stop_newline --entropy_high 3.0 --entropy_mid 1.8 --entropy_low 0.5

    # 关闭自适应温度进行对照
    python predict.py --genre 5 --prompt 春 --no_adaptive
"""

import argparse
import os
import sys
from typing import Dict, List, Tuple, Optional
import json
import random

import torch
import torch.nn.functional as F

from dataset import load_vocab_json, load_topic_vocab
from model import CharGPT
from train import get_device


@torch.no_grad()
def sample_next(
    model: CharGPT,
    idx: torch.Tensor,
    topic_id: Optional[int],
    temperature: float,
    device: torch.device,
    use_adaptive: bool = False,
    base_temp: float = 0.8,
    entropy_high: float = 2.5,    # 新：上阈值，熵大于此值→降温
    entropy_mid: float = 1.5,     # 新：中阈值，分界线
    entropy_low: float = 0.8,     # 新：下阈值，熵小于此值→升温
    verbose: bool = False,        # 新：是否打印每步熵和温度
) -> int:
    """采样下一个 token，支持自适应温度（阈值可配置）"""
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

        # ========== 根据可配置的熵阈值调整温度 ==========
        if entropy.item() > entropy_high:
            temp = base_temp * 0.5      # 很犹豫 → 降温
        elif entropy.item() > entropy_mid:
            temp = base_temp * 0.8      # 中等犹豫 → 微降温
        elif entropy.item() < entropy_low:
            temp = base_temp * 1.5      # 很自信 → 升温
        else:
            temp = base_temp            # 正常 → 不变

        if verbose:
            print(f"熵={entropy.item():.2f}, 温度={temp:.2f}")
    else:
        temp = temperature

    # 用调整后的温度采样
    adjusted_logits = last_logits / max(temp, 1e-6)
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
    use_adaptive: bool = True,
    entropy_high: float = 2.5,
    entropy_mid: float = 1.5,
    entropy_low: float = 0.8,
    verbose: bool = False,
) -> str:
    idx = _encode_chinese_prefix(prompt, stoi, device)
    newline_id = stoi.get("\n", None)
    if len(prompt) == 1 and newline_id is not None:
        if idx.size(1) == 0 or int(idx[0, 0].item()) != int(newline_id):
            idx = torch.cat([torch.tensor([[newline_id]], device=device, dtype=torch.long), idx], dim=1)
    out_ids: List[int] = idx[0].tolist()

    for _ in range(max_new):
        nxt = sample_next(
            model,
            torch.tensor([out_ids], device=device, dtype=torch.long),
            topic_id,
            temperature,
            device,
            use_adaptive=use_adaptive,
            base_temp=temperature,
            entropy_high=entropy_high,
            entropy_mid=entropy_mid,
            entropy_low=entropy_low,
            verbose=verbose,
        )
        out_ids.append(nxt)
        if stop_newline and newline_id is not None and nxt == newline_id:
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
    m.load_state_dict(sd, strict=True)
    m.eval()
    return m, hp


def auto_select_topic(prompt: str, topic_vocab_path: str) -> str:
    """
    根据起笔字的第一个字符，自动选择主题。
    """
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

    p = argparse.ArgumentParser(description="古诗生成 - v6.1 实验版（熵阈值可配置）")
    p.add_argument("--ckpt", type=str, default=None)
    p.add_argument("--genre", type=str, choices=["5", "7", "ci"], required=True, help="体裁: 5/7/ci")
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
    p.add_argument("--temperature", type=float, default=0.8, help="基础温度参数")

    # ========== 新增：熵阈值参数 ==========
    p.add_argument("--no_adaptive", action="store_true",
                   help="关闭自适应温度（等同于 use_adaptive=False）")
    p.add_argument("--entropy_high", type=float, default=2.5,
                   help="熵上阈值：熵大于此值→降温×0.5 (默认2.5)")
    p.add_argument("--entropy_mid", type=float, default=1.5,
                   help="熵中阈值：熵大于此值→微降温×0.8 (默认1.5)")
    p.add_argument("--entropy_low", type=float, default=0.8,
                   help="熵下阈值：熵小于此值→升温×1.5 (默认0.8)")
    p.add_argument("--verbose", action="store_true",
                   help="打印每一步的熵值和调整后的温度")

    p.add_argument("--stop_newline", action="store_true", help="遇到换行符停止")
    args = p.parse_args()

    # 设置 checkpoint 路径
    if args.ckpt is None:
        args.ckpt = f"ckpt_best_{args.genre}.pt"

    if not os.path.isfile(args.ckpt):
        print(f"错误: 未找到模型文件 {args.ckpt}", file=sys.stderr)
        print(f"请先运行: python train.py --genre {args.genre} --use_topic", file=sys.stderr)
        sys.exit(1)

    # 加载词表和主题词表
    stoi, itos, _ = load_vocab_json(args.vocab)
    device = get_device()
    model, hp = load_for_generate(args.ckpt, device)

    # 打印阈值信息
    use_adaptive = not args.no_adaptive
    if use_adaptive:
        print(f"自适应温度已启用，阈值: high={args.entropy_high}, mid={args.entropy_mid}, low={args.entropy_low}")
        # 计算正常区宽度
        normal_width = args.entropy_high - args.entropy_low
        print(f"正常区: [{args.entropy_low}, {args.entropy_high}] 宽度={normal_width:.1f}")
    else:
        print("自适应温度已关闭，使用固定温度")

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
            args.prompt, topic_id,
            args.max_new, args.temperature, args.stop_newline,
            use_adaptive=use_adaptive,
            entropy_high=args.entropy_high,
            entropy_mid=args.entropy_mid,
            entropy_low=args.entropy_low,
            verbose=args.verbose,
        )
        print("\n" + "=" * 40)
        print(text)
        print("=" * 40)
    except Exception as e:
        print(f"生成失败: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
