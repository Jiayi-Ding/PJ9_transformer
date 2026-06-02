# -*- coding: utf-8 -*-
"""
自回归续写 / 生成。
兼容：
1. 通用古诗/词生成
2. 词牌生成
3. 主题控制
4. 动态 temperature
5. stop_newline
6. 重复项惩罚相关采样
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from dataset import load_topic_vocab, load_vocab_json
from model import CharGPT
from poetry_config import POETRY_NAME, list_trainable_from_plan, model_pt, poetry_label, vocab_json
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
    repetition_penalty: float = 1.0,
) -> int:
    model.eval()
    if idx.size(1) == 0:
        raise ValueError("序列为空")
    t = min(idx.size(1), model.block_size)
    x = idx[:, -t:].contiguous()
    topic_tensor = None
    if topic_id is not None and getattr(model, "topic_emb", None) is not None:
        topic_tensor = torch.tensor([topic_id], device=device)
    logits, _ = model(x, topic_ids=topic_tensor)
    last_logits = logits[:, -1, :]

    if repetition_penalty and repetition_penalty > 1.0:
        for token_id in set(idx[0].tolist()):
            last_logits[:, token_id] /= repetition_penalty

    if use_adaptive:
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
    repetition_penalty: float = 1.0,
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
            use_adaptive=True,
            base_temp=temperature,
            repetition_penalty=repetition_penalty,
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
        num_topics=int(hp.get("num_topics", 0)),
    ).to(device)
    m.load_state_dict(sd, strict=True)
    m.eval()
    return m, hp


def auto_select_topic(prompt: str, topic_vocab_path: str) -> str:
    import json
    import random

    with open(topic_vocab_path, "r", encoding="utf-8") as f:
        topic_data = json.load(f)

    keywords = topic_data.get("topic_keywords", {})
    topic_list = topic_data.get(
        "topics",
        [
            "landscape",
            "frontier",
            "homesickness",
            "historical",
            "love",
            "objects",
            "farewell",
            "time_sorrow",
            "festival",
            "palace_grievance",
            "immortal",
            "zen",
            "drinking",
            "elegy",
            "imperial_exam",
            "war_atrocity",
            "feminine_life",
            "reclusion_tourism",
            "other",
        ],
    )

    if not prompt:
        return random.choice(topic_list[:-1])

    first_char = prompt[0]
    matched_topics = []
    for topic, kw_list in keywords.items():
        if first_char in kw_list:
            matched_topics.append(topic)
    if matched_topics:
        return random.choice(matched_topics)
    return random.choice(topic_list[:-1])


def main() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    os.chdir(here)

    topic_choices = [
        "landscape", "frontier", "homesickness", "historical", "love",
        "objects", "farewell", "time_sorrow", "festival", "palace_grievance",
        "immortal", "zen", "drinking", "elegy", "imperial_exam",
        "war_atrocity", "feminine_life", "reclusion_tourism", "other",
    ]
    p = argparse.ArgumentParser(description="古诗生成")
    p.add_argument("--ckpt", type=str, default=None)
    p.add_argument("--genre", type=str, choices=["5", "7", "ci"], default=None)
    p.add_argument("--list-cipai", action="store_true")
    p.add_argument("--poetry_name", type=str, default=None)
    p.add_argument("--cipai", action="store_true")
    p.add_argument("--topic", type=str, choices=topic_choices, default="other",
                   help=f"主题: {', '.join(topic_choices)}")
    p.add_argument("--vocab", type=str, default=None)
    p.add_argument("--topic_vocab", type=str, default="topic_vocab.json")
    p.add_argument("--once", action="store_true")
    p.add_argument("--prompt", type=str, default="春")
    p.add_argument("--full_prompt", action="store_true")
    p.add_argument("--max_new", type=int, default=200)
    p.add_argument("--temperature", type=float, default=0.9)
    p.add_argument("--stop_newline", action="store_true")
    p.add_argument("--use_repeat_penalty", action="store_true")
    p.add_argument("--repetition_penalty", type=float, default=1.15)
    args = p.parse_args()

    if args.list_cipai:
        items = list_trainable_from_plan()
        if not items:
            print("未找到词牌计划文件", file=sys.stderr)
            sys.exit(1)
        print("可训练词牌 / 合并组（--poetry_name 使用右侧前缀）：")
        for x in items:
            print(f"  [{x['tier']}] {x['label']}  ->  {x['poetry_name']}")
        sys.exit(0)

    if args.cipai and args.poetry_name is None:
        args.poetry_name = POETRY_NAME

    if args.poetry_name:
        name = args.poetry_name
        if args.ckpt is None:
            args.ckpt = model_pt(name)
        if args.vocab is None:
            args.vocab = vocab_json(name)
        print(f"当前加载词牌：{poetry_label(name)}")
        print(f"当前加载模型：{args.ckpt}")
    else:
        if args.genre is None:
            if args.once:
                args.genre = "5"
            else:
                while True:
                    prompt = input("请选择体裁 [5/7/ci]：").strip()
                    if prompt in ("5", "7", "ci"):
                        args.genre = prompt
                        break
                    print("无效体裁，请输入 5、7 或 ci。", file=sys.stderr)
        if args.ckpt is None:
            args.ckpt = f"ckpt_best_{args.genre}.pt"
        if args.vocab is None:
            args.vocab = "vocab.json"
        print(f"当前加载模型：{args.ckpt}")

    if not os.path.isfile(args.ckpt):
        print("未找到:", args.ckpt, file=sys.stderr)
        sys.exit(1)

    stoi, itos, _ = load_vocab_json(args.vocab)
    device = get_device()
    model, _hp = load_for_generate(args.ckpt, device)

    topic_id = None
    if getattr(model, "topic_emb", None) is not None:
        try:
            topic_list, _ = load_topic_vocab(args.topic_vocab)
            if args.topic and args.topic != "other":
                target_topic = args.topic
            else:
                target_topic = auto_select_topic(args.prompt, args.topic_vocab)
                print(f"自动选择主题: {target_topic}")
            if target_topic in topic_list:
                topic_id = topic_list.index(target_topic)
        except Exception as e:
            print(f"警告: 主题选择失败: {e}")

    print(f"体裁/模式: {args.genre if args.poetry_name is None else args.poetry_name}, 起笔字: {args.prompt}")
    try:
        prompt = args.prompt if args.full_prompt else first_in_vocab_char(args.prompt, stoi)
        text = generate_one(
            model,
            stoi,
            itos,
            device,
            prompt,
            topic_id,
            args.max_new,
            args.temperature,
            args.stop_newline,
            repetition_penalty=(args.repetition_penalty if args.use_repeat_penalty else 1.0),
        )
        print("\n" + "=" * 40)
        print(text)
        print("=" * 40)
    except Exception as e:
        print(f"生成失败: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
