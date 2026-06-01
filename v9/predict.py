# -*- coding: utf-8 -*-
"""
自回归续写 / 生成 - 支持主题选择版本

使用示例：
    python predict_v4.py --genre 5 --topic landscape --prompt 春
    python predict_v4.py --genre 7 --topic frontier --prompt 月 --max_new 100
"""
"""【修改 predict.py 中 load_for_generate 函数以解决 torch.compile 导致的 state_dict 键名前缀问题】"""

import argparse
import os
import sys
from typing import Dict, List, Tuple, Optional

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
    use_adaptive: bool = False,   # 是否启用自适应
    base_temp: float = 0.8,        # 基础温度
) -> int:
    """采样下一个 token，支持主题"""
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
    """
    last = logits[:, -1, :] / max(temperature, 1e-6)
    p = F.softmax(last, dim=-1)
    nxt = torch.multinomial(p, num_samples=1).item()
    return int(nxt)
    """
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
        
        # # 可选：打印调试信息
        # print(f"熵={entropy.item():.2f}, 温度={temp:.2f}")
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
) -> str:
    idx = _encode_chinese_prefix(prompt, stoi, device)
    newline_id = stoi.get("\n", None)
    if len(prompt) == 1 and newline_id is not None:
        if idx.size(1) == 0 or int(idx[0, 0].item()) != int(newline_id):
            idx = torch.cat([torch.tensor([[newline_id]], device=device, dtype=torch.long), idx], dim=1)
    out_ids: List[int] = idx[0].tolist()
    
    for _ in range(max_new):
        nxt = sample_next(model, torch.tensor([out_ids], device=device, dtype=torch.long), topic_id, temperature, device, use_adaptive=True, base_temp=temperature)
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


#增加自动判别主题函数
def auto_select_topic(prompt: str, topic_vocab_path: str) -> str:
    """
    根据起笔字的第一个字符，自动选择主题。
    如果字符在某个主题的关键词中 → 返回该主题
    如果字符在多个主题的关键词中 → 返回匹配最多的主题（仍可能平局）
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
        # 有匹配的主题 → 随机选一个匹配的（如果有多个，比如"月"）
        return random.choice(matched_topics)
    else:
        # 没有匹配 → 随机选一个具体主题（排除 other）
        return random.choice(topic_list[:-1])

def main():
    here = os.path.dirname(os.path.abspath(__file__))
    os.chdir(here)

    p = argparse.ArgumentParser(description="古诗生成 - 支持主题选择")
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
    p.add_argument("--temperature", type=float, default=0.8, help="温度参数")
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

    # 获取主题 id
    topic_id = None
    if model.topic_emb is not None:
        try:
            topic_list, _ = load_topic_vocab(args.topic_vocab)
            
            # 检查用户是否在命令行中显式指定了 --topic
            import sys
            user_specified_topic = False
            for i, arg in enumerate(sys.argv):
                if arg == '--topic' and i + 1 < len(sys.argv):
                    user_specified_topic = True
                    break
            
            if user_specified_topic:
                # 用户指定了主题，用指定的
                target_topic = args.topic
            else:
                # 用户没有指定主题，自动选择
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
            args.max_new, args.temperature, args.stop_newline
        )
        print("\n" + "=" * 40)
        print(text)
        print("=" * 40)
    except Exception as e:
        print(f"生成失败: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()