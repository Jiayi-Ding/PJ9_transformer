"""
在predict的基础上，每轮都询问体裁（5/7/ci），再接受起笔字。
即每次生成前让用户先选择体裁，然后输入起笔字，而不是启动时固定一种体裁。
此种请情况下每次需要加载新的参数，速度较慢
"""

import argparse
import os
import sys
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from dataset import load_vocab_json
from model import CharGPT
from train import get_device


@torch.no_grad()
def sample_next(
    model: CharGPT, idx: torch.Tensor, temperature: float, device: torch.device
) -> int:
    """idx: (1, t)，取最后一个时间步的 logits 采样下一 token。"""
    model.eval()
    if idx.size(1) == 0:
        raise ValueError("序列为空")
    t = min(idx.size(1), model.block_size)
    x = idx[:, -t:].contiguous()
    logits, _ = model(x)
    last = logits[:, -1, :] / max(temperature, 1e-6)
    p = F.softmax(last, dim=-1)
    nxt = torch.multinomial(p, num_samples=1).item()
    return int(nxt)


def _encode_chinese_prefix(prefix: str, stoi: Dict[str, int], device: torch.device) -> torch.Tensor:
    """将 prefix 中每个字符编为 id；跳过 OOV 字符。"""
    ids = []
    for ch in prefix:
        if ch not in stoi:
            print(f"警告: 字符不在词表，已跳过: {repr(ch)}", file=sys.stderr)
            continue
        ids.append(int(stoi[ch]))
    if not ids:
        raise ValueError("编码后无有效字，请换起笔或检查词表")
    return torch.tensor([ids], dtype=torch.long, device=device)


def first_in_vocab_char(s: str, stoi: Dict[str, int]) -> str:
    """仅将输入中第一个在词表内的字作为起笔条件。"""
    t = s.strip()
    if not t:
        raise ValueError("起笔为空")
    for ch in t:
        if ch in stoi:
            if len(t) > 1:
                print(f"（以首字「{ch}」为起笔；其余字不作为上文）", flush=True)
            return ch
    raise ValueError("输入中无在词表内的字，请另试")


def encode_prompt(prompt: str, stoi: Dict[str, int], device: torch.device) -> torch.Tensor:
    return _encode_chinese_prefix(prompt, stoi, device)


@torch.no_grad()
def generate_one(
    model: CharGPT,
    stoi: Dict[str, int],
    itos: Dict[int, str],
    device: torch.device,
    prompt: str,
    max_new: int,
    temperature: float,
    stop_newline: bool,
) -> str:
    """返回「起笔 + 续写」的完整串。若起笔为单字则自动插入换行符引导新行。"""
    idx = encode_prompt(prompt, stoi, device)
    newline_id = stoi.get("\n", None)
    # 单字起笔且词表包含换行符时，前置换行符引导从新行开始
    if len(prompt) == 1 and newline_id is not None:
        if idx.size(1) == 0 or int(idx[0, 0].item()) != int(newline_id):
            idx = torch.cat([torch.tensor([[newline_id]], device=device, dtype=torch.long), idx], dim=1)
    out_ids = idx[0].tolist()
    for _ in range(max_new):
        nxt = sample_next(
            model,
            torch.tensor([out_ids], device=device, dtype=torch.long),
            temperature,
            device,
        )
        out_ids.append(nxt)
        if stop_newline and newline_id is not None and nxt == newline_id:
            break
    return "".join(itos.get(i, "?") for i in out_ids)


def load_model_for_genre(genre: str, models_cache: Dict[str, Tuple[CharGPT, dict]],
                         ckpt_base: Optional[str], device: torch.device) -> Tuple[CharGPT, dict]:
    """从缓存加载或新加载指定体裁的模型。返回 (model, hparams)。"""
    if genre in models_cache:
        return models_cache[genre]
    ckpt_path = f"ckpt_best_{genre}.pt" if ckpt_base is None else ckpt_base
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"未找到模型文件: {ckpt_path}，请先训练体裁 {genre}")
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
    ).to(device)
    m.load_state_dict(sd, strict=True)
    m.eval()
    models_cache[genre] = (m, hp)
    return m, hp


def run_one_line(
    model: CharGPT,
    stoi: Dict[str, int],
    itos: Dict[int, str],
    device: torch.device,
    args,
    prompt_text: str,
) -> None:
    """执行单次生成并打印结果。"""
    raw = (prompt_text or "").strip() or str(args.prompt).strip()
    try:
        if args.full_prompt:
            p = raw
        else:
            p = first_in_vocab_char(raw, stoi)
    except ValueError as e:
        print(e, file=sys.stderr)
        return
    try:
        text = generate_one(
            model,
            stoi,
            itos,
            device,
            p,
            args.max_new,
            args.temperature,
            args.stop_newline,
        )
    except ValueError as e:
        print(e, file=sys.stderr)
        return
    print(text + "\n", flush=True)


def main() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    os.chdir(here)

    p = argparse.ArgumentParser(description="古诗字符级自回归续写（交互式每轮选体裁+起笔字）")
    p.add_argument("--ckpt", type=str, default=None, help="直接指定checkpoint路径，此时忽略体裁选择")
    p.add_argument("--genre", type=str, choices=["5", "7", "ci"], default=None, help="单次模式时指定体裁")
    p.add_argument("--vocab", type=str, default="vocab.json")
    p.add_argument("--once", action="store_true", help="单次生成模式（使用--genre或交互选择一次体裁）")
    p.add_argument("--prompt", type=str, default="春", help="起笔字（单次模式或交互默认值）")
    p.add_argument("--full_prompt", action="store_true", help="将整段输入作为上文（否则仅取首字）")
    p.add_argument("--max_new", type=int, default=200, help="新增长度（字符数）")
    p.add_argument("--temperature", type=float, default=0.9, help=">0，越大越随机")
    p.add_argument("--stop_newline", action="store_true", help="遇到换行符提前停止")
    args = p.parse_args()

    # 加载词表（全局共享）
    stoi, itos, _ = load_vocab_json(args.vocab)
    device = get_device()

    # 单次模式（--once）
    if args.once:
        # 确定体裁
        genre = args.genre
        if genre is None:
            print("请选择体裁 [5/7/ci]：", end="", flush=True)
            genre = input().strip()
            if genre not in ("5", "7", "ci"):
                print("无效体裁，退出。")
                sys.exit(1)
        try:
            model, _ = load_model_for_genre(genre, {}, args.ckpt, device)
        except FileNotFoundError as e:
            print(e, file=sys.stderr)
            sys.exit(1)
        run_one_line(model, stoi, itos, device, args, args.prompt)
        return

    # 交互模式：循环选择体裁 → 输入起笔字
    models_cache = {}
    print("交互续写：每轮先输入体裁(5/7/ci)，再输入起笔字。输入q退出。")
    while True:
        # 选择体裁
        line = input("体裁 (5/7/ci，q退出): ").strip()
        if line.lower() in ("q", "quit", "exit"):
            break
        if line not in ("5", "7", "ci"):
            print("无效体裁，请重新输入。")
            continue
        genre = line
        try:
            model, _ = load_model_for_genre(genre, models_cache, args.ckpt, device)
        except FileNotFoundError as e:
            print(e, file=sys.stderr)
            continue

        # 起笔字
        prompt_line = input("起笔字 (q退出): ").strip()
        if prompt_line.lower() in ("q", "quit", "exit"):
            break
        if not prompt_line and not args.prompt:
            print("起笔字为空，请重新输入。")
            continue
        run_one_line(model, stoi, itos, device, args, prompt_line)


if __name__ == "__main__":
    main()