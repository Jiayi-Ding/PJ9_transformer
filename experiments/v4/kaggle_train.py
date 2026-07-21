#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Kaggle 词牌一键训练脚本。

在 Notebook 中运行：
    !python kaggle_train.py

或修改下方 POETRY_NAME 后 exec(open("kaggle_train.py").read())
"""
from __future__ import annotations

import json
import os
import random
import shutil
import sys
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# ========== 配置：切换词牌时改这里 ==========
POETRY_NAME = "huanxisha"  # 如 pusaman, zhegutian, merge_medium
DATASET_SLUG = "pj9-cipai-v3"

BLOCK_SIZE = 128
D_MODEL = 256
N_HEAD = 8
N_LAYER = 4
D_FF = 1024
DROPOUT = 0.1
BATCH_SIZE = 32
LR = 3e-4
EPOCHS = 5
SEED = 42
TRAIN_NUM_SAMPLES = 50_000
VAL_BATCHES = 200
WEIGHT_DECAY = 0.01
GRAD_CLIP_NORM = 1.0


def poetry_label(name: str) -> str:
    plan_path = Path("cipai_plan.json")
    if plan_path.is_file():
        with open(plan_path, "r", encoding="utf-8") as f:
            plan = json.load(f)
        for x in plan.get("solo", []):
            if x.get("poetry_name") == name:
                return x.get("cipai", name)
        for g in plan.get("merge_groups", []):
            if g.get("poetry_name") == name:
                return g.get("label", name)
    return name


def setup_workdir() -> Path:
    work = Path("/kaggle/working")
    if work.exists():
        train_file = f"{POETRY_NAME}_train.pt"
        input_root = Path("/kaggle/input")
        input_candidates = [
            input_root / DATASET_SLUG / "pj9-cipai-v3",
            input_root / DATASET_SLUG,
        ]
        datasets_root = input_root / "datasets"
        if datasets_root.is_dir():
            for user_dir in datasets_root.iterdir():
                if user_dir.is_dir():
                    input_candidates.extend([
                        user_dir / DATASET_SLUG / "pj9-cipai-v3",
                        user_dir / DATASET_SLUG,
                    ])
        src = None
        for cand in input_candidates:
            if (cand / train_file).is_file() or (cand / "train.py").is_file():
                src = cand
                break
        if src is None:
            for p in input_root.rglob(train_file):
                src = p.parent
                break
        if src is not None:
            for item in os.listdir(src):
                s, d = src / item, work / item
                if s.is_file() and not d.exists():
                    shutil.copy2(s, d)
        os.chdir(work)
    else:
        root = Path(__file__).resolve().parent.parent
        dataset = root / "dataset" / "pj9-cipai-v3"
        if dataset.is_dir():
            os.chdir(dataset)
        else:
            os.chdir(root)
    sys.path.insert(0, os.getcwd())
    return Path(os.getcwd())


def set_seed(s: int) -> None:
    random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


@torch.no_grad()
def eval_loss(model, loader, device, max_batches=0):
    model.eval()
    total, n = 0.0, 0
    for i, (x, y) in enumerate(loader):
        if max_batches > 0 and i >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        _, loss = model(x, y)
        total += loss.item() * x.size(0)
        n += x.size(0)
    return total / max(n, 1)


def train_main(device: torch.device) -> str:
    from dataset import PoetryBlockDataset, load_vocab_json
    from model import CharGPT

    label = poetry_label(POETRY_NAME)
    train_pt = f"{POETRY_NAME}_train.pt"
    val_pt = f"{POETRY_NAME}_val.pt"
    vocab = f"{POETRY_NAME}_vocab.json"
    ckpt_out = f"{POETRY_NAME}_model.pt"

    for f in (train_pt, val_pt, vocab):
        if not Path(f).is_file():
            raise FileNotFoundError(
                f"缺少 {f}，请确认 Kaggle Input Dataset 已添加且包含该词牌数据"
            )

    print(f"当前训练词牌：{label} ({POETRY_NAME})")
    print(f"保存模型：{ckpt_out}")

    _, __, vocab_size = load_vocab_json(vocab)
    train_ds = PoetryBlockDataset(train_pt, BLOCK_SIZE, num_samples=TRAIN_NUM_SAMPLES, sample_random=True)
    val_ds = PoetryBlockDataset(val_pt, BLOCK_SIZE, num_samples=None, sample_random=False)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)

    model = CharGPT(
        vocab_size=vocab_size,
        block_size=BLOCK_SIZE,
        d_model=D_MODEL,
        n_head=N_HEAD,
        n_layer=N_LAYER,
        d_ff=D_FF,
        dropout=DROPOUT,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    train_losses: List[float] = []
    val_losses: List[float] = []
    best_val = float("inf")

    for ep in range(1, EPOCHS + 1):
        model.train()
        run, cnt = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            _, loss = model(x, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            opt.step()
            run += loss.item() * x.size(0)
            cnt += x.size(0)
        tr = run / max(cnt, 1)
        vb = eval_loss(model, val_loader, device, max_batches=VAL_BATCHES)
        train_losses.append(tr)
        val_losses.append(vb)
        print(f"Epoch {ep}/{EPOCHS}  train_loss={tr:.4f}  val_loss={vb:.4f}")
        if vb < best_val:
            best_val = vb
            torch.save(
                {
                    "model": model.state_dict(),
                    "hparams": {
                        "vocab_size": vocab_size,
                        "block_size": BLOCK_SIZE,
                        "d_model": D_MODEL,
                        "n_head": N_HEAD,
                        "n_layer": N_LAYER,
                        "d_ff": D_FF,
                        "dropout": DROPOUT,
                    },
                },
                ckpt_out,
            )
            print(f"  -> 保存更优模型至 {ckpt_out}")

    plt.figure(figsize=(6, 4))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label="train")
    plt.plot(range(1, len(val_losses) + 1), val_losses, label="val")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"loss_{POETRY_NAME}.png", dpi=150)
    print("训练完成，最佳 val_loss:", best_val)
    return ckpt_out


@torch.no_grad()
def demo_generate(device: torch.device, ckpt_out: str, start_char: str = "春") -> str:
    from dataset import load_vocab_json
    from model import CharGPT

    stoi, itos, _ = load_vocab_json(f"{POETRY_NAME}_vocab.json")
    pack = torch.load(ckpt_out, map_location=device)
    hp = pack["hparams"]
    model = CharGPT(**hp).to(device)
    model.load_state_dict(pack["model"])
    model.eval()

    newline_id = stoi.get("\n")
    ids = [stoi[start_char]]
    if newline_id is not None:
        ids = [newline_id] + ids
    for _ in range(120):
        x = torch.tensor([ids[-model.block_size :]], device=device)
        logits, _ = model(x)
        p = F.softmax(logits[:, -1, :] / 0.9, dim=-1)
        nxt = torch.multinomial(p, 1).item()
        ids.append(nxt)
        if nxt == newline_id:
            break
    text = "".join(itos[i] for i in ids)
    print(f"\n起笔「{start_char}」:\n{text}")
    return text


def main() -> None:
    work = setup_workdir()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("工作目录:", work)
    print("设备:", device)
    set_seed(SEED)
    ckpt = train_main(device)
    for ch in ("春", "月", "花"):
        demo_generate(device, ckpt, ch)


if __name__ == "__main__":
    main()
