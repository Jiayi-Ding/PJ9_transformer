# -*- coding: utf-8 -*-
"""
字符级 Transformer 语言模型训练。
兼容：
1. 通用古诗 5/7/ci 训练
2. 词牌训练
3. 主题 embedding
4. 重复项惩罚相关训练开关
"""
from __future__ import annotations

import argparse
import os
import random
import sys
from typing import List

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import PoetryBlockDataset, load_topic_vocab, load_vocab_json
from model import CharGPT
from poetry_config import POETRY_NAME, model_pt, poetry_label, train_pt, val_pt, vocab_json

TRAIN_PT = "train_data.pt"
VAL_PT = "val_data.pt"
VOCAB = "vocab.json"
TOPIC_VOCAB = "topic_vocab.json"
CKPT_OUT = "ckpt_best.pt"
LOSS_PLOT = "loss_curve.png"

BLOCK_SIZE = 128
D_MODEL = 256
N_HEAD = 8
N_LAYER = 4
D_FF = 1024
DROPOUT = 0.1

BATCH_SIZE = 32
LR = 3e-4
EPOCHS = 3
SEED = 42
VAL_BATCHES = 200
TRAIN_NUM_SAMPLES = 50_000
MAX_TRAIN_BATCHES = 0
LOG_INTERVAL = 50

WEIGHT_DECAY = 0.01
GRAD_CLIP_NORM = 1.0

DATALOADER_NUM_WORKERS = 0
DATALOADER_DROP_LAST = True


def set_seed(s: int) -> None:
    random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def eval_loss(model: nn.Module, loader: DataLoader, device: torch.device, max_batches: int = 0, use_topic: bool = False) -> float:
    model.eval()
    total, n = 0.0, 0
    for i, batch in enumerate(loader):
        if max_batches > 0 and i >= max_batches:
            break
        if use_topic and len(batch) == 3:
            x, y, topic_ids = batch
            x, y, topic_ids = x.to(device), y.to(device), topic_ids.to(device)
            _, loss = model(x, y, topic_ids=topic_ids)
        else:
            if len(batch) == 3:
                x, y, _ = batch
            else:
                x, y = batch
            x, y = x.to(device), y.to(device)
            _, loss = model(x, y)
        total += loss.item() * x.size(0)
        n += x.size(0)
    return total / max(n, 1)


def main() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    os.chdir(here)

    p = argparse.ArgumentParser(description="字符级 Transformer 语言模型训练")
    p.add_argument("--train_pt", type=str, default=None)
    p.add_argument("--val_pt", type=str, default=None)
    p.add_argument("--vocab", type=str, default=VOCAB)
    p.add_argument("--topic_vocab", type=str, default=TOPIC_VOCAB)
    p.add_argument("--genre", type=str, choices=["5", "7", "ci"], default="5")
    p.add_argument("--poetry_name", type=str, default=None)
    p.add_argument("--cipai", action="store_true")
    p.add_argument("--save", type=str, default=None)
    p.add_argument("--use_topic", action="store_true", help="启用主题 embedding")
    p.add_argument("--use_repeat_penalty", action="store_true", help="启用重复项惩罚")
    p.add_argument("--repeat_penalty_alpha", type=float, default=1.0, help="重复项惩罚强度")
    p.add_argument("--block_size", type=int, default=BLOCK_SIZE)
    p.add_argument("--d_model", type=int, default=D_MODEL)
    p.add_argument("--n_head", type=int, default=N_HEAD)
    p.add_argument("--n_layer", type=int, default=N_LAYER)
    p.add_argument("--d_ff", type=int, default=D_FF)
    p.add_argument("--dropout", type=float, default=DROPOUT)
    p.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    p.add_argument("--lr", type=float, default=LR)
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--val_batches", type=int, default=VAL_BATCHES)
    p.add_argument("--train_num_samples", type=int, default=TRAIN_NUM_SAMPLES)
    p.add_argument("--max_train_batches", type=int, default=MAX_TRAIN_BATCHES)
    p.add_argument("--log_interval", type=int, default=LOG_INTERVAL)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--log_plot", type=str, default=LOSS_PLOT)
    args = p.parse_args()

    if args.cipai and args.poetry_name is None:
        args.poetry_name = POETRY_NAME

    if args.poetry_name:
        name = args.poetry_name
        args.train_pt = train_pt(name)
        args.val_pt = val_pt(name)
        args.vocab = vocab_json(name)
        if args.save is None:
            args.save = model_pt(name)
        print(f"当前训练词牌：{poetry_label(name)}")
        print(f"词牌前缀：{name}")
    else:
        if args.train_pt is None:
            args.train_pt = f"train_data_{args.genre}.pt"
        if args.val_pt is None:
            args.val_pt = f"val_data_{args.genre}.pt"
        if args.save is None:
            args.save = f"ckpt_best_{args.genre}.pt"
        print(f"选定体裁: {args.genre}")

    for f in (args.train_pt, args.val_pt, args.vocab):
        if not os.path.isfile(f):
            raise FileNotFoundError(f"缺少 {f}，请先运行 prepare_data.py")

    if args.use_topic and not os.path.isfile(args.topic_vocab):
        print("警告: 主题词表不存在，将不使用主题 embedding", file=sys.stderr)
        args.use_topic = False

    print(f"训练数据: {args.train_pt}, 验证数据: {args.val_pt}, 保存模型: {args.save}")
    print(f"启用主题: {args.use_topic}, 启用重复惩罚: {args.use_repeat_penalty}")

    set_seed(args.seed)
    device = get_device()
    print("设备:", device)

    _, __, vocab_size = load_vocab_json(args.vocab)
    num_topics = 0
    if args.use_topic:
        _, num_topics = load_topic_vocab(args.topic_vocab)

    train_n = int(args.train_num_samples) if int(args.train_num_samples) > 0 else None
    train_ds = PoetryBlockDataset(
        args.train_pt,
        args.block_size,
        topics_path=(f"topics_train_{args.genre}.pt" if args.use_topic and not args.poetry_name else None),
        boundaries_path=(f"boundaries_train_{args.genre}.pt" if args.use_topic and not args.poetry_name else None),
        num_samples=train_n,
        sample_random=True,
    )
    val_ds = PoetryBlockDataset(
        args.val_pt,
        args.block_size,
        topics_path=(f"topics_val_{args.genre}.pt" if args.use_topic and not args.poetry_name else None),
        boundaries_path=(f"boundaries_val_{args.genre}.pt" if args.use_topic and not args.poetry_name else None),
        num_samples=None,
        sample_random=False,
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=DATALOADER_NUM_WORKERS, drop_last=DATALOADER_DROP_LAST)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=DATALOADER_NUM_WORKERS, drop_last=DATALOADER_DROP_LAST)

    model = CharGPT(
        vocab_size=vocab_size,
        block_size=args.block_size,
        d_model=args.d_model,
        n_head=args.n_head,
        n_layer=args.n_layer,
        d_ff=args.d_ff,
        dropout=args.dropout,
        num_topics=num_topics if args.use_topic else 0,
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY)
    train_losses: List[float] = []
    val_losses: List[float] = []
    best_val = float("inf")

    for ep in range(1, args.epochs + 1):
        model.train()
        run, cnt = 0.0, 0
        bi = 0
        for batch in train_loader:
            if args.max_train_batches > 0 and bi >= args.max_train_batches:
                break
            if args.use_topic and len(batch) == 3:
                x, y, topic_ids = batch
                x, y, topic_ids = x.to(device), y.to(device), topic_ids.to(device)
                opt.zero_grad()
                _, loss = model(x, y, topic_ids=topic_ids)
            else:
                x, y = batch[0], batch[1]
                x, y = x.to(device), y.to(device)
                opt.zero_grad()
                _, loss = model(x, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            opt.step()
            run += loss.item() * x.size(0)
            cnt += x.size(0)
            bi += 1
            if args.log_interval > 0 and bi % args.log_interval == 0:
                print(f"  epoch {ep} step {bi} loss {loss.item():.4f}")
        tr = run / max(cnt, 1)
        vb = eval_loss(model, val_loader, device, max_batches=args.val_batches, use_topic=args.use_topic)
        train_losses.append(tr)
        val_losses.append(vb)
        print(f"Epoch {ep}/{args.epochs}  train_loss={tr:.4f}  val_loss={vb:.4f}")
        if vb < best_val:
            best_val = vb
            torch.save(
                {
                    "model": model.state_dict(),
                    "hparams": {
                        "vocab_size": vocab_size,
                        "block_size": args.block_size,
                        "d_model": args.d_model,
                        "n_head": args.n_head,
                        "n_layer": args.n_layer,
                        "d_ff": args.d_ff,
                        "dropout": args.dropout,
                        "num_topics": num_topics if args.use_topic else 0,
                    },
                },
                args.save,
            )
            print(f"  -> 保存更优模型 val_loss={vb:.4f} 至 {args.save}")

    if train_losses and val_losses:
        plt.figure(figsize=(6, 4))
        plt.plot(range(1, len(train_losses) + 1), train_losses, label="train")
        plt.plot(range(1, len(val_losses) + 1), val_losses, label="val")
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.legend()
        plt.tight_layout()
        plt.savefig(args.log_plot, dpi=150)
        print(f"已保存曲线: {os.path.abspath(args.log_plot)}")
    print(f"训练结束。最佳 val_loss: {best_val:.4f}")


if __name__ == "__main__":
    main()
