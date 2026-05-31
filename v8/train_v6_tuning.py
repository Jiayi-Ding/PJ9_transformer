# -*- coding: utf-8 -*-
"""
v6 训练与调参记录脚本

用途
- 训练 v6 模型
- 记录每次超参数组合的训练/验证损失
- 用 matplotlib 输出可视化曲线
- 方便老师查看“我们是怎么调参的”

说明
- 默认假设本文件与 v6 的 dataset.py / model.py / prepare_data.py 在同一目录下使用。
- 如果你的项目结构不同，把 DATA_DIR / 相关路径改成实际路径即可。
- 这个脚本保留了“过程记录”的信息：每个 epoch 的 loss、最佳 val_loss、超参数配置、曲线图。

运行示例
python train_v6_tuning.py --genre 5 --use_topic --lr 3e-4 --batch_size 32 --epochs 20 --dropout 0.1 --d_model 256 --n_head 8 --n_layer 4 --block_size 128

如果想做多组调参对比，可以多次运行，或者使用 --sweep 开启小范围搜索。
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import time
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import PoetryBlockDataset, load_vocab_json, load_topic_vocab
from model import CharGPT


# =========================
# 默认路径与超参数起点
# =========================
VOCAB = "vocab.json"
TOPIC_VOCAB = "topic_vocab.json"
LOSS_DIR = "tuning_logs"
CKPT_DIR = "checkpoints"

TRAIN_TEMPLATE = "train_data_{genre}.pt"
VAL_TEMPLATE = "val_data_{genre}.pt"
TOPICS_TRAIN_TEMPLATE = "topics_train_{genre}.pt"
TOPICS_VAL_TEMPLATE = "topics_val_{genre}.pt"
BOUND_TRAIN_TEMPLATE = "boundaries_train_{genre}.pt"
BOUND_VAL_TEMPLATE = "boundaries_val_{genre}.pt"

DEFAULT_LR = 3e-4
DEFAULT_BATCH_SIZE = 32
DEFAULT_EPOCHS = 20
DEFAULT_DROPOUT = 0.1
DEFAULT_D_MODEL = 256
DEFAULT_N_HEAD = 8
DEFAULT_N_LAYER = 4
DEFAULT_D_FF = 1024
DEFAULT_BLOCK_SIZE = 128
DEFAULT_WEIGHT_DECAY = 0.01
DEFAULT_GRAD_CLIP = 1.0
DEFAULT_SEED = 42
DEFAULT_TRAIN_NUM_SAMPLES = 50000
DEFAULT_VAL_BATCHES = 200
DEFAULT_LOG_INTERVAL = 50


@dataclass
class RunConfig:
    genre: str
    use_topic: bool
    lr: float
    batch_size: int
    epochs: int
    dropout: float
    d_model: int
    n_head: int
    n_layer: int
    d_ff: int
    block_size: int
    weight_decay: float = DEFAULT_WEIGHT_DECAY
    grad_clip: float = DEFAULT_GRAD_CLIP
    seed: int = DEFAULT_SEED
    train_num_samples: int = DEFAULT_TRAIN_NUM_SAMPLES
    val_batches: int = DEFAULT_VAL_BATCHES
    log_interval: int = DEFAULT_LOG_INTERVAL


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def collate_with_topic(batch):
    x_list, y_list, topic_list = zip(*batch)
    x = torch.stack(x_list, dim=0)
    y = torch.stack(y_list, dim=0)
    topic_ids = torch.tensor(topic_list, dtype=torch.long)
    return x, y, topic_ids


def collate_without_topic(batch):
    if len(batch[0]) == 3:
        x_list, y_list, _ = zip(*batch)
    else:
        x_list, y_list = zip(*batch)
    x = torch.stack(x_list, dim=0)
    y = torch.stack(y_list, dim=0)
    return x, y, None


@torch.no_grad()
def eval_loss(model: nn.Module, loader: DataLoader, device: torch.device, max_batches: int, use_topic: bool) -> float:
    model.eval()
    total_loss, total_count = 0.0, 0
    for step, batch in enumerate(loader):
        if max_batches > 0 and step >= max_batches:
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

        total_loss += loss.item() * x.size(0)
        total_count += x.size(0)

    return total_loss / max(total_count, 1)


def build_dataloaders(cfg: RunConfig, vocab_path: str, topic_vocab_path: str):
    train_pt = TRAIN_TEMPLATE.format(genre=cfg.genre)
    val_pt = VAL_TEMPLATE.format(genre=cfg.genre)

    if not os.path.isfile(train_pt) or not os.path.isfile(val_pt) or not os.path.isfile(vocab_path):
        raise FileNotFoundError("缺少训练所需数据文件，请先运行 prepare_data.py")

    if cfg.use_topic and not os.path.isfile(topic_vocab_path):
        raise FileNotFoundError("use_topic=True 但找不到 topic_vocab.json")

    if cfg.use_topic:
        train_ds = PoetryBlockDataset(
            train_pt,
            cfg.block_size,
            topics_path=TOPICS_TRAIN_TEMPLATE.format(genre=cfg.genre),
            boundaries_path=BOUND_TRAIN_TEMPLATE.format(genre=cfg.genre),
            num_samples=cfg.train_num_samples,
            sample_random=True,
        )
        val_ds = PoetryBlockDataset(
            val_pt,
            cfg.block_size,
            topics_path=TOPICS_VAL_TEMPLATE.format(genre=cfg.genre),
            boundaries_path=BOUND_VAL_TEMPLATE.format(genre=cfg.genre),
            num_samples=None,
            sample_random=False,
        )
        collate_fn = collate_with_topic
    else:
        train_ds = PoetryBlockDataset(
            train_pt,
            cfg.block_size,
            num_samples=cfg.train_num_samples,
            sample_random=True,
        )
        val_ds = PoetryBlockDataset(
            val_pt,
            cfg.block_size,
            num_samples=None,
            sample_random=False,
        )
        collate_fn = collate_without_topic

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        drop_last=True,
        num_workers=0,
        collate_fn=collate_fn,
    )
    return train_loader, val_loader


def build_model(cfg: RunConfig, vocab_size: int, num_topics: int) -> CharGPT:
    return CharGPT(
        vocab_size=vocab_size,
        block_size=cfg.block_size,
        d_model=cfg.d_model,
        n_head=cfg.n_head,
        n_layer=cfg.n_layer,
        d_ff=cfg.d_ff,
        dropout=cfg.dropout,
        num_topics=num_topics if cfg.use_topic else 0,
    )


def train_one_run(
    cfg: RunConfig,
    vocab_path: str = VOCAB,
    topic_vocab_path: str = TOPIC_VOCAB,
    out_dir: str = LOSS_DIR,
    ckpt_dir: str = CKPT_DIR,
) -> Dict:
    ensure_dir(out_dir)
    ensure_dir(ckpt_dir)
    set_seed(cfg.seed)
    device = get_device()

    _, _, vocab_size = load_vocab_json(vocab_path)
    num_topics = 0
    topic_list = []
    if cfg.use_topic:
        topic_list, num_topics = load_topic_vocab(topic_vocab_path)

    train_loader, val_loader = build_dataloaders(cfg, vocab_path, topic_vocab_path)
    model = build_model(cfg, vocab_size, num_topics).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    history: List[Dict] = []
    best_val = float("inf")
    best_epoch = -1

    t0 = time.time()
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        running_loss, running_count = 0.0, 0

        for step, batch in enumerate(train_loader, start=1):
            if cfg.use_topic:
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

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()

            running_loss += loss.item() * x.size(0)
            running_count += x.size(0)

            if cfg.log_interval > 0 and step % cfg.log_interval == 0:
                print(
                    f"[epoch {epoch}/{cfg.epochs}] step={step} "
                    f"loss={loss.item():.4f} lr={cfg.lr:.2e}"
                )

        train_loss = running_loss / max(running_count, 1)
        val_loss = eval_loss(model, val_loader, device, cfg.val_batches, cfg.use_topic)

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "lr": cfg.lr,
            "batch_size": cfg.batch_size,
            "dropout": cfg.dropout,
            "d_model": cfg.d_model,
            "n_head": cfg.n_head,
            "n_layer": cfg.n_layer,
            "block_size": cfg.block_size,
        }
        history.append(row)
        print(
            f"[epoch {epoch}/{cfg.epochs}] train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} best_val={best_val:.4f if best_val < float('inf') else float('nan')}"
        )

        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            ckpt_path = os.path.join(
                ckpt_dir,
                f"best_genre{cfg.genre}_bs{cfg.batch_size}_lr{cfg.lr}_dm{cfg.d_model}_nl{cfg.n_layer}.pt",
            )
            torch.save(
                {
                    "model": model.state_dict(),
                    "hparams": {
                        "vocab_size": vocab_size,
                        "block_size": cfg.block_size,
                        "d_model": cfg.d_model,
                        "n_head": cfg.n_head,
                        "n_layer": cfg.n_layer,
                        "d_ff": cfg.d_ff,
                        "dropout": cfg.dropout,
                        "num_topics": num_topics if cfg.use_topic else 0,
                    },
                    "config": asdict(cfg),
                    "best_epoch": best_epoch,
                    "best_val_loss": best_val,
                },
                ckpt_path,
            )

    elapsed = time.time() - t0
    run_name = f"genre{cfg.genre}_bs{cfg.batch_size}_lr{cfg.lr}_dm{cfg.d_model}_nl{cfg.n_layer}_bsz{cfg.block_size}"
    history_csv = os.path.join(out_dir, f"{run_name}_history.csv")
    plot_path = os.path.join(out_dir, f"{run_name}_loss_curve.png")
    meta_path = os.path.join(out_dir, f"{run_name}_meta.json")

    with open(history_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()) if history else [])
        if history:
            writer.writeheader()
            writer.writerows(history)

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "config": asdict(cfg),
                "best_epoch": best_epoch,
                "best_val_loss": best_val,
                "elapsed_seconds": elapsed,
                "history_csv": history_csv,
                "plot_path": plot_path,
                "topic_list": topic_list,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    plot_history(history, plot_path, title=run_name)

    return {
        "run_name": run_name,
        "best_epoch": best_epoch,
        "best_val_loss": best_val,
        "elapsed_seconds": elapsed,
        "history_csv": history_csv,
        "plot_path": plot_path,
        "meta_path": meta_path,
    }


def plot_history(history: List[Dict], out_path: str, title: str = "training") -> None:
    if not history:
        return
    epochs = [r["epoch"] for r in history]
    train_loss = [r["train_loss"] for r in history]
    val_loss = [r["val_loss"] for r in history]

    plt.figure(figsize=(9, 5))
    plt.plot(epochs, train_loss, marker="o", label="train loss")
    plt.plot(epochs, val_loss, marker="o", label="val loss")
    plt.title(title)
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def plot_compare(results: List[Dict], out_path: str) -> None:
    if not results:
        return
    labels = [r["run_name"] for r in results]
    vals = [r["best_val_loss"] for r in results]
    epochs = [r["best_epoch"] for r in results]

    fig, ax1 = plt.subplots(figsize=(11, 5))
    x = list(range(len(labels)))
    ax1.bar(x, vals, color="#38bdf8", alpha=0.85)
    ax1.set_ylabel("best val loss")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=25, ha="right")
    ax1.set_title("hyperparameter tuning comparison")
    ax1.grid(axis="y", alpha=0.2)

    for i, (v, e) in enumerate(zip(vals, epochs)):
        ax1.text(i, v, f"{v:.3f}\nE{e}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="v6 训练调参记录脚本")
    p.add_argument("--genre", type=str, choices=["5", "7", "ci"], default="5")
    p.add_argument("--use_topic", action="store_true")
    p.add_argument("--lr", type=float, default=DEFAULT_LR)
    p.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    p.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    p.add_argument("--dropout", type=float, default=DEFAULT_DROPOUT)
    p.add_argument("--d_model", type=int, default=DEFAULT_D_MODEL)
    p.add_argument("--n_head", type=int, default=DEFAULT_N_HEAD)
    p.add_argument("--n_layer", type=int, default=DEFAULT_N_LAYER)
    p.add_argument("--d_ff", type=int, default=DEFAULT_D_FF)
    p.add_argument("--block_size", type=int, default=DEFAULT_BLOCK_SIZE)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--train_num_samples", type=int, default=DEFAULT_TRAIN_NUM_SAMPLES)
    p.add_argument("--val_batches", type=int, default=DEFAULT_VAL_BATCHES)
    p.add_argument("--log_interval", type=int, default=DEFAULT_LOG_INTERVAL)
    p.add_argument("--sweep", action="store_true", help="运行小范围调参对比")
    p.add_argument("--compare_plot", type=str, default=os.path.join(LOSS_DIR, "compare.png"))
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.sweep:
        sweep_configs = [
            RunConfig(args.genre, args.use_topic, 3e-4, 32, 12, 0.1, 256, 8, 4, 1024, 128, seed=args.seed),
            RunConfig(args.genre, args.use_topic, 1e-4, 32, 12, 0.1, 256, 8, 4, 1024, 128, seed=args.seed),
            RunConfig(args.genre, args.use_topic, 3e-4, 16, 12, 0.2, 256, 8, 4, 1024, 128, seed=args.seed),
            RunConfig(args.genre, args.use_topic, 3e-4, 32, 12, 0.1, 512, 8, 6, 2048, 128, seed=args.seed),
        ]
        results: List[Dict] = []
        for i, cfg in enumerate(sweep_configs, start=1):
            print("\n" + "=" * 80)
            print(f"[调参实验 {i}/{len(sweep_configs)}] {cfg}")
            print("=" * 80)
            res = train_one_run(cfg)
            results.append(res)

        ensure_dir(os.path.dirname(args.compare_plot) or ".")
        plot_compare(results, args.compare_plot)
        print("\n调参对比完成：")
        for r in results:
            print(r)
        print(f"对比图已保存: {os.path.abspath(args.compare_plot)}")
        return

    cfg = RunConfig(
        genre=args.genre,
        use_topic=args.use_topic,
        lr=args.lr,
        batch_size=args.batch_size,
        epochs=args.epochs,
        dropout=args.dropout,
        d_model=args.d_model,
        n_head=args.n_head,
        n_layer=args.n_layer,
        d_ff=args.d_ff,
        block_size=args.block_size,
        seed=args.seed,
        train_num_samples=args.train_num_samples,
        val_batches=args.val_batches,
        log_interval=args.log_interval,
    )
    result = train_one_run(cfg)
    print("\n训练完成，结果如下：")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
