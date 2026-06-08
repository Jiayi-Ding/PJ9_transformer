# -*- coding: utf-8 -*-
"""
字符级 Transformer 语言模型训练。
依赖：同目录下已运行 prepare_data.py 生成数据；model.py、dataset.py。

【v5 主题 embedding 改动 4/4】
修改位置：
1. 新增命令行参数: --use_topic（是否启用主题 embedding）
2. 新增命令行参数: --topic_vocab（主题词表路径）
3. 加载训练/验证集时，同时加载对应的主题标签文件
4. 修改 DataLoader 的 collate_fn 或直接处理 DataSet 返回的 (x, y, topic_id)
5. 训练循环中将 topic_ids 传给 model.forward()
6. 保存 checkpoint 时增加 num_topics 字段
"""

"""
【v7改动】
1.训练提速(提升训练速度)    
   ① 混合精度训练（AMP）：使用 torch.autocast + GradScaler，在 CUDA 上自动使用 FP16 计算，显存减半，训练速度提升 1.5~2 倍。
   ② DataLoader 多进程 + pin_memory：num_workers=4（GPU 时），pin_memory=True，消除数据加载 I/O 瓶颈，加速 20~40%。
   ③ AdamW 融合版本：optimizer 设置 fused=True（仅 CUDA），加速参数更新 5~10%。
   ④ 矩阵乘法精度优化：torch.set_float32_matmul_precision('high')，利用 Tensor Core 加速 FP32 矩阵乘，提速 10~20%。
   ⑤ torch.compile 模型编译：使用 torch.compile 对模型进行优化，加速 10~30%（仅 PyTorch 2.0+ 且 CUDA 有效）。
效果：以“!python train.py --genre 5 --use_topic --epochs 3 --batch_size 32 --train_num_samples 50000 --save /kaggle/working/ckpt_best_5.pt”为例，需要2-3分钟
   
2.学习率递减(相同训练轮次下提升训练效果)
   - 线性预热 + 余弦退火 (OneCycleLR)：前 10% 步数线性增加到 max_lr，后 90% 步数余弦衰减到 max_lr/100。
     该调度器可稳定训练初期，并帮助模型在后期精细收敛，在相同 epoch 下获得更低的验证损失。
"""
import argparse
import os
import random
import sys
from contextlib import nullcontext
from typing import List, Optional

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import PoetryBlockDataset, load_vocab_json, load_topic_vocab
from model import CharGPT

# =============================================================================
# 默认可调超参：优先在此修改。下方 argparse 的 default 会引用这些常量。
# =============================================================================

# ---- 数据与文件路径 ----
TRAIN_PT = "train_data.pt"
VAL_PT = "val_data.pt"
VOCAB = "vocab.json"
TOPIC_VOCAB = "topic_vocab.json"      # 新增：主题词表
CKPT_OUT = "ckpt_best.pt"
LOSS_PLOT = "loss_curve.png"

# ---- 模型结构 ----
BLOCK_SIZE = 128
D_MODEL = 256
N_HEAD = 8
N_LAYER = 4
D_FF = 1024
DROPOUT = 0.1

# ---- 训练过程 ----
'''批量大小'''
BATCH_SIZE = 32
'''学习率'''
LR = 3e-4              
EPOCHS = 3
SEED = 42
VAL_BATCHES = 200
TRAIN_NUM_SAMPLES = 50_000
MAX_TRAIN_BATCHES = 0
LOG_INTERVAL = 50

# ---- 优化与数值稳定 ----
WEIGHT_DECAY = 0.01
GRAD_CLIP_NORM = 1.0

# ---- DataLoader 行为 ----
DATALOADER_NUM_WORKERS = 0
DATALOADER_DROP_LAST = True

# =============================================================================


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


def collate_fn_with_topic(batch):
    """
    自定义 collate 函数，处理 (x, y, topic_id) 三元组
    """
    x_list, y_list, topic_list = zip(*batch)
    x = torch.stack(x_list, dim=0)
    y = torch.stack(y_list, dim=0)
    topic_ids = torch.tensor(topic_list, dtype=torch.long)
    return x, y, topic_ids


def collate_fn_without_topic(batch):
    """
    兼容无主题的情况
    """
    # 如果 batch 是三元组，取前两个
    if len(batch[0]) == 3:
        x_list, y_list, _ = zip(*batch)
    else:
        x_list, y_list = zip(*batch)
    x = torch.stack(x_list, dim=0)
    y = torch.stack(y_list, dim=0)
    return x, y, None


@torch.no_grad()
def eval_loss(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    max_batches: int = 0,
    use_topic: bool = False,
) -> float:
    """评估损失函数，支持主题"""
    model.eval()
    total, n = 0.0, 0
    for i, batch in enumerate(loader):
        if max_batches > 0 and i >= max_batches:
            break
        
        if use_topic and len(batch) == 3:
            x, y, topic_ids = batch
            x, y = x.to(device), y.to(device)
            topic_ids = topic_ids.to(device)
            _, loss = model(x, y, topic_ids=topic_ids)
        else:
            # 兼容无主题或二元组
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

    p = argparse.ArgumentParser(description="字符级 Transformer 语言模型训练（支持主题 embedding）")
    p.add_argument("--train_pt", type=str, default=None)
    p.add_argument("--val_pt", type=str, default=None)
    p.add_argument("--vocab", type=str, default=VOCAB)
    p.add_argument("--topic_vocab", type=str, default=TOPIC_VOCAB)
    p.add_argument("--genre", type=str, default="5",
               help="体裁或词牌名，支持：5,7,ci,浣溪沙,水调歌头,鹧鸪天,菩萨蛮,临江仙,满江红")
    p.add_argument("--save", type=str, default=None)
    p.add_argument("--use_topic", action="store_true", help="启用主题 embedding")
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

    # 设置默认路径
    if args.train_pt is None:
        args.train_pt = f"train_data_{args.genre}.pt"
    if args.val_pt is None:
        args.val_pt = f"val_data_{args.genre}.pt"
    if args.save is None:
        args.save = f"ckpt_best_{args.genre}.pt"

    # 检查文件存在性
    for f in (args.train_pt, args.val_pt, args.vocab):
        if not os.path.isfile(f):
            raise FileNotFoundError(f"缺少 {f}，请先运行 prepare_data.py")
    
    if args.use_topic and not os.path.isfile(args.topic_vocab):
        print(f"警告: 主题词表不存在，将不使用主题 embedding", file=sys.stderr)
        args.use_topic = False

    print(f"选定体裁: {args.genre}")
    print(f"启用主题: {args.use_topic}")
    print(f"训练数据: {args.train_pt}, 验证数据: {args.val_pt}")
    print(f"保存模型: {args.save}")

    set_seed(args.seed)
    device = get_device()
    print("设备:", device)

    # 优化9：矩阵乘法精度优化（仅对 CUDA 生效）
    if device.type == 'cuda':
        torch.set_float32_matmul_precision('high')

    # 加载词表和主题信息
    _, __, vocab_size = load_vocab_json(args.vocab)
    
    num_topics = 0
    if args.use_topic:
        try:
            topic_list, num_topics = load_topic_vocab(args.topic_vocab)
            print(f"主题数量: {num_topics}, 主题列表: {topic_list}")
        except Exception as e:
            print(f"加载主题词表失败: {e}，将不使用主题", file=sys.stderr)
            args.use_topic = False

    # 创建数据集
    print("正在加载训练/验证张量...")
    
    if args.use_topic:
        topics_train_path = f"topics_train_{args.genre}.pt"
        topics_val_path = f"topics_val_{args.genre}.pt"
        boundaries_train_path = f"boundaries_train_{args.genre}.pt"  # 新增
        boundaries_val_path = f"boundaries_val_{args.genre}.pt"      # 新增
        
        train_ds = PoetryBlockDataset(
            args.train_pt, args.block_size,
            topics_path=topics_train_path,
            boundaries_path=boundaries_train_path,
            num_samples=args.train_num_samples if args.train_num_samples > 0 else None,
            sample_random=True
        )
        val_ds = PoetryBlockDataset(
            args.val_pt, args.block_size,
            topics_path=topics_val_path,
            boundaries_path=boundaries_val_path,
            num_samples=None, sample_random=False
        )
        collate_fn = collate_fn_with_topic
    else:
        train_ds = PoetryBlockDataset(
            args.train_pt, args.block_size,
            num_samples=args.train_num_samples if args.train_num_samples > 0 else None,
            sample_random=True
        )
        val_ds = PoetryBlockDataset(
            args.val_pt, args.block_size,
            num_samples=None, sample_random=False
        )
        collate_fn = collate_fn_without_topic

    # 优化2：DataLoader 多进程 + pin_memory
    num_workers = 4 if device.type == 'cuda' else 0   # GPU 训练时用 4 进程，CPU 训练时用 0
    pin_memory = True if device.type == 'cuda' else False

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory,
        drop_last=DATALOADER_DROP_LAST, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
        drop_last=DATALOADER_DROP_LAST, collate_fn=collate_fn
    )

    # 创建模型
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

    # 优化3：torch.compile 模型编译（PyTorch 2.0+，仅 CUDA 有效）
    if hasattr(torch, 'compile') and device.type == 'cuda':
        try:
            print("正在使用 torch.compile 编译模型（首次编译耗时较长）...")
            model = torch.compile(model, mode='reduce-overhead')
            print("torch.compile 编译完成")
        except Exception as e:
            print(f"torch.compile 编译失败，跳过: {e}", file=sys.stderr)

    # 优化1：混合精度训练（AMP）—— 使用新 API 避免 FutureWarning
    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None

    # 优化5：AdamW 融合版本 (fused=True)
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=WEIGHT_DECAY,
        fused=True if device.type == 'cuda' else False
    )

    # # ========== 学习率调度：线性预热 + 余弦退火 (OneCycleLR) ==========
    # # 计算总训练步数（考虑 max_train_batches 截断）
    # if args.max_train_batches > 0:
    #     steps_per_epoch = min(len(train_loader), args.max_train_batches)
    # else:
    #     steps_per_epoch = len(train_loader)
    # total_steps = args.epochs * steps_per_epoch

    # # 使用 OneCycleLR 实现前 pct_start 比例步数线性预热，后余弦退火到 max_lr/100
    # scheduler = torch.optim.lr_scheduler.OneCycleLR(
    #     opt,
    #     max_lr=args.lr,
    #     total_steps=total_steps,
    #     pct_start=0.1,           # 前 10% 步数预热
    #     anneal_strategy='cos',
    #     final_div_factor=100.0    # 最终学习率为 max_lr / final_div_factor = 3e-6
    # )
    # ================================================================

    train_losses, val_losses = [], []
    best_val = float("inf")

    for ep in range(1, args.epochs + 1):
        model.train()
        total_loss, total_cnt = 0.0, 0
        max_tb = args.max_train_batches if args.max_train_batches > 0 else None
        
        for bi, batch in enumerate(train_loader):
            if max_tb and bi >= max_tb:
                break
            
            if args.use_topic:
                x, y, topic_ids = batch
                x, y, topic_ids = x.to(device), y.to(device), topic_ids.to(device)
                opt.zero_grad()
                # 使用 AMP 上下文
                with torch.autocast(device_type=device.type, dtype=torch.float16) if scaler else nullcontext():
                    _, loss = model(x, y, topic_ids=topic_ids)
            else:
                x, y = batch[0], batch[1]
                if len(batch) == 3:
                    x, y = x, y
                x, y = x.to(device), y.to(device)
                opt.zero_grad()
                with torch.autocast(device_type=device.type, dtype=torch.float16) if scaler else nullcontext():
                    _, loss = model(x, y)
            
            # 反向传播与梯度更新（支持 AMP）
            if scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
                opt.step()
            
            # # 更新学习率调度器（每个参数更新步后）
            # scheduler.step()
            
            total_loss += loss.item() * x.size(0)
            total_cnt += x.size(0)
            
            # if args.log_interval > 0 and (bi + 1) % args.log_interval == 0:
            #     current_lr = scheduler.get_last_lr()[0]
            #     print(f"  epoch {ep} step {bi+1} loss {loss.item():.4f} lr={current_lr:.2e}")
            if args.log_interval > 0 and (bi + 1) % args.log_interval == 0:
                print(f"  epoch {ep} step {bi+1} loss {loss.item():.4f}")

        train_loss = total_loss / max(total_cnt, 1)
        val_loss = eval_loss(model, val_loader, device, args.val_batches, args.use_topic)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        print(f"Epoch {ep}/{args.epochs} train_loss={train_loss:.4f} val_loss={val_loss:.4f}")
        
        if val_loss < best_val:
            best_val = val_loss
            torch.save({
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
            }, args.save)
            print(f"  -> 保存更优模型 val_loss={val_loss:.4f} 至 {args.save}")

    # 绘制 loss 曲线
    if train_losses and val_losses:
        plt.figure(figsize=(6, 4))
        plt.plot(range(1, len(train_losses) + 1), train_losses, label="train")
        plt.plot(range(1, len(val_losses) + 1), val_losses, label="val")

        # 添加标题
        plt.title(f"Loss curve (lr={args.lr})")

        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.legend()
        plt.tight_layout()
        plt.savefig(args.log_plot, dpi=150)
        print(f"已保存曲线: {os.path.abspath(args.log_plot)}")
    
    print(f"训练结束。最佳 val_loss: {best_val:.4f}")


if __name__ == "__main__":
    main()