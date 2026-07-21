# -*- coding: utf-8 -*-
"""Transformer 语言模型定义模块。

本模块实现了字符级 GPT 风格模型，用于诗歌生成。
包含因果自注意力、前馈网络、LayerNorm、位置 embedding 以及可选主题 embedding。
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

class CausalSelfAttention(nn.Module):
    """因果自注意力模块，保证模型只能看到当前词及之前词。

    该模块实现多头注意力，并在注意力权重上应用上三角遮罩，
    防止模型在生成时访问未来位置的信息。
    """

    def __init__(
        self,
        d_model: int,
        n_head: int,
        block_size: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        assert d_model % n_head == 0, "d_model 应能被 n_head 整除"
        self.d_model = d_model
        self.n_head = n_head
        self.d_head = d_model // n_head
        self.w_q = nn.Linear(d_model, d_model, bias=True)
        self.w_k = nn.Linear(d_model, d_model, bias=True)
        self.w_v = nn.Linear(d_model, d_model, bias=True)
        self.w_o = nn.Linear(d_model, d_model, bias=True)
        self.dropout = nn.Dropout(dropout)
        self.block_size = block_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        执行自注意力计算，返回与输入同形状的上下文特征。
        """    

        B, T, C = x.size()
        # 1. 投影（QKV）
        q = self.w_q(x)  # (B, T, C)
        k = self.w_k(x)
        v = self.w_v(x)
        
        # 2. 拆头；转换形状为 (B, n_head, T, d_head)
        q = q.view(B, T, self.n_head, self.d_head).transpose(1, 2)  # (B, nH, T, dH)
        k = k.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        
        # 3. 缩放点积
        att = (q @ k.transpose(-2, -1)) * (1.0 / (self.d_head ** 0.5))  # (B, nH, T, T)
        
        # 4. 上三角掩码屏蔽未来位置，保证因果生成
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        att = att.masked_fill(mask[None, None, :, :], float('-inf'))

        # 5. softmax + dropout
        att = F.softmax(att, dim=-1)
        att = self.dropout(att)

        # 6. 加权V
        y = att @ v  # (B, nH, T, dH)

        # 7. 合并头
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # 8. 输出投影
        y = self.w_o(y)
        return y

class FeedForward(nn.Module):
    """Transformer 中的前馈网络子层。
    使用两层线性网络与 GELU 激活，以增强模型的非线性表达能力。
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),       # 升维
            nn.GELU(),                      # 激活函数
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),       # 降维
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class TransformerBlock(nn.Module):
    """单个 Transformer 块，包括自注意力、前馈网络与残差连接。"""

    def __init__(self, d_model: int, n_head: int, block_size: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_head, block_size, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, d_ff, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x

class CharGPT(nn.Module):
    """字符级 GPT 模型，用于古诗生成。

    模型由 token embedding、位置 embedding、可选主题 embedding、
    多层 Transformer Block、LayerNorm 及线性输出层组成。
    """

    def __init__(
        self,
        vocab_size: int,
        block_size: int,
        d_model: int = 256,
        n_head: int = 8,
        n_layer: int = 4,
        d_ff: int = 1024,
        dropout: float = 0.1,
        num_topics: int = 0,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.d_model = d_model
        self.num_topics = num_topics

        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, block_size, d_model))

        self.topic_emb = None
        if num_topics > 0:
            self.topic_emb = nn.Embedding(num_topics, d_model)
            torch.nn.init.normal_(self.topic_emb.weight, mean=0.0, std=0.01)

        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, n_head, block_size, d_ff, dropout) for _ in range(n_layer)]
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        self.lm_head.weight = self.tok_emb.weight
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            torch.nn.init.zeros_(m.bias)

    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        topic_ids: Optional[torch.Tensor] = None,
    ):
        """前向计算。

        参数:
          idx: 输入 token id 张量，形状 (B, T)。
          targets: 可选目标 token id 張量，用于计算交叉熵损失。
          topic_ids: 可选主题 id 张量，用于叠加主题 embedding。
        返回:
          logits: 形状 (B, T, vocab_size) 的预测得分。
          loss: 可选交叉熵损失。
        """

        b, t = idx.size()
        assert t <= self.block_size, f"长度 {t} 超过 block_size {self.block_size}"

        x = self.tok_emb(idx) + self.pos_emb[:, :t, :]

        if self.topic_emb is not None and topic_ids is not None:
            if topic_ids.dim() == 1:
                topic_emb = self.topic_emb(topic_ids).unsqueeze(1)
                x = x + topic_emb
            else:
                topic_emb = self.topic_emb(topic_ids)
                x = x + topic_emb

        x = self.drop(x)

        for blk in self.blocks:
            x = blk(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, self.vocab_size), targets.view(-1))
        return logits, loss
