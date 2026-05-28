# -*- coding: utf-8 -*-
"""
字符级 Decoder-only 语言模型（GPT 式）。
用多头因果自注意力 + FFN 堆叠；**未使用** nn.Transformer 封装。

【v5 主题 embedding 改动 3/4】
修改位置：
1. CharGPT.__init__ 新增参数: num_topics（主题数量，默认 0 表示不使用主题）
2. CharGPT.__init__ 新增 self.topic_emb（如果 num_topics > 0）
3. CharGPT.forward 新增参数 topic_ids（可选，shape: [B] 或 [B, T]）
4. forward 中将 topic_emb 加到 token_emb + pos_emb 上（诗级别：广播到所有位置）
5. checkpoint 保存/加载时兼容无主题的旧模型

原理：每首诗固定一个主题 embedding，与字符 embedding、位置 embedding 相加，
      使模型在生成时始终保持对主题的感知
"""
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
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
        B, T, C = x.size()
        q = self.w_q(x)
        k = self.w_k(x)
        v = self.w_v(x)
        q = q.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / (self.d_head ** 0.5))
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        att = att.masked_fill(mask[None, None, :, :], float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.dropout(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.w_o(y)
        return y


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
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
    def __init__(
        self,
        vocab_size: int,
        block_size: int,
        d_model: int = 256,
        n_head: int = 8,
        n_layer: int = 4,
        d_ff: int = 1024,
        dropout: float = 0.1,
        num_topics: int = 0,           # 新增：主题数量，0 表示不使用主题
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.d_model = d_model
        self.num_topics = num_topics
        
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, block_size, d_model))
        
        # ========== 新增：主题 embedding（诗级别） ==========
        self.topic_emb = None
        if num_topics > 0:
            self.topic_emb = nn.Embedding(num_topics, d_model)
            # 使用较小的初始化，避免主题信号过强
            torch.nn.init.normal_(self.topic_emb.weight, mean=0.0, std=0.01)
        
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, n_head, block_size, d_ff, dropout) for _ in range(n_layer)]
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        # 权重共享
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
        topic_ids: Optional[torch.Tensor] = None,   # 新增：主题 id，shape [B] 或 [B, T]
    ):
        """
        idx: [B, T] 字符 id
        targets: [B, T] 目标字符 id（可选）
        topic_ids: [B] 或 [B, T]，每首诗的主题 id。若为 [B]，则广播到所有时间步
        
        返回: logits [B, T, V], loss (optional)
        """
        b, t = idx.size()
        assert t <= self.block_size, f"长度 {t} 超过 block_size {self.block_size}"
        
        # 字符 embedding + 位置 embedding
        x = self.tok_emb(idx) + self.pos_emb[:, :t, :]
        
        # ========== 新增：添加主题 embedding ==========
        if self.topic_emb is not None and topic_ids is not None:
            # 确保 topic_ids 是 [B] 或 [B, T]
            if topic_ids.dim() == 1:
                # [B] -> [B, 1] -> 广播到 [B, T]
                topic_emb = self.topic_emb(topic_ids).unsqueeze(1)  # [B, 1, D]
                x = x + topic_emb
            else:
                # [B, T] 每个位置不同主题（token 级别，较少用）
                topic_emb = self.topic_emb(topic_ids)  # [B, T, D]
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