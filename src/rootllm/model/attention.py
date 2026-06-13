"""Grouped-Query Attention with a FlashAttention path and a KV cache."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import ModelConfig
from .norm import RMSNorm
from .rope import apply_rope

KVCache = Tuple[torch.Tensor, torch.Tensor]


class Attention(nn.Module):
    """Multi-head self-attention with Grouped-Query Attention (GQA).

    GQA shares each KV head across several query heads (``n_rep`` of them),
    shrinking the KV cache and memory bandwidth at decode time. ``n_kv_heads ==
    n_heads`` recovers standard MHA; ``n_kv_heads == 1`` is MQA.

    Scoring is delegated to :func:`torch.nn.functional.scaled_dot_product_attention`,
    which selects a FlashAttention / memory-efficient / math backend automatically.
    A KV cache makes incremental decoding O(1) per step instead of O(T).
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.head_dim = cfg.head_dim
        self.n_rep = self.n_heads // self.n_kv_heads  # query heads per KV head
        self.dropout = cfg.dropout
        self.use_qk_norm = cfg.use_qk_norm

        self.wq = nn.Linear(cfg.dim, self.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(cfg.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(cfg.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(self.n_heads * self.head_dim, cfg.dim, bias=False)

        # QK-Norm (Henry et al., 2020): RMSNorm q,k per head to curb attention-logit
        # blow-up and stabilise large-scale / low-precision training.
        if self.use_qk_norm:
            self.q_norm = RMSNorm(self.head_dim, cfg.norm_eps)
            self.k_norm = RMSNorm(self.head_dim, cfg.norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        kv_cache: Optional[KVCache] = None,
    ) -> Tuple[torch.Tensor, KVCache]:
        B, T, _ = x.shape

        q = self.wq(x).view(B, T, self.n_heads, self.head_dim)
        k = self.wk(x).view(B, T, self.n_kv_heads, self.head_dim)
        v = self.wv(x).view(B, T, self.n_kv_heads, self.head_dim)

        if self.use_qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        # (B, T, H, hd) -> (B, H, T, hd)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        q, k = apply_rope(q, k, cos, sin)

        # Incremental decoding: append new keys/values to the cached prefix.
        if kv_cache is not None:
            past_k, past_v = kv_cache
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)
        new_cache = (k, v)

        # Expand KV heads to match query heads. (SDPA's enable_gqa exists but an
        # explicit repeat keeps behaviour identical across torch versions.)
        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)

        # Causal masking is only needed when the query block spans multiple
        # positions (training / prefill). A single-token decode step attends to
        # the whole cached prefix, so no mask is required.
        is_causal = kv_cache is None and T > 1
        attn = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=is_causal,
        )

        attn = attn.transpose(1, 2).contiguous().view(B, T, -1)
        return self.wo(attn), new_cache
