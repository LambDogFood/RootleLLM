"""A single pre-norm transformer block."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn

from ..config import ModelConfig
from .attention import Attention, KVCache
from .feedforward import MoEFeedForward, SwiGLU
from .norm import RMSNorm


class Block(nn.Module):
    """Pre-norm transformer block: ``x + attn(norm(x))`` then ``x + ffn(norm(x))``.

    Pre-normalisation (norm *inside* each residual branch) keeps a clean identity
    path through the network, which is what makes deep transformers trainable.
    The feed-forward is dense SwiGLU, or a sparse MoE when ``cfg.n_experts > 0``.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.dim, cfg.norm_eps)
        self.attn = Attention(cfg)
        self.ffn_norm = RMSNorm(cfg.dim, cfg.norm_eps)
        self.is_moe = cfg.n_experts > 0
        self.ffn = (
            MoEFeedForward(cfg)
            if self.is_moe
            else SwiGLU(cfg.dim, cfg.ffn_hidden_dim, cfg.dropout)
        )

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        kv_cache: Optional[KVCache] = None,
    ) -> Tuple[torch.Tensor, KVCache, Optional[torch.Tensor]]:
        h, new_cache = self.attn(self.attn_norm(x), cos, sin, kv_cache)
        x = x + h
        x = x + self.ffn(self.ffn_norm(x))
        # Return the MoE aux loss as a graph-connected output (rather than reading
        # it off the module afterwards) so it survives a gradient-checkpoint boundary.
        aux = self.ffn.last_aux_loss if self.is_moe else None
        return x, new_cache, aux
