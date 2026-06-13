"""Feed-forward networks: dense SwiGLU and a sparse Mixture-of-Experts."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import ModelConfig


class SwiGLU(nn.Module):
    """SwiGLU feed-forward (Shazeer, 2020).

    ``down(silu(gate(x)) * up(x))`` — the gated activation that replaced GELU MLPs
    in Llama/PaLM-class models for better quality at matched parameter count.
    """

    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)   # gate
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)   # up
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)   # down
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class MoEFeedForward(nn.Module):
    """Top-k token-choice Mixture-of-Experts (Switch / Mixtral style).

    Each token is routed to its ``top_k`` highest-scoring experts; outputs are a
    weighted sum of those experts' SwiGLU transforms. A load-balancing auxiliary
    loss (cached in :attr:`last_aux_loss` and collected by the top-level model)
    discourages router collapse onto a few experts.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n_experts = cfg.n_experts
        self.top_k = cfg.n_experts_per_token
        self.aux_loss_coef = cfg.moe_aux_loss_coef
        self.gate = nn.Linear(cfg.dim, cfg.n_experts, bias=False)
        self.experts = nn.ModuleList(
            [SwiGLU(cfg.dim, cfg.ffn_hidden_dim, cfg.dropout) for _ in range(cfg.n_experts)]
        )
        # Most recent aux loss; read by Transformer.forward when targets are given.
        self.last_aux_loss: torch.Tensor = torch.tensor(0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        x_flat = x.view(-1, D)                       # (N, D)
        N = x_flat.shape[0]

        logits = self.gate(x_flat)                   # (N, E)
        probs = F.softmax(logits, dim=-1, dtype=torch.float)
        topk_probs, topk_idx = probs.topk(self.top_k, dim=-1)          # (N, k)
        topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True)  # renormalise
        topk_probs = topk_probs.to(x.dtype)

        out = torch.zeros_like(x_flat)
        # Group tokens by expert and run each expert once over its slice. This is
        # vectorised across tokens (no Python loop over the N tokens).
        flat_expert = topk_idx.view(-1)              # (N*k,)
        flat_weight = topk_probs.view(-1)            # (N*k,)
        flat_token = torch.arange(N, device=x.device).repeat_interleave(self.top_k)

        for e in range(self.n_experts):
            mask = flat_expert == e
            if not mask.any():
                continue
            tok_ids = flat_token[mask]
            w = flat_weight[mask].unsqueeze(-1)
            out.index_add_(0, tok_ids, self.experts[e](x_flat[tok_ids]) * w)

        self.last_aux_loss = self._aux_loss(probs, topk_idx)
        return out.view(B, T, D)

    def _aux_loss(self, probs: torch.Tensor, topk_idx: torch.Tensor) -> torch.Tensor:
        """Switch-Transformer load-balancing loss.

        Encourages the fraction of tokens routed to each expert and the mean
        router probability for that expert to both be uniform.
        """
        with torch.no_grad():
            dispatch = F.one_hot(topk_idx, self.n_experts).float().sum(dim=1)  # (N, E)
        tokens_per_expert = dispatch.mean(dim=0)         # (E,)
        router_prob_per_expert = probs.mean(dim=0)       # (E,)
        return (
            self.aux_loss_coef
            * self.n_experts
            * (tokens_per_expert * router_prob_per_expert).sum()
        )
