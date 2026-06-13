"""Rotary Position Embeddings (RoPE; Su et al., 2021).

Positions are encoded by rotating pairs of query/key channels by an angle that
grows with absolute position. Because attention scores depend on the *relative*
rotation between a query and key, RoPE generalises better than learned absolute
embeddings and supports context-length extension via ``scaling_factor``.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch


def precompute_rope_cache(
    head_dim: int,
    max_seq_len: int,
    theta: float,
    scaling_factor: float = 1.0,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Precompute the rotation table.

    Returns ``(cos, sin)``, each of shape ``(max_seq_len, head_dim)``. A
    ``scaling_factor > 1`` performs linear position interpolation, stretching the
    effective context window (Chen et al., 2023).
    """
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(max_seq_len, device=device).float() / scaling_factor
    freqs = torch.outer(t, inv_freq)            # (T, head_dim/2)
    emb = torch.cat((freqs, freqs), dim=-1)     # (T, head_dim)
    return emb.cos(), emb.sin()


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate the two halves of the last dimension: ``[a, b] -> [-b, a]``."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary embeddings to query/key tensors.

    ``q``/``k`` are ``(B, n_heads, T, head_dim)``; ``cos``/``sin`` are ``(T, head_dim)``.
    """
    cos = cos.unsqueeze(0).unsqueeze(0)  # (1, 1, T, head_dim)
    sin = sin.unsqueeze(0).unsqueeze(0)
    q = (q * cos) + (_rotate_half(q) * sin)
    k = (k * cos) + (_rotate_half(k) * sin)
    return q, k
