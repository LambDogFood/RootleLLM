"""Transformer building blocks.

Importing from here gives you the model and every component it is assembled from,
so blocks can be reused or unit-tested in isolation.
"""

from __future__ import annotations

from .attention import Attention, KVCache
from .block import Block
from .feedforward import MoEFeedForward, SwiGLU
from .norm import RMSNorm
from .rope import apply_rope, precompute_rope_cache
from .transformer import IGNORE_INDEX, Transformer

__all__ = [
    "Attention",
    "KVCache",
    "Block",
    "MoEFeedForward",
    "SwiGLU",
    "RMSNorm",
    "apply_rope",
    "precompute_rope_cache",
    "Transformer",
    "IGNORE_INDEX",
]
