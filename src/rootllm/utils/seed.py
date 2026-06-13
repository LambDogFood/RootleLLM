"""Reproducibility helpers."""

from __future__ import annotations

import random

import torch


def set_seed(seed: int) -> None:
    """Seed Python, NumPy (if present), and torch RNGs for reproducible runs."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        import numpy as np

        np.random.seed(seed % (2**32))
    except ImportError:
        pass
