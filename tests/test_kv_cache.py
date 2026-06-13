"""Incremental decoding with the KV cache must match a single full forward pass.

This is the load-bearing correctness check for attention + RoPE + caching: if the
cache, position offsets, or masking were wrong, the per-token logits would drift
from the full-sequence logits.
"""

from __future__ import annotations

import torch

from rootllm.model import Transformer
from tests.helpers import make_model_config


def test_incremental_matches_full_forward():
    cfg = make_model_config(dropout=0.0)
    model = Transformer(cfg).eval()
    seq = torch.randint(0, cfg.vocab_size, (2, 10))

    # Full forward: causal attention over the whole sequence at once.
    with torch.no_grad():
        full_logits, _ = model(seq)
    full_last = full_logits[:, -1, :]

    # Incremental: feed one token at a time, growing the KV cache.
    caches = [None] * cfg.n_layers
    logits = None
    with torch.no_grad():
        for t in range(seq.shape[1]):
            logits, caches = model(seq[:, t:t + 1], kv_caches=caches, start_pos=t)
    incr_last = logits[:, -1, :]

    assert torch.allclose(full_last, incr_last, atol=1e-4, rtol=1e-4)


def test_prefill_then_decode_matches_full():
    """Prefill a prefix in one shot, then decode the rest one token at a time."""
    cfg = make_model_config(dropout=0.0)
    model = Transformer(cfg).eval()
    seq = torch.randint(0, cfg.vocab_size, (1, 12))
    prefix = 7

    with torch.no_grad():
        full_logits, _ = model(seq)
        logits, caches = model(
            seq[:, :prefix], kv_caches=[None] * cfg.n_layers, start_pos=0
        )
        for t in range(prefix, seq.shape[1]):
            logits, caches = model(seq[:, t:t + 1], kv_caches=caches, start_pos=t)

    assert torch.allclose(full_logits[:, -1, :], logits[:, -1, :], atol=1e-4, rtol=1e-4)
