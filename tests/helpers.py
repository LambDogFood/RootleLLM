"""Test helpers shared across modules."""

from __future__ import annotations

from rootllm.config import ModelConfig


def make_model_config(**overrides) -> ModelConfig:
    """A tiny, CPU-friendly model config; override any field via kwargs."""
    base = dict(
        vocab_size=64,
        dim=32,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        max_seq_len=64,
        dropout=0.0,
        use_qk_norm=True,
        rope_theta=10000.0,
        tie_embeddings=True,
    )
    base.update(overrides)
    return ModelConfig(**base)
