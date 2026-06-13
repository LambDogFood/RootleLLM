from __future__ import annotations

import torch

from rootllm.model.rope import _rotate_half, apply_rope, precompute_rope_cache


def test_cache_shapes():
    cos, sin = precompute_rope_cache(head_dim=16, max_seq_len=32, theta=10000.0)
    assert cos.shape == (32, 16)
    assert sin.shape == (32, 16)
    # cos(0)=1, sin(0)=0 at position 0
    assert torch.allclose(cos[0], torch.ones(16))
    assert torch.allclose(sin[0], torch.zeros(16))


def test_rotate_half():
    x = torch.tensor([1.0, 2.0, 3.0, 4.0])
    # [a, b] -> [-b, a] with a=[1,2], b=[3,4]
    assert torch.allclose(_rotate_half(x), torch.tensor([-3.0, -4.0, 1.0, 2.0]))


def test_apply_rope_preserves_shape_and_norm():
    B, H, T, D = 2, 4, 8, 16
    q = torch.randn(B, H, T, D)
    k = torch.randn(B, H, T, D)
    cos, sin = precompute_rope_cache(D, T, theta=10000.0)
    q2, k2 = apply_rope(q, k, cos, sin)
    assert q2.shape == q.shape and k2.shape == k.shape
    # Rotation is norm-preserving per (pair) — total vector norm is unchanged.
    assert torch.allclose(q2.norm(dim=-1), q.norm(dim=-1), atol=1e-5)


def test_scaling_factor_stretches_positions():
    cos1, _ = precompute_rope_cache(16, 32, theta=10000.0, scaling_factor=1.0)
    cos2, _ = precompute_rope_cache(16, 32, theta=10000.0, scaling_factor=2.0)
    # position 2 under 2x scaling equals position 1 under 1x scaling
    assert torch.allclose(cos2[2], cos1[1], atol=1e-6)
