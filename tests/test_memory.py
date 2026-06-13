from __future__ import annotations

import math

import torch

from rootllm.utils.device import apply_memory_cap, memory_cap_fraction


def test_fraction_math():
    # 32 GB of a ~55.66 GB recommended max -> ~0.575
    frac = memory_cap_fraction(32, 55.662788608e9, max_fraction=2.0)
    assert math.isclose(frac, 32e9 / 55.662788608e9, rel_tol=1e-9)


def test_fraction_clamps_to_max():
    # cap larger than total clamps to max_fraction
    assert memory_cap_fraction(100, 50e9, max_fraction=1.0) == 1.0
    assert memory_cap_fraction(200, 50e9, max_fraction=2.0) == 2.0


def test_no_cap_returns_none():
    cpu = torch.device("cpu")
    assert apply_memory_cap(cpu, None) is None
    assert apply_memory_cap(cpu, 0) is None


def test_cpu_cannot_enforce():
    # A cap on CPU is a no-op (no per-process device allocator to cap).
    assert apply_memory_cap(torch.device("cpu"), 32) is None


def test_mps_apply_returns_fraction():
    if not torch.backends.mps.is_available():
        import pytest

        pytest.skip("MPS not available")
    try:
        frac = apply_memory_cap(torch.device("mps"), 50)  # generous cap
        assert frac is not None and 0.0 < frac <= 2.0
    finally:
        # Restore a high ceiling so we don't constrain other tests in this process.
        torch.mps.set_per_process_memory_fraction(2.0)
