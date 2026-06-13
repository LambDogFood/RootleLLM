from __future__ import annotations

import math

import pytest

from rootllm.config import Config, ModelConfig


def test_derived_fields():
    cfg = ModelConfig(dim=512, n_heads=8, n_kv_heads=None, head_dim=None, ffn_hidden_dim=None)
    assert cfg.head_dim == 64
    assert cfg.n_kv_heads == 8  # None -> n_heads (MHA)
    # SwiGLU hidden dim rounds 8/3*dim up to a multiple of ffn_multiple_of.
    assert cfg.ffn_hidden_dim % cfg.ffn_multiple_of == 0
    assert cfg.ffn_hidden_dim >= int(8 / 3 * 512)


def test_invalid_head_division():
    with pytest.raises(AssertionError):
        ModelConfig(dim=100, n_heads=8)  # 100 not divisible by 8


def test_gqa_divisibility_check():
    with pytest.raises(AssertionError):
        ModelConfig(dim=64, n_heads=8, n_kv_heads=3)  # 8 % 3 != 0


def test_from_dict_nested():
    cfg = Config.from_dict(
        {
            "model": {"dim": 128, "n_heads": 4},
            "train": {"batch_size": 2, "optimizer": {"lr": 1e-3}},
        }
    )
    assert cfg.model.dim == 128
    assert cfg.train.batch_size == 2
    assert math.isclose(cfg.train.optimizer.lr, 1e-3)
    # untouched fields keep their defaults
    assert cfg.train.schedule.warmup_steps == 100


def test_from_dict_rejects_unknown_key():
    with pytest.raises(KeyError):
        Config.from_dict({"model": {"not_a_field": 1}})


def test_apply_overrides_reparses_and_revalidates():
    cfg = Config()
    cfg.apply_overrides(["train.optimizer.lr=1e-4", "model.dim=256", "model.n_heads=8"])
    assert math.isclose(cfg.train.optimizer.lr, 1e-4)
    assert cfg.model.dim == 256
    # __post_init__ rerun -> head_dim recomputed for the new dim
    assert cfg.model.head_dim == 256 // 8


def test_yaml_roundtrip(tmp_path):
    cfg = Config()
    cfg.model.dim = 128
    cfg.model.n_heads = 4
    path = tmp_path / "cfg.yaml"
    cfg.to_yaml(str(path))
    loaded = Config.from_yaml(str(path))
    assert loaded.model.dim == 128
    assert loaded.train.optimizer.lr == cfg.train.optimizer.lr
