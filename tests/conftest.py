"""Shared fixtures: tiny CPU-friendly configs so tests run in well under a second."""

from __future__ import annotations

import pytest
import torch

from rootllm.config import Config, ModelConfig
from tests.helpers import make_model_config


@pytest.fixture(autouse=True)
def _deterministic():
    """Seed every test for reproducibility."""
    torch.manual_seed(0)
    yield


@pytest.fixture
def model_config() -> ModelConfig:
    return make_model_config()


@pytest.fixture
def tiny_config(tmp_path) -> Config:
    cfg = Config()
    cfg.model = make_model_config()
    cfg.train.out_dir = str(tmp_path / "out")
    cfg.train.device = "cpu"
    cfg.train.dtype = "float32"
    cfg.train.seq_len = 16
    cfg.train.batch_size = 4
    cfg.train.eval_interval = 5
    cfg.train.eval_iters = 2
    cfg.train.log_interval = 5
    cfg.train.schedule.warmup_steps = 2
    cfg.train.schedule.max_steps = 8
    cfg.data.synthetic_tokens = 2000
    return cfg
