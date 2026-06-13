from __future__ import annotations

import math

import torch

from rootllm.config import OptimizerConfig, ScheduleConfig
from rootllm.model import Transformer
from rootllm.training.optim import configure_optimizer, lr_at_step
from tests.helpers import make_model_config


def test_lr_warmup_then_cosine_floor():
    sched = ScheduleConfig(warmup_steps=10, max_steps=100, min_lr_ratio=0.1)
    base = 1.0
    # linear warmup
    assert math.isclose(lr_at_step(0, base, sched), base * 1 / 10, rel_tol=1e-6)
    assert math.isclose(lr_at_step(9, base, sched), base * 10 / 10, rel_tol=1e-6)
    # peak right after warmup
    assert lr_at_step(10, base, sched) > lr_at_step(50, base, sched)
    # decays monotonically through the cosine region
    assert lr_at_step(50, base, sched) > lr_at_step(99, base, sched)
    # floors at min_lr beyond max_steps
    assert math.isclose(lr_at_step(200, base, sched), base * 0.1, rel_tol=1e-6)


def test_configure_optimizer_param_groups():
    model = Transformer(make_model_config())
    opt = configure_optimizer(model, OptimizerConfig(weight_decay=0.1), device_type="cpu")
    assert isinstance(opt, torch.optim.AdamW)
    decay_group, no_decay_group = opt.param_groups
    assert decay_group["weight_decay"] == 0.1
    assert no_decay_group["weight_decay"] == 0.0
    # 1-D params (norm gains) must not be decayed
    assert all(p.dim() >= 2 for p in decay_group["params"])
    assert all(p.dim() < 2 for p in no_decay_group["params"])
