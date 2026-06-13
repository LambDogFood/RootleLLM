"""Optimiser construction and the learning-rate schedule."""

from __future__ import annotations

import inspect
import math

import torch
import torch.nn as nn

from ..config import OptimizerConfig, ScheduleConfig


def configure_optimizer(
    model: nn.Module,
    cfg: OptimizerConfig,
    device_type: str = "cpu",
) -> torch.optim.Optimizer:
    """Build an AdamW optimiser with decoupled weight decay.

    Following common practice (nanoGPT), weight decay is applied only to matrices
    (``ndim >= 2``: linears, embeddings) and not to 1-D parameters (norm gains,
    biases). The fused CUDA kernel is used when available.
    """
    decay, no_decay = [], []
    for p in model.parameters():
        if not p.requires_grad:
            continue
        (decay if p.dim() >= 2 else no_decay).append(p)

    groups = [
        {"params": decay, "weight_decay": cfg.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]

    extra = {}
    if device_type == "cuda" and "fused" in inspect.signature(torch.optim.AdamW).parameters:
        extra["fused"] = True

    return torch.optim.AdamW(
        groups,
        lr=cfg.lr,
        betas=(cfg.beta1, cfg.beta2),
        eps=cfg.eps,
        **extra,
    )


def lr_at_step(step: int, base_lr: float, sched: ScheduleConfig) -> float:
    """Linear warmup then cosine decay to ``base_lr * min_lr_ratio``.

    ``step`` is 0-indexed. Before ``warmup_steps`` the LR ramps linearly; after
    ``max_steps`` it holds at the floor.
    """
    min_lr = base_lr * sched.min_lr_ratio
    if step < sched.warmup_steps:
        return base_lr * (step + 1) / max(1, sched.warmup_steps)
    if step >= sched.max_steps:
        return min_lr
    progress = (step - sched.warmup_steps) / max(1, sched.max_steps - sched.warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))  # 1 -> 0
    return min_lr + coeff * (base_lr - min_lr)
