"""Gradient checkpointing must be mathematically transparent.

It only changes *when* activations are computed (recomputed in backward), never the
result — so loss and gradients must match the non-checkpointed path, and the MoE
router must still receive gradient through the checkpoint boundary.
"""

from __future__ import annotations

import torch

from rootllm.generation import generate
from rootllm.model import Transformer
from tests.helpers import make_model_config


def test_checkpointing_matches_dense_loss_and_grads():
    cfg = make_model_config(dropout=0.0)
    model = Transformer(cfg).train()
    x = torch.randint(0, cfg.vocab_size, (2, 16))

    _, loss0 = model(x, targets=x)
    loss0.backward()
    grads0 = [p.grad.clone() for p in model.parameters()]

    model.zero_grad(set_to_none=True)
    model.cfg.gradient_checkpointing = True
    _, loss1 = model(x, targets=x)
    loss1.backward()
    grads1 = [p.grad for p in model.parameters()]

    assert torch.allclose(loss0, loss1, atol=1e-6)
    assert all(torch.allclose(a, b, atol=1e-5) for a, b in zip(grads0, grads1))


def test_checkpointing_moe_loss_matches_and_router_gets_grad():
    cfg = make_model_config(n_experts=4, n_experts_per_token=2, dropout=0.0)
    model = Transformer(cfg).train()
    x = torch.randint(0, cfg.vocab_size, (2, 16))

    _, loss_off = model(x, targets=x)
    model.cfg.gradient_checkpointing = True
    _, loss_on = model(x, targets=x)
    assert torch.allclose(loss_off, loss_on, atol=1e-6)

    loss_on.backward()
    gate_grad = model.layers[0].ffn.gate.weight.grad
    assert gate_grad is not None and gate_grad.abs().sum() > 0


def test_checkpointing_off_during_inference():
    # Checkpointing requires grad; generation runs under no_grad and must be unaffected.
    cfg = make_model_config(gradient_checkpointing=True)
    model = Transformer(cfg).eval()
    out = generate(model, torch.randint(0, cfg.vocab_size, (1, 4)), max_new_tokens=4)
    assert out.shape == (1, 8)
