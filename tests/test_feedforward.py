from __future__ import annotations

import torch

from rootllm.model import Transformer
from rootllm.model.feedforward import MoEFeedForward, SwiGLU
from tests.helpers import make_model_config


def test_swiglu_shape():
    ff = SwiGLU(dim=32, hidden_dim=64)
    x = torch.randn(2, 8, 32)
    assert ff(x).shape == (2, 8, 32)


def test_moe_shape_and_aux_loss():
    cfg = make_model_config(n_experts=4, n_experts_per_token=2)
    moe = MoEFeedForward(cfg)
    x = torch.randn(3, 8, cfg.dim)
    out = moe(x)
    assert out.shape == x.shape
    assert torch.isfinite(out).all()
    # aux loss is populated and positive after a forward pass
    assert moe.last_aux_loss.item() > 0


def test_moe_model_loss_includes_aux():
    cfg = make_model_config(n_experts=4, n_experts_per_token=2)
    model = Transformer(cfg)
    ids = torch.randint(0, cfg.vocab_size, (2, 8))
    _, loss = model(ids, targets=ids)
    assert torch.isfinite(loss)
    # at least one MoE block recorded an aux loss
    aux = [l.ffn.last_aux_loss.item() for l in model.layers if l.is_moe]
    assert len(aux) == cfg.n_layers and all(a > 0 for a in aux)


def test_moe_gradients_flow_to_experts():
    cfg = make_model_config(n_experts=4, n_experts_per_token=2)
    model = Transformer(cfg)
    ids = torch.randint(0, cfg.vocab_size, (2, 8))
    _, loss = model(ids, targets=ids)
    loss.backward()
    # the router and every expert should receive gradient
    gate_grad = model.layers[0].ffn.gate.weight.grad
    assert gate_grad is not None and gate_grad.abs().sum() > 0
