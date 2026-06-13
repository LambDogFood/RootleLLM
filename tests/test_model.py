from __future__ import annotations

import pytest
import torch

from rootllm.model import Transformer
from tests.helpers import make_model_config


def test_forward_train_shapes(model_config):
    model = Transformer(model_config)
    ids = torch.randint(0, model_config.vocab_size, (2, 16))
    logits, loss = model(ids, targets=ids)
    assert logits.shape == (2, 16, model_config.vocab_size)
    assert loss.ndim == 0 and torch.isfinite(loss)


def test_forward_inference_returns_caches(model_config):
    model = Transformer(model_config)
    ids = torch.randint(0, model_config.vocab_size, (2, 16))
    logits, caches = model(ids)
    assert logits.shape == (2, 16, model_config.vocab_size)
    assert len(caches) == model_config.n_layers
    k, v = caches[0]
    assert k.shape == (2, model_config.n_kv_heads, 16, model_config.head_dim)


def test_backward_populates_grads(model_config):
    model = Transformer(model_config)
    ids = torch.randint(0, model_config.vocab_size, (2, 8))
    _, loss = model(ids, targets=ids)
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert all(g is not None for g in grads)
    assert any(g.abs().sum() > 0 for g in grads)


def test_tied_embeddings_share_storage():
    model = Transformer(make_model_config(tie_embeddings=True))
    assert model.lm_head.weight.data_ptr() == model.tok_emb.weight.data_ptr()


def test_untied_embeddings_are_separate():
    model = Transformer(make_model_config(tie_embeddings=False))
    assert model.lm_head.weight.data_ptr() != model.tok_emb.weight.data_ptr()


@pytest.mark.parametrize("n_kv_heads", [1, 2, 4])  # MQA, GQA, MHA
def test_attention_variants_run(n_kv_heads):
    cfg = make_model_config(n_heads=4, n_kv_heads=n_kv_heads)
    model = Transformer(cfg)
    ids = torch.randint(0, cfg.vocab_size, (2, 12))
    logits, _ = model(ids)
    assert logits.shape == (2, 12, cfg.vocab_size)


def test_num_params_excludes_tied_embedding():
    model = Transformer(make_model_config(tie_embeddings=True))
    total = sum(p.numel() for p in model.parameters())
    assert model.num_params(non_embedding=True) == total - model.tok_emb.weight.numel()
    assert model.num_params(non_embedding=False) == total
