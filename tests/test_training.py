from __future__ import annotations

import os

import torch

from rootllm.data.dataset import RandomTokenDataset
from rootllm.model import Transformer
from rootllm.training import Trainer
from rootllm.training.checkpoint import load_checkpoint, save_checkpoint
from tests.helpers import make_model_config


def test_model_overfits_single_batch():
    """A few steps of Adam on one fixed batch should sharply reduce the loss."""
    torch.manual_seed(0)
    cfg = make_model_config()
    model = Transformer(cfg).train()
    x = torch.randint(0, cfg.vocab_size, (4, 16))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

    _, first = model(x, targets=x)
    for _ in range(60):
        opt.zero_grad()
        _, loss = model(x, targets=x)
        loss.backward()
        opt.step()

    assert loss.item() < first.item() * 0.5


def test_trainer_runs_and_checkpoints(tiny_config):
    cfg = tiny_config
    model = Transformer(cfg.model)
    train_data = RandomTokenDataset(cfg.model.vocab_size, n_tokens=2000, seed=0)
    val_data = RandomTokenDataset(cfg.model.vocab_size, n_tokens=2000, seed=1)

    history = Trainer(cfg, model, train_data, val_data).train()

    assert len(history["loss"]) == cfg.train.schedule.max_steps
    assert all(lr >= 0 for lr in history["lr"])
    assert os.path.exists(os.path.join(cfg.train.out_dir, "ckpt.pt"))
    assert os.path.exists(os.path.join(cfg.train.out_dir, "metrics.jsonl"))


def test_checkpoint_roundtrip_preserves_weights(tmp_path):
    cfg_model = make_model_config()
    model = Transformer(cfg_model).eval()
    from rootllm.config import Config

    cfg = Config()
    cfg.model = cfg_model
    path = str(tmp_path / "ckpt.pt")
    save_checkpoint(path, model, optimizer=None, config=cfg, step=42, best_val_loss=1.23)

    payload = load_checkpoint(path, map_location="cpu")
    assert payload["step"] == 42
    assert payload["best_val_loss"] == 1.23
    assert payload["config_obj"].model.dim == cfg_model.dim

    restored = Transformer(payload["config_obj"].model).eval()
    restored.load_state_dict(payload["model"])

    ids = torch.randint(0, cfg_model.vocab_size, (1, 8))
    with torch.no_grad():
        a, _ = model(ids)
        b, _ = restored(ids)
    assert torch.allclose(a, b, atol=1e-6)
