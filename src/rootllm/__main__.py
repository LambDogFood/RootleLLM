"""Smoke entry point: ``python -m rootllm``.

Builds a small model, runs a forward+loss pass and a short generation, and prints
the parameter count — a quick sanity check that the install works.
"""

from __future__ import annotations

import torch

from .config import ModelConfig
from .model import Transformer


def main() -> None:
    cfg = ModelConfig(
        vocab_size=32000, dim=512, n_layers=6, n_heads=8, n_kv_heads=2, max_seq_len=1024
    )
    model = Transformer(cfg)
    print(f"Params (non-emb): {model.num_params() / 1e6:.1f}M")

    ids = torch.randint(0, cfg.vocab_size, (2, 16))
    logits, loss = model(ids, targets=ids)
    print("logits", tuple(logits.shape), "loss", round(loss.item(), 4))

    out = model.generate(ids[:, :4], max_new_tokens=8, top_k=50)
    print("generated", tuple(out.shape))


if __name__ == "__main__":
    main()
