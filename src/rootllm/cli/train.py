"""``rootllm-train`` — train a model from a YAML config.

Examples::

    rootllm-train --config configs/small.yaml
    rootllm-train --config configs/debug.yaml --set train.schedule.max_steps=50
    rootllm-train --set model.dim=256 model.n_layers=4        # no file, pure overrides
    rootllm-train --config configs/m5pro.yaml --resume out/m5pro/ckpt.pt   # continue a run
"""

from __future__ import annotations

import argparse
import os
from typing import List, Optional, Tuple

from ..config import Config
from ..data.dataset import BinaryTokenDataset, RandomTokenDataset
from ..data.sft import SFTDataset
from ..data.tokenizer import ByteTokenizer, load_tokenizer
from ..model import Transformer
from ..training import Trainer
from ..training.checkpoint import load_checkpoint
from ..training.optim import configure_optimizer
from ..utils.device import resolve_device
from ..utils.logging import get_logger


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train a rootllm transformer.")
    p.add_argument("--config", default=None, help="path to a YAML config (see configs/)")
    p.add_argument(
        "--set", nargs="*", default=[], metavar="KEY=VALUE",
        help="dotted config overrides, e.g. train.optimizer.lr=1e-4",
    )
    p.add_argument("--resume", default=None,
                   help="checkpoint .pt to continue (keeps step, optimizer, schedule)")
    p.add_argument("--init-from", default=None,
                   help="checkpoint .pt to load weights from, with a fresh training state "
                        "(for fine-tuning / SFT)")
    p.add_argument("--sft", default=None,
                   help="JSONL of instruction/response pairs -> supervised fine-tuning")
    p.add_argument("--tokenizer", default=None,
                   help="tokenizer.json for SFT (defaults to the one beside the training data)")
    return p


def _make_dataset(path: str, cfg: Config, *, seed_offset: int):
    """A binary dataset if the file exists, else a deterministic synthetic stream."""
    if path and os.path.exists(path):
        return BinaryTokenDataset(path, cfg.data.token_dtype)
    return RandomTokenDataset(
        cfg.model.vocab_size,
        n_tokens=cfg.data.synthetic_tokens,
        seed=cfg.data.synthetic_seed + seed_offset,
    )


def _run_tokenizer(cfg: Config):
    """The tokenizer saved next to the training data, if any (for sample logging)."""
    if not cfg.data.train_path:
        return None
    path = os.path.join(os.path.dirname(cfg.data.train_path), "tokenizer.json")
    if os.path.exists(path):
        try:
            return load_tokenizer(path)
        except Exception:
            return None
    return None


def _build_config(args) -> Config:
    cfg = Config.from_yaml(args.config) if args.config else Config()
    cfg.apply_overrides(args.set)
    return cfg


def _resume(args, log) -> Tuple[Config, Transformer, object, int, float, object]:
    """Rebuild model + optimizer from a checkpoint. The model architecture comes
    from the checkpoint (it must match the weights); train/data settings come from
    the supplied --config/--set so you can extend or retune the run."""
    payload = load_checkpoint(args.resume, map_location="cpu")
    ckpt_cfg = payload["config_obj"]

    cfg = _build_config(args) if args.config or args.set else ckpt_cfg
    cfg.model = ckpt_cfg.model  # arch must match the saved weights

    device = resolve_device(cfg.train.device)
    # Move the model to the device *before* loading optimizer state, so AdamW's
    # state tensors land on the right device (load_state_dict follows the params).
    payload = load_checkpoint(args.resume, map_location=device)
    model = Transformer(cfg.model)
    model.load_state_dict(payload["model"])
    model.to(device)

    optimizer = configure_optimizer(model, cfg.train.optimizer, device.type)
    if payload.get("optimizer"):
        optimizer.load_state_dict(payload["optimizer"])

    start_step = int(payload.get("step", 0))
    best = float(payload.get("best_val_loss", float("inf")))
    batch_rng = payload.get("batch_rng_state")
    log.info("resuming from %s at step %d (best val %.4f)", args.resume, start_step, best)
    return cfg, model, optimizer, start_step, best, batch_rng


def _init_from(args, log) -> Tuple[Config, Transformer]:
    """Load *only* the weights from a checkpoint and start a fresh training run.

    Used for fine-tuning / SFT: the base model architecture and weights come from
    the checkpoint; the optimizer, step counter, and schedule reset to zero."""
    payload = load_checkpoint(args.init_from, map_location="cpu")
    ckpt_cfg = payload["config_obj"]
    cfg = _build_config(args) if (args.config or args.set) else ckpt_cfg
    cfg.model = ckpt_cfg.model  # arch must match the saved weights
    model = Transformer(cfg.model)
    model.load_state_dict(payload["model"])
    log.info("initialised weights from %s (fresh training state)", args.init_from)
    return cfg, model


def main(argv: Optional[List[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    log = get_logger()

    if args.resume:
        cfg, model, optimizer, start_step, best, batch_rng = _resume(args, log)
    elif args.init_from:
        cfg, model = _init_from(args, log)
        optimizer, start_step, best, batch_rng = None, 0, float("inf"), None
    else:
        cfg = _build_config(args)
        model = Transformer(cfg.model)
        optimizer, start_step, best, batch_rng = None, 0, float("inf"), None

    if args.sft:
        tokenizer = (
            load_tokenizer(args.tokenizer) if args.tokenizer
            else _run_tokenizer(cfg) or ByteTokenizer()
        )
        train_data = SFTDataset.from_jsonl(args.sft, tokenizer)
        val_data = train_data
        log.info("SFT on %s (%d examples, vocab %d)", args.sft, len(train_data), tokenizer.vocab_size)
    else:
        if not (cfg.data.train_path and os.path.exists(cfg.data.train_path)):
            log.warning("no train data at %r — using a synthetic random-token stream",
                        cfg.data.train_path)
        train_data = _make_dataset(cfg.data.train_path, cfg, seed_offset=0)
        val_data = _make_dataset(cfg.data.val_path, cfg, seed_offset=1)
        tokenizer = _run_tokenizer(cfg)

    log.info("model: %.2fM non-embedding params", model.num_params() / 1e6)

    # Persist the fully-resolved config alongside the checkpoints.
    os.makedirs(cfg.train.out_dir, exist_ok=True)
    cfg.to_yaml(os.path.join(cfg.train.out_dir, "config.yaml"))

    Trainer(
        cfg, model, train_data, val_data,
        optimizer=optimizer, start_step=start_step, best_val_loss=best,
        tokenizer=tokenizer, batch_rng_state=batch_rng,
    ).train()


if __name__ == "__main__":
    main()
