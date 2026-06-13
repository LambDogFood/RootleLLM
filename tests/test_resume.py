"""Resume-from-checkpoint must continue the run, not restart it."""

from __future__ import annotations

from rootllm.cli import train as train_cli
from rootllm.training.checkpoint import load_checkpoint


def _common(out_dir) -> list:
    return [
        "model.vocab_size=64", "model.dim=32", "model.n_layers=2",
        "model.n_heads=4", "model.n_kv_heads=2", "model.max_seq_len=64",
        f"train.out_dir={out_dir}", "train.device=cpu", "train.dtype=float32",
        "train.seq_len=16", "train.batch_size=4", "train.eval_interval=100",
        "train.eval_iters=1", "train.log_interval=5", "train.sample_tokens=0",
        "train.schedule.warmup_steps=1",
    ]


def test_resume_continues_step_and_optimizer(tmp_path):
    out = tmp_path / "out"
    ckpt = out / "ckpt.pt"

    # initial run to step 4
    train_cli.main(["--set", *_common(out), "train.schedule.max_steps=4"])
    first = load_checkpoint(str(ckpt), map_location="cpu")
    assert first["step"] == 4
    assert first["optimizer"] is not None
    assert "batch_rng_state" in first  # RNG persisted for stream continuity

    # resume and extend to step 8
    train_cli.main(["--resume", str(ckpt), "--set", *_common(out), "train.schedule.max_steps=8"])
    second = load_checkpoint(str(ckpt), map_location="cpu")
    assert second["step"] == 8


def test_init_from_resets_training_state(tmp_path):
    base = tmp_path / "base"
    train_cli.main(["--set", *_common(base), "train.schedule.max_steps=4"])

    # init-from loads weights but starts a fresh run at step 0 in a new out_dir
    ft = tmp_path / "ft"
    train_cli.main(["--init-from", str(base / "ckpt.pt"), "--set", *_common(ft),
                    "train.schedule.max_steps=3"])
    payload = load_checkpoint(str(ft / "ckpt.pt"), map_location="cpu")
    assert payload["step"] == 3  # reached its own horizon, not continued from 4


def test_resume_without_config_uses_checkpoint_config(tmp_path):
    out = tmp_path / "out"
    ckpt = out / "ckpt.pt"
    train_cli.main(["--set", *_common(out), "train.schedule.max_steps=3"])

    # No --config and no --set: should fall back entirely to the saved config and
    # re-save at the same horizon (step 3) without error.
    train_cli.main(["--resume", str(ckpt)])
    assert load_checkpoint(str(ckpt), map_location="cpu")["step"] == 3
