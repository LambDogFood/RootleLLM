"""End-to-end smoke of the user-facing CLI: prepare-data -> train -> generate."""

from __future__ import annotations

import os

from rootllm.cli import generate as generate_cli
from rootllm.cli import prepare_data as prepare_cli
from rootllm.cli import train as train_cli


def test_full_pipeline(tmp_path, capsys):
    # --- 1. prepare data from a tiny text corpus ---
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("the quick brown fox jumps over the lazy dog. " * 40, encoding="utf-8")
    data_dir = tmp_path / "data"
    prepare_cli.main(["--input", str(corpus), "--output-dir", str(data_dir), "--val-frac", "0.1"])

    train_bin = data_dir / "train.bin"
    tok_json = data_dir / "tokenizer.json"
    assert train_bin.exists() and tok_json.exists()

    # --- 2. train a tiny model on it ---
    out_dir = tmp_path / "out"
    train_cli.main([
        "--set",
        "model.vocab_size=258",
        "model.dim=32",
        "model.n_layers=2",
        "model.n_heads=4",
        "model.n_kv_heads=2",
        "model.max_seq_len=64",
        f"train.out_dir={out_dir}",
        "train.device=cpu",
        "train.dtype=float32",
        "train.seq_len=16",
        "train.batch_size=4",
        "train.eval_interval=100",
        "train.eval_iters=2",
        "train.log_interval=2",
        "train.schedule.warmup_steps=1",
        "train.schedule.max_steps=4",
        f"data.train_path={train_bin}",
        f"data.val_path={data_dir / 'val.bin'}",
        "data.token_dtype=uint16",
    ])

    ckpt = out_dir / "ckpt.pt"
    assert ckpt.exists()
    assert (out_dir / "config.yaml").exists()

    # --- 3. generate from the checkpoint ---
    generate_cli.main([
        "--ckpt", str(ckpt),
        "--prompt", "the quick",
        "--tokenizer", str(tok_json),
        "--max-new-tokens", "10",
        "--device", "cpu",
        "--temperature", "0.8",
    ])
    out = capsys.readouterr().out
    assert len(out.strip()) > 0
