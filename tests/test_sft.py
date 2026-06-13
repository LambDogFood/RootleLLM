"""SFT: loss must be computed on the response only, not the prompt."""

from __future__ import annotations

import json

import torch

from rootllm.data.sft import SFTDataset, format_prompt
from rootllm.data.tokenizer import ByteTokenizer
from rootllm.model import Transformer
from rootllm.model.transformer import IGNORE_INDEX
from tests.helpers import make_model_config


def _write_jsonl(path, records):
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def test_format_prompt_includes_instruction():
    p = format_prompt("Say hi")
    assert "Say hi" in p and "### Response:" in p
    assert "### Input:" not in p  # omitted when no input
    assert "### Input:" in format_prompt("translate", "bonjour")


def test_prompt_tokens_are_masked(tmp_path):
    path = tmp_path / "sft.jsonl"
    _write_jsonl(path, [{"instruction": "Say hi", "response": "Hello there"}])
    tok = ByteTokenizer()
    ds = SFTDataset.from_jsonl(str(path), tok)

    full, prompt_len = ds.examples[0]
    seq_len = len(full) - 1
    x, y = ds.get_batch(1, seq_len, torch.device("cpu"))

    assert x.shape == (1, seq_len) and y.shape == (1, seq_len)
    # the first (prompt_len - 1) targets are prompt -> masked
    assert torch.all(y[0, : prompt_len - 1] == IGNORE_INDEX)
    # the first response target is not masked
    assert y[0, prompt_len - 1] != IGNORE_INDEX
    # number of supervised tokens == response length (incl. appended eos)
    n_supervised = int((y[0] != IGNORE_INDEX).sum())
    assert n_supervised == len(full) - prompt_len


def test_padding_is_masked(tmp_path):
    path = tmp_path / "sft.jsonl"
    _write_jsonl(path, [{"instruction": "hi", "response": "yo"}])
    ds = SFTDataset.from_jsonl(str(path), ByteTokenizer())
    x, y = ds.get_batch(1, seq_len=64, device=torch.device("cpu"))  # longer than the example
    assert x.shape == (1, 64)
    # trailing pad positions contribute no loss
    assert (y[0] == IGNORE_INDEX).sum() > 0
    assert x[0, -1].item() == ds.pad_id


def test_sft_step_reduces_response_loss(tmp_path):
    path = tmp_path / "sft.jsonl"
    _write_jsonl(path, [{"instruction": "Greet me", "response": "Hello, friend!"}])
    tok = ByteTokenizer()
    ds = SFTDataset.from_jsonl(str(path), tok)

    cfg = make_model_config(vocab_size=tok.vocab_size, max_seq_len=128)
    model = Transformer(cfg).train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    gen = torch.Generator().manual_seed(0)

    x, y = ds.get_batch(4, 48, torch.device("cpu"), gen)
    _, first = model(x, targets=y)
    for _ in range(60):
        opt.zero_grad()
        x, y = ds.get_batch(4, 48, torch.device("cpu"), gen)
        _, loss = model(x, targets=y)
        loss.backward()
        opt.step()
    assert loss.item() < first.item() * 0.5
