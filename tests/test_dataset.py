from __future__ import annotations

import numpy as np
import torch

from rootllm.data.dataset import BinaryTokenDataset, RandomTokenDataset, write_token_bin


def test_write_and_read_bin(tmp_path):
    ids = list(range(1000))
    path = tmp_path / "toks.bin"
    n = write_token_bin(ids, str(path), dtype="uint16")
    assert n == 1000
    ds = BinaryTokenDataset(str(path), token_dtype="uint16")
    assert len(ds) == 1000
    # tokens are read straight from the memmap, not a materialised tensor
    assert ds[0] == 0 and ds[999] == 999


def test_binary_dataset_does_not_load_full_corpus(tmp_path):
    path = tmp_path / "toks.bin"
    write_token_bin(list(range(10000)), str(path), dtype="uint16")
    ds = BinaryTokenDataset(str(path), token_dtype="uint16")
    # the corpus stays memory-mapped on disk (no in-RAM copy of the whole array)
    assert isinstance(ds._mm, np.memmap)


def test_get_batch_shapes_and_shift(tmp_path):
    path = tmp_path / "toks.bin"
    write_token_bin(list(range(500)), str(path), dtype="uint16")
    ds = BinaryTokenDataset(str(path), token_dtype="uint16")
    x, y = ds.get_batch(batch_size=4, seq_len=8, device=torch.device("cpu"))
    assert x.shape == (4, 8) and y.shape == (4, 8)
    assert x.dtype == torch.long
    # targets are inputs shifted by one position
    assert torch.equal(y[:, :-1], x[:, 1:])


def test_random_dataset_is_deterministic():
    a = RandomTokenDataset(vocab_size=50, n_tokens=200, seed=7)
    b = RandomTokenDataset(vocab_size=50, n_tokens=200, seed=7)
    assert torch.equal(a.data, b.data)
    assert int(a.data.max()) < 50


def test_get_batch_is_reproducible_with_generator():
    ds = RandomTokenDataset(vocab_size=50, n_tokens=500, seed=1)
    g1 = torch.Generator().manual_seed(123)
    g2 = torch.Generator().manual_seed(123)
    x1, _ = ds.get_batch(4, 8, torch.device("cpu"), generator=g1)
    x2, _ = ds.get_batch(4, 8, torch.device("cpu"), generator=g2)
    assert torch.equal(x1, x2)
