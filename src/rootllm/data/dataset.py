"""Token streams that yield ``(inputs, targets)`` batches for causal LM training.

Data is stored as a flat 1-D array of token ids in a ``.bin`` file (nanoGPT
style). Batches are sampled by drawing random windows of length ``seq_len + 1``;
the targets are the inputs shifted by one position.

:class:`BinaryTokenDataset` reads windows *directly from the memory-mapped file*
and only materialises the sampled windows — so a multi-GB corpus never lands in
RAM in full.
"""

from __future__ import annotations

import os
from typing import List, Tuple

import numpy as np
import torch

_NP_DTYPES = {"uint16": np.uint16, "uint32": np.uint32}


def np_dtype(name: str) -> "np.dtype":
    if name not in _NP_DTYPES:
        raise ValueError(f"unsupported token dtype {name!r}; use one of {sorted(_NP_DTYPES)}")
    return np.dtype(_NP_DTYPES[name])


def write_token_bin(ids, path: str, dtype: str = "uint16") -> int:
    """Write a sequence of token ids to ``path`` as a flat binary array.

    Returns the number of tokens written. ``uint16`` covers vocabularies up to
    65535; use ``uint32`` for larger ones.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    arr = np.asarray(ids, dtype=np_dtype(dtype))
    arr.tofile(path)
    return int(arr.size)


class TokenDataset:
    """Base class for next-token training streams.

    Subclasses set ``self._n`` (total token count) and implement
    :meth:`_gather`, which returns a ``(len(starts), length)`` int64 CPU tensor of
    windows. The batch assembly and device transfer live here so every backend
    behaves identically.
    """

    _n: int

    def __len__(self) -> int:
        return self._n

    def _gather(self, starts: List[int], length: int) -> torch.Tensor:
        raise NotImplementedError

    def get_batch(
        self,
        batch_size: int,
        seq_len: int,
        device: torch.device,
        generator: torch.Generator = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample a random batch of contiguous windows.

        Returns ``(x, y)`` of shape ``(batch_size, seq_len)`` where ``y`` is ``x``
        shifted one token to the right (the next-token prediction targets).
        """
        n = self._n
        if n < seq_len + 1:
            raise ValueError(f"dataset has {n} tokens but needs at least seq_len+1={seq_len + 1}")
        ix = torch.randint(n - seq_len, (batch_size,), generator=generator).tolist()
        x = self._gather(ix, seq_len)
        y = self._gather([i + 1 for i in ix], seq_len)
        x, y = x.long(), y.long()
        if device.type == "cuda":
            return (
                x.pin_memory().to(device, non_blocking=True),
                y.pin_memory().to(device, non_blocking=True),
            )
        return x.to(device), y.to(device)


class BinaryTokenDataset(TokenDataset):
    """Reads a ``.bin`` token file produced by :func:`write_token_bin`.

    The file stays memory-mapped on disk; only the sampled windows are read and
    up-cast to int64, so even a corpus far larger than RAM works.
    """

    def __init__(self, path: str, token_dtype: str = "uint16"):
        if not os.path.exists(path):
            raise FileNotFoundError(f"token file not found: {path}")
        self._mm = np.memmap(path, dtype=np_dtype(token_dtype), mode="r")
        self._n = int(self._mm.shape[0])
        self.path = path

    def _gather(self, starts: List[int], length: int) -> torch.Tensor:
        windows = [np.asarray(self._mm[i:i + length], dtype=np.int64) for i in starts]
        return torch.from_numpy(np.stack(windows))

    def __getitem__(self, i: int) -> int:
        return int(self._mm[i])


class RandomTokenDataset(TokenDataset):
    """Deterministic synthetic token stream for smoke tests and CI.

    Lets the whole pipeline run end-to-end with no data files. The stream is
    reproducible given ``seed``.
    """

    def __init__(self, vocab_size: int, n_tokens: int = 50000, seed: int = 0):
        g = torch.Generator().manual_seed(seed)
        self.data = torch.randint(0, vocab_size, (n_tokens,), generator=g, dtype=torch.int64)
        self.vocab_size = vocab_size
        self._n = int(self.data.numel())

    def _gather(self, starts: List[int], length: int) -> torch.Tensor:
        return torch.stack([self.data[i:i + length] for i in starts])
