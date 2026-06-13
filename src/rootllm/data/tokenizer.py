"""Tokenisers.

The default :class:`ByteTokenizer` is dependency-free and lossless: it maps text
to its UTF-8 bytes (vocab 256) plus optional special tokens. That means you can
train end-to-end immediately, with no external tokenizer files to manage. For a
larger, more efficient vocabulary, install ``tiktoken`` and use
:class:`TiktokenTokenizer`.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Protocol, runtime_checkable


@runtime_checkable
class Tokenizer(Protocol):
    """Minimal tokenizer interface used across training and generation."""

    @property
    def vocab_size(self) -> int: ...

    @property
    def eos_id(self) -> int: ...

    def encode(self, text: str) -> List[int]: ...

    def decode(self, ids: List[int]) -> str: ...


class ByteTokenizer:
    """Lossless byte-level tokenizer.

    Bytes ``0..255`` are the base vocabulary. Special tokens (``<bos>``, ``<eos>``
    by default) occupy ids ``256+``. Decoding skips special tokens and replaces
    malformed UTF-8 rather than raising.
    """

    name = "byte"

    def __init__(self, special_tokens: List[str] = ("<bos>", "<eos>")):
        self.special_tokens = list(special_tokens)
        self._special_to_id: Dict[str, int] = {
            tok: 256 + i for i, tok in enumerate(self.special_tokens)
        }

    @property
    def vocab_size(self) -> int:
        return 256 + len(self.special_tokens)

    @property
    def bos_id(self) -> int:
        return self._special_to_id.get("<bos>", -1)

    @property
    def eos_id(self) -> int:
        return self._special_to_id.get("<eos>", -1)

    def encode(self, text: str) -> List[int]:
        return list(text.encode("utf-8"))

    def decode(self, ids: List[int]) -> str:
        byte_vals = bytes(i for i in ids if 0 <= i < 256)
        return byte_vals.decode("utf-8", errors="replace")

    # -- persistence ---------------------------------------------------------
    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump({"type": self.name, "special_tokens": self.special_tokens}, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "ByteTokenizer":
        with open(path, "r") as f:
            meta = json.load(f)
        return cls(special_tokens=meta.get("special_tokens", ["<bos>", "<eos>"]))


class TiktokenTokenizer:
    """BPE tokenizer backed by ``tiktoken`` (optional dependency)."""

    name = "tiktoken"

    def __init__(self, encoding: str = "gpt2"):
        import tiktoken  # imported lazily so tiktoken stays optional

        self.encoding_name = encoding
        self._enc = tiktoken.get_encoding(encoding)

    @property
    def vocab_size(self) -> int:
        return self._enc.n_vocab

    @property
    def eos_id(self) -> int:
        return self._enc.eot_token

    def encode(self, text: str) -> List[int]:
        # Encode special-token strings (e.g. the "<|endoftext|>" document separators
        # used by TinyStories) as their special ids rather than erroring.
        return self._enc.encode(text, allowed_special="all")

    def decode(self, ids: List[int]) -> str:
        return self._enc.decode(list(ids))

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump({"type": self.name, "encoding": self.encoding_name}, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "TiktokenTokenizer":
        with open(path, "r") as f:
            meta = json.load(f)
        return cls(encoding=meta.get("encoding", "gpt2"))


def build_tokenizer(kind: str = "byte", **kwargs) -> Tokenizer:
    """Construct a tokenizer by name: ``"byte"`` or ``"tiktoken"``."""
    if kind == "byte":
        return ByteTokenizer(**kwargs)
    if kind == "tiktoken":
        return TiktokenTokenizer(**kwargs)
    raise ValueError(f"unknown tokenizer kind {kind!r}")


def load_tokenizer(path: str) -> Tokenizer:
    """Load a tokenizer from a metadata JSON written by ``.save``."""
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with open(path, "r") as f:
        meta = json.load(f)
    kind = meta.get("type", "byte")
    if kind == "byte":
        return ByteTokenizer.load(path)
    if kind == "tiktoken":
        return TiktokenTokenizer.load(path)
    raise ValueError(f"unknown tokenizer type {kind!r} in {path}")
