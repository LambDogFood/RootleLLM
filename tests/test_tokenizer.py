from __future__ import annotations

import pytest

from rootllm.data.tokenizer import ByteTokenizer, load_tokenizer


def test_byte_roundtrip_ascii():
    tok = ByteTokenizer()
    text = "Hello, world!"
    assert tok.decode(tok.encode(text)) == text


def test_byte_roundtrip_unicode():
    tok = ByteTokenizer()
    text = "héllo — 世界 🌍"
    assert tok.decode(tok.encode(text)) == text


def test_vocab_and_special_ids():
    tok = ByteTokenizer()
    assert tok.vocab_size == 258  # 256 bytes + <bos> + <eos>
    assert tok.bos_id == 256
    assert tok.eos_id == 257


def test_decode_skips_special_tokens():
    tok = ByteTokenizer()
    ids = tok.encode("hi") + [tok.eos_id]
    assert tok.decode(ids) == "hi"


def test_tokenizer_save_load(tmp_path):
    tok = ByteTokenizer(special_tokens=["<bos>", "<eos>", "<pad>"])
    path = tmp_path / "tokenizer.json"
    tok.save(str(path))
    loaded = load_tokenizer(str(path))
    assert loaded.vocab_size == tok.vocab_size
    assert loaded.encode("abc") == tok.encode("abc")


def test_tiktoken_roundtrip_and_save_load(tmp_path):
    pytest.importorskip("tiktoken")
    from rootllm.data.tokenizer import TiktokenTokenizer

    tok = TiktokenTokenizer(encoding="gpt2")
    assert tok.vocab_size == 50257
    assert tok.decode(tok.encode("Hello world")) == "Hello world"

    path = tmp_path / "tokenizer.json"
    tok.save(str(path))
    loaded = load_tokenizer(str(path))
    assert loaded.vocab_size == tok.vocab_size
    assert loaded.encode("abc") == tok.encode("abc")
