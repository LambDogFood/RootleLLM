"""``rootllm-prepare-data`` — tokenise a text corpus into train/val ``.bin`` files.

Example::

    rootllm-prepare-data --input corpus.txt --output-dir data --val-frac 0.1
"""

from __future__ import annotations

import argparse
import json
import os
from typing import List, Optional

from ..data.dataset import write_token_bin
from ..data.download import DATASETS, download_dataset
from ..data.tokenizer import build_tokenizer
from ..utils.logging import get_logger


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Tokenise text into train/val token bins.")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--input", nargs="+", help="one or more UTF-8 text files")
    src.add_argument("--dataset", choices=sorted(DATASETS), help="download a built-in dataset")
    p.add_argument("--output-dir", default="data", help="directory for the .bin / meta files")
    p.add_argument("--tokenizer", default="byte", choices=["byte", "tiktoken"])
    p.add_argument("--encoding", default="gpt2", help="tiktoken encoding name (if --tokenizer tiktoken)")
    p.add_argument("--val-frac", type=float, default=0.1, help="fraction of tokens held out for validation")
    p.add_argument("--dtype", default="uint16", choices=["uint16", "uint32"])
    return p


def main(argv: Optional[List[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    log = get_logger()
    os.makedirs(args.output_dir, exist_ok=True)

    tok = build_tokenizer(args.tokenizer, **({"encoding": args.encoding} if args.tokenizer == "tiktoken" else {}))

    if args.dataset:
        log.info("downloading dataset %r ...", args.dataset)
        input_paths = download_dataset(args.dataset)
    else:
        input_paths = args.input

    text = []
    for path in input_paths:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            text.append(f.read())
    ids = tok.encode("\n".join(text))
    log.info("encoded %d tokens with %s tokenizer (vocab %d)", len(ids), args.tokenizer, tok.vocab_size)

    n_val = int(len(ids) * args.val_frac)
    train_ids, val_ids = ids[:len(ids) - n_val], ids[len(ids) - n_val:]

    train_path = os.path.join(args.output_dir, "train.bin")
    val_path = os.path.join(args.output_dir, "val.bin")
    n_train = write_token_bin(train_ids, train_path, args.dtype)
    n_val = write_token_bin(val_ids, val_path, args.dtype) if val_ids else 0

    tok.save(os.path.join(args.output_dir, "tokenizer.json"))
    meta = {
        "tokenizer": args.tokenizer,
        "vocab_size": tok.vocab_size,
        "token_dtype": args.dtype,
        "n_train": n_train,
        "n_val": n_val,
    }
    with open(os.path.join(args.output_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    log.info("wrote %s (%d) and %s (%d)", train_path, n_train, val_path, n_val)
    log.info("set model.vocab_size=%d and data.token_dtype=%s in your config", tok.vocab_size, args.dtype)


if __name__ == "__main__":
    main()
