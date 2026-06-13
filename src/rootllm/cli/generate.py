"""``rootllm-generate`` — sample text from a trained checkpoint.

Example::

    rootllm-generate --ckpt out/ckpt.pt --prompt "Hello" --max-new-tokens 100 --top-k 50
"""

from __future__ import annotations

import argparse
import os
from typing import List, Optional

import torch

from ..data.tokenizer import ByteTokenizer, load_tokenizer
from ..generation import generate
from ..model import Transformer
from ..training.checkpoint import load_checkpoint
from ..utils.device import resolve_device
from ..utils.logging import get_logger
from ..utils.seed import set_seed


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Generate text from a rootllm checkpoint.")
    p.add_argument("--ckpt", required=True, help="path to a checkpoint .pt file")
    p.add_argument("--prompt", default="", help="prompt text")
    p.add_argument("--chat", action="store_true",
                   help="wrap the prompt in the SFT instruction template (for fine-tuned models)")
    p.add_argument("--tokenizer", default=None, help="tokenizer.json (defaults to byte tokenizer)")
    p.add_argument("--max-new-tokens", type=int, default=100)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=None)
    p.add_argument("--top-p", type=float, default=None)
    p.add_argument("--min-p", type=float, default=None)
    p.add_argument("--repetition-penalty", type=float, default=1.0)
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=1337)
    return p


def main(argv: Optional[List[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    set_seed(args.seed)
    device = resolve_device(args.device)

    payload = load_checkpoint(args.ckpt, map_location=device)
    cfg = payload["config_obj"]
    model = Transformer(cfg.model)
    model.load_state_dict(payload["model"])
    model.to(device).eval()

    # Tokenizer: explicit path, else one sitting next to the checkpoint, else byte.
    tok_path = args.tokenizer or os.path.join(os.path.dirname(args.ckpt), "tokenizer.json")
    tokenizer = load_tokenizer(tok_path) if os.path.exists(tok_path) else ByteTokenizer()

    prompt = args.prompt
    if args.chat:
        from ..data.sft import format_prompt

        prompt = format_prompt(prompt)

    ids = tokenizer.encode(prompt)
    if not ids:
        # Empty prompt: seed from the start-of-sequence token (not eos, which tells
        # the model to *stop*) and warn — output will be far less coherent.
        bos = getattr(tokenizer, "bos_id", -1)
        ids = [bos if bos is not None and bos >= 0 else 0]
        get_logger().warning("empty prompt — seeding from a start token; pass --prompt for coherent output")
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)

    out = generate(
        model,
        input_ids,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        min_p=args.min_p,
        repetition_penalty=args.repetition_penalty,
        eos_token_id=tokenizer.eos_id if tokenizer.eos_id >= 0 else None,
    )
    print(tokenizer.decode(out[0].tolist()))


if __name__ == "__main__":
    main()
