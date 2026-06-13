"""``rootllm-serve`` — run an HTTP inference server (on the GPU machine).

Example::

    rootllm-serve --ckpt out/rtx5070/ckpt.pt --tokenizer data/tinystories/tokenizer.json
"""

from __future__ import annotations

import argparse
import os
from typing import List, Optional

from ..serve.server import serve


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Serve a rootllm checkpoint over HTTP.")
    p.add_argument("--ckpt", required=True, help="checkpoint .pt to serve")
    p.add_argument("--tokenizer", default=None, help="tokenizer.json (defaults to byte tokenizer)")
    p.add_argument("--host", default="0.0.0.0", help="0.0.0.0 to accept LAN connections")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--device", default="auto")
    return p


def main(argv: Optional[List[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    # Fail with a clear message instead of a deep traceback (the usual startup crash).
    if not os.path.exists(args.ckpt):
        # Common case: the run was cancelled before the final ckpt.pt, but best.pt
        # (saved at each val improvement) exists. Fall back to it automatically.
        alt = os.path.join(os.path.dirname(args.ckpt) or ".", "best.pt")
        if os.path.basename(args.ckpt) != "best.pt" and os.path.exists(alt):
            print(f"{args.ckpt} not found; serving {alt} instead")
            args.ckpt = alt
        else:
            raise SystemExit(
                f"checkpoint not found: {args.ckpt}\n"
                "Did a training run finish and write it here? Try out/<config>/best.pt, "
                "or run a training run (it now saves a checkpoint at every eval)."
            )
    if args.tokenizer and not os.path.exists(args.tokenizer):
        raise SystemExit(
            f"tokenizer not found: {args.tokenizer}\n"
            "Point --tokenizer at the one for your dataset, e.g. data/luau/tokenizer.json."
        )
    serve(args.ckpt, args.tokenizer, host=args.host, port=args.port, device=args.device)


if __name__ == "__main__":
    main()
