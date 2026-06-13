"""``rootllm-serve`` — run an HTTP inference server (on the GPU machine).

Example::

    rootllm-serve --ckpt out/rtx5070/ckpt.pt --tokenizer data/tinystories/tokenizer.json
"""

from __future__ import annotations

import argparse
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
    serve(args.ckpt, args.tokenizer, host=args.host, port=args.port, device=args.device)


if __name__ == "__main__":
    main()
