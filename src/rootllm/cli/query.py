"""``rootllm-query`` — query a running rootllm server (from the laptop).

No model or torch needed on this side — it's just an HTTP client.

Example::

    rootllm-query --host 192.168.1.50 --prompt "Once upon a time" --repetition-penalty 1.2
    rootllm-query --host 192.168.1.50 --chat --prompt "Give a blessing."
"""

from __future__ import annotations

import argparse
import json
import urllib.request
from typing import List, Optional


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Send a prompt to a rootllm HTTP server.")
    p.add_argument("--host", required=True, help="server host[:port] or full URL (default port 8000)")
    p.add_argument("--prompt", default="", help="prompt text")
    p.add_argument("--chat", action="store_true", help="wrap prompt in the SFT instruction template")
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=None)
    p.add_argument("--top-p", type=float, default=None)
    p.add_argument("--min-p", type=float, default=None)
    p.add_argument("--repetition-penalty", type=float, default=1.0)
    p.add_argument("--full", action="store_true", help="print prompt + completion, not just the completion")
    return p


def _base_url(host: str) -> str:
    if host.startswith("http://") or host.startswith("https://"):
        return host.rstrip("/")
    return f"http://{host}:8000" if ":" not in host else f"http://{host}"


def main(argv: Optional[List[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    payload = {
        "prompt": args.prompt,
        "chat": args.chat,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "top_p": args.top_p,
        "min_p": args.min_p,
        "repetition_penalty": args.repetition_penalty,
    }
    req = urllib.request.Request(
        _base_url(args.host) + "/generate",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req) as resp:
        result = json.loads(resp.read())
    print(result["text"] if args.full else result["completion"])


if __name__ == "__main__":
    main()
