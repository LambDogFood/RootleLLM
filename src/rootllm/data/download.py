"""Tiny registry of downloadable text corpora for getting a real run going fast.

These are small, license-friendly datasets fetched on demand and cached locally.
For anything larger, just point ``prepare_data --input`` at your own files.
"""

from __future__ import annotations

import os
import ssl
import urllib.request
from typing import Dict, List

# name -> list of source URLs (concatenated in order when tokenising)
DATASETS: Dict[str, List[str]] = {
    # ~1.1 MB of Shakespeare — the classic nanoGPT smoke corpus. Trains in minutes.
    "shakespeare": [
        "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    ],
    # TinyStories validation split (~22 MB): short, simple children's stories.
    "tinystories": [
        "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt"
    ],
}


def _ssl_context() -> "ssl.SSLContext":
    """Prefer certifi's CA bundle — the system LibreSSL store is often incomplete."""
    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return ssl.create_default_context()


def _download(url: str, dest: str) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": "rootllm/0.1"})
    with urllib.request.urlopen(req, context=_ssl_context()) as resp, open(dest, "wb") as f:
        while True:
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)


def download_dataset(name: str, cache_dir: str = "data/.cache") -> List[str]:
    """Download (and cache) a named dataset, returning local file paths."""
    if name not in DATASETS:
        raise KeyError(f"unknown dataset {name!r}; choose from {sorted(DATASETS)}")
    os.makedirs(cache_dir, exist_ok=True)
    paths = []
    for url in DATASETS[name]:
        dest = os.path.join(cache_dir, f"{name}__{url.rsplit('/', 1)[-1]}")
        if not os.path.exists(dest):
            _download(url, dest)
        paths.append(dest)
    return paths
