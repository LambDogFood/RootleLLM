"""Build a Luau training corpus from public, permissively-licensed Roblox repos.

For a proof-of-concept Roblox-dev model: concatenate ``.lua``/``.luau`` source
from well-known open-source Roblox libraries into one corpus, separated by the
end-of-text token. Tokenise it with ``prepare_data --dataset luau`` and train as
usual. To make it mimic *your* code, add your own files with ``--input`` instead.
"""

from __future__ import annotations

import io
import os
import ssl
import tarfile
import urllib.request
from typing import List, Tuple

# (owner, repo) — MIT/Apache, Luau-heavy. The fetcher tries main then master.
LUAU_REPOS: List[Tuple[str, str]] = [
    ("Roblox", "roact"),
    ("Roblox", "rodux"),
    ("Sleitnick", "Knit"),
    ("Sleitnick", "RbxUtil"),
    ("evaera", "roblox-lua-promise"),
    ("matter-ecs", "matter"),
    ("osyrisrblx", "t"),
    ("Quenty", "NevermoreEngine"),
]


def _ssl_context() -> "ssl.SSLContext":
    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return ssl.create_default_context()


def _download_tarball(owner: str, repo: str) -> bytes:
    last = None
    for branch in ("main", "master"):
        url = f"https://codeload.github.com/{owner}/{repo}/tar.gz/refs/heads/{branch}"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "rootllm/0.1"})
            with urllib.request.urlopen(req, context=_ssl_context()) as resp:
                return resp.read()
        except Exception as e:  # try the other branch name
            last = e
    raise RuntimeError(f"could not fetch {owner}/{repo}: {last}")


def fetch_luau_corpus(
    out_path: str,
    repos: List[Tuple[str, str]] = LUAU_REPOS,
    cache_dir: str = "data/.cache",
) -> Tuple[str, int]:
    """Download the repos, extract every ``.lua``/``.luau`` file, concatenate.

    Files are separated by ``<|endoftext|>`` so the tokenizer learns boundaries.
    Returns ``(out_path, n_files)``.
    """
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)
    n_files = 0
    with open(out_path, "w", encoding="utf-8") as out:
        for owner, repo in repos:
            cache = os.path.join(cache_dir, f"{owner}__{repo}.tar.gz")
            if os.path.exists(cache):
                with open(cache, "rb") as f:
                    blob = f.read()
            else:
                blob = _download_tarball(owner, repo)
                with open(cache, "wb") as f:
                    f.write(blob)
            with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as tar:
                for member in tar.getmembers():
                    if not member.isfile():
                        continue
                    if not (member.name.endswith(".lua") or member.name.endswith(".luau")):
                        continue
                    f = tar.extractfile(member)
                    if f is None:
                        continue
                    out.write(f.read().decode("utf-8", errors="replace"))
                    out.write("\n<|endoftext|>\n")
                    n_files += 1
    return out_path, n_files
