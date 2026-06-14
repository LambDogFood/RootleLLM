"""``rootllm-ingest`` — research a topic from whitelisted sources and grow the corpus.

Appends fresh Wikipedia articles to ``corpus.txt`` and Stack Overflow
problem->solution pairs to ``sft.jsonl`` (deduped against previous runs).

    rootllm-ingest --topic "roblox pathfinding" --tags lua roblox
"""

from __future__ import annotations

import argparse
import os
from typing import List, Optional

from ..data.ingest import ingest_topic
from ..utils.logging import get_logger


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Research a topic into the training corpus.")
    p.add_argument("--topic", required=True, help="what to research")
    p.add_argument("--out-dir", default="data/research", help="accumulating corpus directory")
    p.add_argument("--tags", nargs="*", default=["lua", "roblox"], help="Stack Overflow tags")
    p.add_argument("--no-wikipedia", action="store_true")
    p.add_argument("--no-stackoverflow", action="store_true")
    return p


def main(argv: Optional[List[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    log = get_logger()
    log.info("researching %r from whitelisted sources ...", args.topic)

    n_corpus, n_pairs = ingest_topic(
        args.topic, args.out_dir,
        wikipedia=not args.no_wikipedia,
        stackoverflow=not args.no_stackoverflow,
        tags=tuple(args.tags),
    )
    log.info("added %d corpus docs and %d problem->solution pairs", n_corpus, n_pairs)

    corpus = os.path.join(args.out_dir, "corpus.txt")
    sft = os.path.join(args.out_dir, "sft.jsonl")
    if os.path.exists(corpus):
        log.info("corpus.txt total: %.0f KB", os.path.getsize(corpus) / 1024)
    if os.path.exists(sft):
        with open(sft) as f:
            total = sum(1 for _ in f)
        log.info("sft.jsonl total: %d pairs", total)


if __name__ == "__main__":
    main()
