"""Research ingester: pull fresh content on a topic from whitelisted sources.

Favours recent material and dedups against what's already been ingested. Produces
two streams:
  - a text **corpus** (Wikipedia articles, docs) for continued pretraining
  - **problem -> solution** pairs (Stack Overflow Q&A) for SFT

Only whitelisted sources are queried. Starts with Wikipedia + Stack Overflow
(no auth needed); docs/GitHub-code adapters plug in the same way.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import re
import ssl
import time
import urllib.parse
import urllib.request
from html import unescape
from typing import List, Tuple

_UA = {"User-Agent": "rootllm-ingest/0.1 (personal research bot)"}


def _ssl_context() -> "ssl.SSLContext":
    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return ssl.create_default_context()


def _get(url: str, retries: int = 3) -> bytes:
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={**_UA, "Accept-Encoding": "gzip"})
            with urllib.request.urlopen(req, context=_ssl_context(), timeout=30) as resp:
                data = resp.read()
                if resp.headers.get("Content-Encoding") == "gzip":  # Stack Exchange always gzips
                    data = gzip.decompress(data)
                return data
        except urllib.error.HTTPError as e:
            if e.code == 429 and attempt < retries - 1:  # rate limited; back off
                time.sleep(2 * (attempt + 1))
                continue
            raise
    raise RuntimeError("unreachable")


def _get_json(url: str) -> dict:
    return json.loads(_get(url))


def _strip_html(html: str) -> str:
    html = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", html, flags=re.S | re.I)
    text = re.sub(r"<[^>]+>", "", html)
    return unescape(text).strip()


# --------------------------------------------------------------------------- #
# Source adapters
# --------------------------------------------------------------------------- #
def fetch_wikipedia(topic: str, max_articles: int = 5) -> List[str]:
    """Top article extracts for a topic (always current)."""
    api = "https://en.wikipedia.org/w/api.php"
    q = urllib.parse.urlencode(
        {"action": "query", "list": "search", "srsearch": topic,
         "format": "json", "srlimit": max_articles}
    )
    titles = [hit["title"] for hit in _get_json(f"{api}?{q}").get("query", {}).get("search", [])]
    if not titles:
        return []
    # One batched extract request for all titles (avoids per-article rate limiting).
    q2 = urllib.parse.urlencode(
        {"action": "query", "prop": "extracts", "explaintext": "1", "exlimit": "max",
         "titles": "|".join(titles), "format": "json"}
    )
    pages = _get_json(f"{api}?{q2}").get("query", {}).get("pages", {})
    out = []
    for page in pages.values():
        extract = (page.get("extract") or "").strip()
        if extract:
            out.append(f"# {page.get('title', '')}\n\n{extract}")
    return out


def fetch_stackoverflow(
    topic: str, tags: Tuple[str, ...] = ("lua", "roblox"), max_questions: int = 15
) -> List[Tuple[str, str]]:
    """Recent, accepted-answer Q&A as (problem, solution) pairs."""
    api = "https://api.stackexchange.com/2.3"
    q = urllib.parse.urlencode(
        {"order": "desc", "sort": "activity", "accepted": "True", "tagged": ";".join(tags),
         "q": topic, "site": "stackoverflow", "pagesize": max_questions}
    )
    items = _get_json(f"{api}/search/advanced?{q}").get("items", [])
    title_by_answer = {it["accepted_answer_id"]: it["title"]
                       for it in items if it.get("accepted_answer_id")}
    if not title_by_answer:
        return []
    ids = ";".join(str(i) for i in title_by_answer)
    q2 = urllib.parse.urlencode(
        {"order": "desc", "sort": "votes", "site": "stackoverflow", "filter": "withbody"}
    )
    answers = _get_json(f"{api}/answers/{ids}?{q2}").get("items", [])
    pairs = []
    for a in answers:
        title = title_by_answer.get(a.get("answer_id"))
        body = _strip_html(a.get("body", ""))
        if title and body:
            pairs.append((title, body))
    return pairs


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def _seen_path(out_dir: str) -> str:
    return os.path.join(out_dir, "ingested.json")


def _load_seen(out_dir: str) -> set:
    path = _seen_path(out_dir)
    if os.path.exists(path):
        return set(json.load(open(path)))
    return set()


def _save_seen(out_dir: str, seen: set) -> None:
    json.dump(sorted(seen), open(_seen_path(out_dir), "w"))


def _key(prefix: str, text: str) -> str:
    return prefix + ":" + hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def ingest_topic(topic: str, out_dir: str, wikipedia: bool = True, stackoverflow: bool = True,
                 tags: Tuple[str, ...] = ("lua", "roblox")) -> Tuple[int, int]:
    """Fetch a topic and append new items to corpus.txt and sft.jsonl. Returns
    ``(n_corpus_docs, n_qa_pairs)`` actually added (after dedup)."""
    os.makedirs(out_dir, exist_ok=True)
    seen = _load_seen(out_dir)
    n_corpus = n_pairs = 0

    if wikipedia:
        try:
            texts = fetch_wikipedia(topic)
        except Exception as e:  # one source failing must not abort the whole run
            texts = []
            print(f"wikipedia fetch failed: {e}")
        with open(os.path.join(out_dir, "corpus.txt"), "a", encoding="utf-8") as f:
            for text in texts:
                key = _key("wiki", text)
                if key in seen:
                    continue
                f.write(text + "\n<|endoftext|>\n")
                seen.add(key)
                n_corpus += 1

    if stackoverflow:
        try:
            pairs = fetch_stackoverflow(topic, tags)
        except Exception as e:
            pairs = []
            print(f"stackoverflow fetch failed: {e}")
        with open(os.path.join(out_dir, "sft.jsonl"), "a", encoding="utf-8") as f:
            for problem, solution in pairs:
                key = _key("so", problem)
                if key in seen:
                    continue
                f.write(json.dumps({"instruction": problem, "response": solution}) + "\n")
                seen.add(key)
                n_pairs += 1

    _save_seen(out_dir, seen)
    return n_corpus, n_pairs
