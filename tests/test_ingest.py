"""Ingester orchestration (network calls are mocked)."""

from __future__ import annotations

import json

import rootllm.data.ingest as ing


def test_strip_html():
    html = "<p>Use <code>game:GetService(\"Players\")</code> &amp; enjoy.</p>"
    text = ing._strip_html(html)
    assert "game:GetService" in text and "&amp;" not in text and "<" not in text


def test_ingest_writes_corpus_and_pairs_and_dedups(tmp_path, monkeypatch):
    monkeypatch.setattr(ing, "fetch_wikipedia", lambda topic, **k: ["# Topic\n\nbody text"])
    monkeypatch.setattr(
        ing, "fetch_stackoverflow", lambda topic, *a, **k: [("How do I X?", "Do Y like this.")]
    )

    n_corpus, n_pairs = ing.ingest_topic("anything", str(tmp_path))
    assert n_corpus == 1 and n_pairs == 1

    assert "body text" in (tmp_path / "corpus.txt").read_text()
    rec = json.loads((tmp_path / "sft.jsonl").read_text().splitlines()[0])
    assert rec["instruction"] == "How do I X?" and rec["response"] == "Do Y like this."

    # second run: identical content is deduped, nothing added
    n_corpus2, n_pairs2 = ing.ingest_topic("anything", str(tmp_path))
    assert n_corpus2 == 0 and n_pairs2 == 0


def test_ingest_resilient_to_source_failure(tmp_path, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("API down")

    monkeypatch.setattr(ing, "fetch_wikipedia", boom)
    monkeypatch.setattr(ing, "fetch_stackoverflow", lambda topic, *a, **k: [("q", "a")])
    # wikipedia failing must not stop stackoverflow from being ingested
    n_corpus, n_pairs = ing.ingest_topic("anything", str(tmp_path))
    assert n_corpus == 0 and n_pairs == 1
