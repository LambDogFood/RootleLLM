from __future__ import annotations

import os

import pytest

import rootllm.data.download as dl


def test_registry_has_expected_datasets():
    assert "shakespeare" in dl.DATASETS
    assert all(isinstance(v, list) and v for v in dl.DATASETS.values())


def test_download_caches(tmp_path, monkeypatch):
    calls = {"n": 0}

    def fake_download(url, dest):
        calls["n"] += 1
        with open(dest, "w") as f:
            f.write("hello")

    monkeypatch.setattr(dl, "_download", fake_download)

    paths = dl.download_dataset("shakespeare", cache_dir=str(tmp_path))
    assert len(paths) == 1 and os.path.exists(paths[0])
    assert calls["n"] == 1

    # second call hits the cache, no re-download
    dl.download_dataset("shakespeare", cache_dir=str(tmp_path))
    assert calls["n"] == 1


def test_unknown_dataset_raises():
    with pytest.raises(KeyError):
        dl.download_dataset("does-not-exist")
