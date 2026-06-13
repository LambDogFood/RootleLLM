from __future__ import annotations

import io
import tarfile

import rootllm.data.luau as luau


def _fake_tarball(owner, repo):
    """A tarball with one Lua file and one non-Lua file."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for name, content in [
            (f"{repo}-main/src/Main.lua", b"local part = Instance.new(\"Part\")\n"),
            (f"{repo}-main/README.md", b"not code"),
        ]:
            info = tarfile.TarInfo(name)
            info.size = len(content)
            tar.addfile(info, io.BytesIO(content))
    return buf.getvalue()


def test_fetch_luau_corpus_extracts_only_lua(tmp_path, monkeypatch):
    monkeypatch.setattr(luau, "_download_tarball", _fake_tarball)
    out, n_files = luau.fetch_luau_corpus(
        str(tmp_path / "corpus.txt"),
        repos=[("a", "b"), ("c", "d")],
        cache_dir=str(tmp_path / "cache"),
    )
    assert n_files == 2  # one .lua from each repo
    text = open(out).read()
    assert "Instance.new" in text          # the Lua content is included
    assert "not code" not in text          # the README is skipped
    assert "<|endoftext|>" in text         # files are separated by the eot marker


def test_repo_list_is_nonempty():
    assert len(luau.LUAU_REPOS) >= 3
    assert all(isinstance(o, str) and isinstance(r, str) for o, r in luau.LUAU_REPOS)
