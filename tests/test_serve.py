"""The inference server: a real client -> server -> generation round-trip."""

from __future__ import annotations

import json
import threading
import urllib.request

import pytest

from rootllm.cli import query as query_cli
from rootllm.config import Config
from rootllm.model import Transformer
from rootllm.serve.server import ModelService, build_server
from rootllm.training.checkpoint import save_checkpoint
from tests.helpers import make_model_config


@pytest.fixture
def ckpt(tmp_path):
    cfg = Config()
    cfg.model = make_model_config(vocab_size=258, max_seq_len=64)
    model = Transformer(cfg.model)
    path = str(tmp_path / "ckpt.pt")
    save_checkpoint(path, model, optimizer=None, config=cfg, step=7)
    return path


def test_model_service_completes(ckpt):
    service = ModelService(ckpt, device="cpu")
    out = service.complete({"prompt": "hi", "max_new_tokens": 5, "temperature": 0.0})
    assert "completion" in out and "text" in out
    assert out["text"].startswith("hi")


def test_query_url_normalisation():
    assert query_cli._base_url("1.2.3.4") == "http://1.2.3.4:8000"
    assert query_cli._base_url("1.2.3.4:9000") == "http://1.2.3.4:9000"
    assert query_cli._base_url("http://host:8000/") == "http://host:8000"


def test_http_roundtrip_and_query_cli(ckpt, capsys):
    service = ModelService(ckpt, device="cpu")
    httpd = build_server(service, host="127.0.0.1", port=0)  # port 0 -> ephemeral
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        port = httpd.server_address[1]

        # GET / -> info
        info = json.loads(urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=10).read())
        assert info["status"] == "ok" and info["step"] == 7

        # POST /generate directly
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/generate",
            data=json.dumps({"prompt": "hi", "max_new_tokens": 4, "temperature": 0.0}).encode(),
            headers={"Content-Type": "application/json"},
        )
        out = json.loads(urllib.request.urlopen(req, timeout=10).read())
        assert "completion" in out

        # the laptop-side client CLI hitting the same server
        query_cli.main(["--host", f"127.0.0.1:{port}", "--prompt", "hi",
                        "--max-new-tokens", "4", "--temperature", "0.0"])
        assert capsys.readouterr().out.strip() != ""
    finally:
        httpd.shutdown()
