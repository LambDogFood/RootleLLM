"""A tiny, dependency-free HTTP inference server.

Loads a checkpoint once, keeps the model warm on the device, and answers
``POST /generate`` requests with a JSON body of sampling parameters. A single
lock serialises generation (one GPU, one request at a time), which is exactly
right for a personal LAN server. No framework — just the standard library.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, Optional

import torch

from ..data.sft import format_prompt
from ..data.tokenizer import ByteTokenizer, load_tokenizer
from ..generation import generate
from ..model import Transformer
from ..training.checkpoint import load_checkpoint
from ..utils.device import resolve_device
from ..utils.logging import get_logger


def _opt_int(v) -> Optional[int]:
    return int(v) if v is not None else None


def _opt_float(v) -> Optional[float]:
    return float(v) if v is not None else None


class ModelService:
    """Holds a checkpoint on the device and turns request dicts into completions."""

    def __init__(
        self,
        ckpt: str,
        tokenizer_path: Optional[str] = None,
        device: str = "auto",
        default_max_new_tokens: int = 128,
    ):
        self.log = get_logger()
        self.device = resolve_device(device)
        payload = load_checkpoint(ckpt, map_location=self.device)
        cfg = payload["config_obj"]
        self.model = Transformer(cfg.model)
        self.model.load_state_dict(payload["model"])
        self.model.to(self.device).eval()
        self.tokenizer = load_tokenizer(tokenizer_path) if tokenizer_path else ByteTokenizer()
        self.default_max_new_tokens = default_max_new_tokens
        self._lock = threading.Lock()
        self.info = {
            "params": self.model.num_params(),
            "device": str(self.device),
            "vocab_size": cfg.model.vocab_size,
            "ckpt": ckpt,
            "step": payload.get("step"),
        }

    def complete(self, req: Dict[str, Any]) -> Dict[str, Any]:
        prompt = req.get("prompt", "")
        if req.get("chat"):
            prompt = format_prompt(prompt)

        ids = self.tokenizer.encode(prompt)
        if not ids:
            bos = getattr(self.tokenizer, "bos_id", -1)
            ids = [bos if isinstance(bos, int) and bos >= 0 else 0]
        input_ids = torch.tensor([ids], dtype=torch.long, device=self.device)
        eos = self.tokenizer.eos_id if self.tokenizer.eos_id >= 0 else None

        with self._lock:  # one generation at a time on the GPU
            out = generate(
                self.model,
                input_ids,
                max_new_tokens=int(req.get("max_new_tokens", self.default_max_new_tokens)),
                temperature=float(req.get("temperature", 0.8)),
                top_k=_opt_int(req.get("top_k")),
                top_p=_opt_float(req.get("top_p")),
                min_p=_opt_float(req.get("min_p")),
                repetition_penalty=float(req.get("repetition_penalty", 1.0)),
                eos_token_id=eos,
            )
        text = self.tokenizer.decode(out[0].tolist())
        completion = text[len(prompt):] if text.startswith(prompt) else text
        return {"prompt": prompt, "completion": completion, "text": text}


def _make_handler(service: ModelService):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, code: int, obj: Dict[str, Any]) -> None:
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):  # noqa: N802 (stdlib API name)
            if self.path.rstrip("/") in ("", "/health"):
                self._send(200, {"status": "ok", **service.info})
            else:
                self._send(404, {"error": "GET / for info, POST /generate to sample"})

        def do_POST(self):  # noqa: N802
            if self.path.rstrip("/") != "/generate":
                self._send(404, {"error": "POST /generate"})
                return
            try:
                n = int(self.headers.get("Content-Length", 0) or 0)
                req = json.loads(self.rfile.read(n) or b"{}")
                self._send(200, service.complete(req))
            except Exception as e:  # never crash the server on a bad request
                self._send(400, {"error": str(e)})

        def log_message(self, *args):  # keep the console quiet
            pass

    return Handler


def build_server(service: ModelService, host: str = "0.0.0.0", port: int = 8000) -> ThreadingHTTPServer:
    return ThreadingHTTPServer((host, port), _make_handler(service))


def serve(
    ckpt: str,
    tokenizer_path: Optional[str] = None,
    host: str = "0.0.0.0",
    port: int = 8000,
    device: str = "auto",
) -> None:
    service = ModelService(ckpt, tokenizer_path, device)
    httpd = build_server(service, host, port)
    get_logger().info(
        "serving %s (%.1fM params, %s) on http://%s:%d  — POST /generate",
        ckpt, service.info["params"] / 1e6, service.info["device"], host, port,
    )
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        get_logger().info("shutting down")
        httpd.shutdown()
