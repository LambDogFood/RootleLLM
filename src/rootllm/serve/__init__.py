"""HTTP inference server: keep a model warm on the GPU and answer prompts."""

from __future__ import annotations

from .server import ModelService, build_server, serve

__all__ = ["ModelService", "build_server", "serve"]
