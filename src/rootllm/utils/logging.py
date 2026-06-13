"""Console logging plus a tiny JSONL metrics sink."""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, Optional

_CONFIGURED = False


def get_logger(name: str = "rootllm", level: int = logging.INFO) -> logging.Logger:
    """Return a process-wide logger with a single clean stream handler."""
    global _CONFIGURED
    logger = logging.getLogger(name)
    if not _CONFIGURED:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s %(message)s", "%H:%M:%S"))
        root = logging.getLogger("rootllm")
        root.addHandler(handler)
        root.setLevel(level)
        root.propagate = False
        _CONFIGURED = True
    return logger


class JsonlLogger:
    """Append-only JSONL writer for training metrics (one JSON object per line)."""

    def __init__(self, path: Optional[str]):
        self.path = path
        if path:
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            # Truncate any stale log from a previous run sharing this out_dir.
            open(path, "w").close()

    def log(self, record: Dict[str, Any]) -> None:
        if not self.path:
            return
        with open(self.path, "a") as f:
            f.write(json.dumps(record) + "\n")
