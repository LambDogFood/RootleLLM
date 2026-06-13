#!/usr/bin/env python
"""Serve a checkpoint: ``python scripts/serve.py --ckpt out/rtx5070/ckpt.pt``."""

import _bootstrap  # noqa: F401  (adds src/ to sys.path)
from rootllm.cli.serve import main

if __name__ == "__main__":
    main()
