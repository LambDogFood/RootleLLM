#!/usr/bin/env python
"""Run generation: ``python scripts/generate.py --ckpt out/ckpt.pt --prompt "Hi"``."""

import _bootstrap  # noqa: F401  (adds src/ to sys.path)
from rootllm.cli.generate import main

if __name__ == "__main__":
    main()
