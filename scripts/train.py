#!/usr/bin/env python
"""Run training: ``python scripts/train.py --config configs/small.yaml``."""

import _bootstrap  # noqa: F401  (adds src/ to sys.path)
from rootllm.cli.train import main

if __name__ == "__main__":
    main()
