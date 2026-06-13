#!/usr/bin/env python
"""Prepare data: ``python scripts/prepare_data.py --input corpus.txt``."""

import _bootstrap  # noqa: F401  (adds src/ to sys.path)
from rootllm.cli.prepare_data import main

if __name__ == "__main__":
    main()
