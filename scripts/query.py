#!/usr/bin/env python
"""Query a server: ``python scripts/query.py --host 192.168.1.50 --prompt "Hi"``."""

import _bootstrap  # noqa: F401  (adds src/ to sys.path)
from rootllm.cli.query import main

if __name__ == "__main__":
    main()
