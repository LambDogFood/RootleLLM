"""Make ``src/`` importable when running these scripts without installing the package.

Allows ``python scripts/train.py ...`` to work straight from a checkout. After
``pip install -e .`` the console entry points (``rootllm-train`` etc.) are the
preferred interface and this shim is unnecessary.
"""

from __future__ import annotations

import pathlib
import sys

_SRC = pathlib.Path(__file__).resolve().parent.parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
