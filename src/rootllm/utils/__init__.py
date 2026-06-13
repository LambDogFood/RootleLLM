"""Cross-cutting helpers: device/dtype resolution, seeding, logging."""

from __future__ import annotations

from .device import (
    AmpSettings,
    apply_memory_cap,
    memory_cap_fraction,
    resolve_amp,
    resolve_device,
)
from .logging import JsonlLogger, get_logger
from .seed import set_seed

__all__ = [
    "AmpSettings",
    "apply_memory_cap",
    "memory_cap_fraction",
    "resolve_amp",
    "resolve_device",
    "JsonlLogger",
    "get_logger",
    "set_seed",
]
