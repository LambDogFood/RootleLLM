"""rootllm — a readable, modular decoder-only transformer with training tooling.

Public surface::

    from rootllm import Transformer, ModelConfig, Config
    from rootllm import generate, Trainer

The package is organised by concern:

* :mod:`rootllm.config`     — typed configuration dataclasses
* :mod:`rootllm.model`      — the transformer and its building blocks
* :mod:`rootllm.data`       — tokenisers and token datasets
* :mod:`rootllm.training`   — optimiser, schedule, checkpoints, trainer
* :mod:`rootllm.generation` — autoregressive sampling
"""

from __future__ import annotations

from .config import (
    Config,
    DataConfig,
    ModelConfig,
    OptimizerConfig,
    ScheduleConfig,
    TrainConfig,
)
from .generation import generate
from .model import Transformer
from .training import Trainer

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "Config",
    "ModelConfig",
    "TrainConfig",
    "DataConfig",
    "OptimizerConfig",
    "ScheduleConfig",
    "Transformer",
    "generate",
    "Trainer",
]
