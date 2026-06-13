"""Checkpoint serialisation.

A checkpoint is a single ``.pt`` file bundling the model weights, optimiser state,
the full :class:`~rootllm.config.Config` (as a dict, so a run is fully
reproducible), and bookkeeping such as the step count and best validation loss.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

import torch

from ..config import Config


def save_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    config: Config,
    step: int,
    best_val_loss: float = float("inf"),
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """Atomically write a checkpoint to ``path``."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    payload: Dict[str, Any] = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "config": config.to_dict(),
        "step": step,
        "best_val_loss": best_val_loss,
    }
    if extra:
        payload.update(extra)
    # Write to a temp file then rename so an interrupted save can't corrupt the
    # previous good checkpoint.
    tmp = path + ".tmp"
    torch.save(payload, tmp)
    os.replace(tmp, path)


def load_checkpoint(
    path: str,
    map_location: Optional[Any] = None,
) -> Dict[str, Any]:
    """Load a checkpoint dict and rebuild its :class:`Config`.

    Returns the raw payload with an added ``"config_obj"`` key holding a
    reconstructed :class:`Config`. Pass the returned ``model``/``optimizer``
    state dicts to ``load_state_dict`` yourself.
    """
    payload = torch.load(path, map_location=map_location, weights_only=False)
    payload["config_obj"] = Config.from_dict(payload.get("config"))
    return payload
