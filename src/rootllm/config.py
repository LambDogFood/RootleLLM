"""Typed configuration for the model, data pipeline, and training loop.

Everything is plain :mod:`dataclasses` so configs are introspectable, diffable,
and serialisable. A run is fully described by a single :class:`Config`, which can
be built from a YAML file (see ``configs/``) and tweaked with dotted CLI overrides
such as ``train.optimizer.lr=1e-4``.
"""

from __future__ import annotations

import math
from dataclasses import MISSING, asdict, dataclass, field, fields, is_dataclass
from typing import Any, Dict, List, Optional


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
@dataclass
class ModelConfig:
    """Architecture hyper-parameters for the decoder-only transformer."""

    # --- core dims ---
    vocab_size: int = 32000
    dim: int = 2048                 # model / residual-stream width
    n_layers: int = 24
    n_heads: int = 16               # number of query heads
    n_kv_heads: Optional[int] = 4   # KV heads (GQA). None -> n_heads (MHA). 1 -> MQA
    head_dim: Optional[int] = None  # defaults to dim // n_heads

    # --- feed-forward ---
    ffn_hidden_dim: Optional[int] = None  # defaults to a multiple-of-256 ~ 8/3 * dim
    ffn_multiple_of: int = 256

    # --- mixture of experts (set n_experts > 0 to enable) ---
    n_experts: int = 0              # 0 disables MoE -> dense SwiGLU FFN
    n_experts_per_token: int = 2    # top-k routing
    moe_aux_loss_coef: float = 0.01  # load-balancing loss weight

    # --- norm / activation ---
    norm_eps: float = 1e-5
    use_qk_norm: bool = True        # RMSNorm on q,k per head (stability at scale)

    # --- rope ---
    rope_theta: float = 500000.0    # large base for long context (Llama-3 style)
    rope_scaling_factor: float = 1.0  # >1.0 stretches context (linear interpolation)

    # --- training / regularisation ---
    dropout: float = 0.0
    max_seq_len: int = 8192
    gradient_checkpointing: bool = False  # recompute blocks in backward to save activation memory

    # --- misc ---
    tie_embeddings: bool = True
    initializer_range: float = 0.02

    def __post_init__(self) -> None:
        if self.n_kv_heads is None:
            self.n_kv_heads = self.n_heads
        if self.head_dim is None:
            assert self.dim % self.n_heads == 0, "dim must be divisible by n_heads"
            self.head_dim = self.dim // self.n_heads
        assert self.n_heads % self.n_kv_heads == 0, "n_heads must be divisible by n_kv_heads"
        if self.n_experts > 0:
            assert 0 < self.n_experts_per_token <= self.n_experts, (
                "n_experts_per_token must be in [1, n_experts]"
            )
        if self.ffn_hidden_dim is None:
            # SwiGLU uses 3 matrices, so 8/3*dim keeps param parity with a 4*dim GELU FFN.
            hidden = int(8 / 3 * self.dim)
            self.ffn_hidden_dim = self.ffn_multiple_of * math.ceil(hidden / self.ffn_multiple_of)


# --------------------------------------------------------------------------- #
# Optimisation
# --------------------------------------------------------------------------- #
@dataclass
class OptimizerConfig:
    lr: float = 3e-4
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    grad_clip: float = 1.0          # max global grad norm; <= 0 disables clipping


@dataclass
class ScheduleConfig:
    """Linear warmup followed by cosine decay to ``lr * min_lr_ratio``.

    ``max_steps`` is the LR horizon *and* the total number of optimiser steps the
    trainer runs, so the schedule is the single source of truth for run length.
    """

    warmup_steps: int = 100
    max_steps: int = 1000
    min_lr_ratio: float = 0.1


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
@dataclass
class DataConfig:
    """Where to find tokenised ``.bin`` corpora produced by ``prepare_data``.

    Leave ``train_path`` empty to train on a deterministic synthetic stream
    (handy for smoke tests and CI).
    """

    train_path: str = ""
    val_path: str = ""
    token_dtype: str = "uint16"     # uint16 (vocab <= 65535) or uint32
    # Synthetic-data fallback (used only when train_path is empty).
    synthetic_tokens: int = 50000
    synthetic_seed: int = 0


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
@dataclass
class TrainConfig:
    out_dir: str = "out"
    seq_len: int = 256              # training context window (<= model.max_seq_len)
    batch_size: int = 8             # micro-batch size
    grad_accum_steps: int = 1       # effective batch = batch_size * grad_accum_steps

    eval_interval: int = 250
    eval_iters: int = 50
    log_interval: int = 10
    ckpt_interval: int = 500
    sample_tokens: int = 64         # tokens to generate + log at each eval (0 disables; needs a tokenizer)
    always_save_checkpoint: bool = False  # save every ckpt_interval, not just on val improvement

    seed: int = 1337
    device: str = "auto"            # auto -> cuda > mps > cpu
    dtype: str = "auto"             # auto -> bf16 on capable cuda, else fp32
    compile: bool = False           # torch.compile the model (cuda recommended)
    memory_cap_gb: Optional[float] = None  # hard device-memory ceiling (MPS/CUDA); None = no cap

    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    schedule: ScheduleConfig = field(default_factory=ScheduleConfig)


# --------------------------------------------------------------------------- #
# Top-level bundle
# --------------------------------------------------------------------------- #
@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    data: DataConfig = field(default_factory=DataConfig)

    # -- construction --------------------------------------------------------
    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "Config":
        return _build_dataclass(cls, data or {})

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        with open(path, "r") as f:
            raw = _yaml_load(f) or {}
        return cls.from_dict(raw)

    # -- serialisation -------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_yaml(self, path: str) -> None:
        import yaml

        with open(path, "w") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False)

    # -- overrides -----------------------------------------------------------
    def apply_overrides(self, overrides: List[str]) -> "Config":
        """Apply ``a.b.c=value`` strings in place and return self.

        Values are parsed as YAML scalars, so ``1e-4``, ``true``, ``null`` and
        ``[1, 2]`` all do the sensible thing.
        """
        touched = set()
        for item in overrides:
            if "=" not in item:
                raise ValueError(f"override must look like key.path=value, got: {item!r}")
            key, _, raw = item.partition("=")
            path = [p.strip() for p in key.strip().split(".")]
            _set_nested(self, path, _yaml_load(raw))
            touched.add(".".join(path))

        # Recompute derived model fields whose inputs changed, unless the user set
        # them explicitly. Without this, overriding `model.dim` would leave the
        # already-materialised `head_dim` / `ffn_hidden_dim` stale.
        model_fields = {k[len("model."):] for k in touched if k.startswith("model.")}
        if model_fields:
            if ({"dim", "n_heads"} & model_fields) and "head_dim" not in model_fields:
                self.model.head_dim = None
            if ({"dim", "ffn_multiple_of"} & model_fields) and "ffn_hidden_dim" not in model_fields:
                self.model.ffn_hidden_dim = None
            self.model.__post_init__()
        return self


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
_YAML_LOADER = None


def _yaml_load(stream: Any) -> Any:
    """YAML load with a corrected float resolver.

    PyYAML's stock implicit resolver does not recognise ``1e-4`` as a float
    (it requires a decimal point), which makes ``lr=1e-4`` silently become a
    string. This loader fixes the resolver so both config files and CLI
    overrides parse scientific notation as numbers.
    """
    global _YAML_LOADER
    if _YAML_LOADER is None:
        import re

        import yaml

        class _Loader(yaml.SafeLoader):
            pass

        _Loader.add_implicit_resolver(
            "tag:yaml.org,2002:float",
            re.compile(
                r"""^(?:
                    [-+]?[0-9][0-9_]*\.[0-9_]*(?:[eE][-+]?[0-9]+)?
                   |[-+]?\.[0-9_]+(?:[eE][-+]?[0-9]+)?
                   |[-+]?[0-9][0-9_]*[eE][-+]?[0-9]+
                   |[-+]?\.(?:inf|Inf|INF)
                   |\.(?:nan|NaN|NAN))$""",
                re.X,
            ),
            list("-+0123456789."),
        )
        _YAML_LOADER = _Loader

    import yaml

    return yaml.load(stream, Loader=_YAML_LOADER)


def _default_for(f) -> Any:
    if f.default is not MISSING:
        return f.default
    if f.default_factory is not MISSING:  # type: ignore[misc]
        return f.default_factory()        # type: ignore[misc]
    return None


def _build_dataclass(cls, data: Dict[str, Any]):
    """Recursively build a (possibly nested) dataclass from a plain dict."""
    if not isinstance(data, dict):
        raise TypeError(f"expected a mapping for {cls.__name__}, got {type(data).__name__}")
    known = {f.name: f for f in fields(cls)}
    kwargs: Dict[str, Any] = {}
    for key, value in data.items():
        if key not in known:
            raise KeyError(f"unknown config key {key!r} for {cls.__name__}")
        default = _default_for(known[key])
        if is_dataclass(default) and isinstance(value, dict):
            kwargs[key] = _build_dataclass(type(default), value)
        else:
            kwargs[key] = value
    return cls(**kwargs)


def _set_nested(obj: Any, path: List[str], value: Any) -> None:
    for part in path[:-1]:
        if not hasattr(obj, part):
            raise KeyError(f"unknown config section {part!r}")
        obj = getattr(obj, part)
    leaf = path[-1]
    if not hasattr(obj, leaf):
        raise KeyError(f"unknown config key {leaf!r}")
    setattr(obj, leaf, value)
