"""The full decoder-only transformer."""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from ..config import ModelConfig
from .attention import KVCache
from .block import Block
from .norm import RMSNorm
from .rope import precompute_rope_cache

# Cross-entropy ignores this target id (e.g. padding / prompt tokens).
IGNORE_INDEX = -100


def _run_block(block: Block, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    """Block call for the checkpointed training path: drop the (unused) KV cache so
    those k/v tensors aren't retained as graph outputs, which would waste the memory
    checkpointing is meant to save."""
    x, _cache, aux = block(x, cos, sin, None)
    return x, aux


class Transformer(nn.Module):
    """A Llama/Mixtral-class decoder-only transformer.

    Stacks token embeddings, ``n_layers`` pre-norm :class:`Block` s, a final norm,
    and an LM head (optionally weight-tied to the embedding). The RoPE table is a
    non-persistent buffer rebuilt on construction, so it never bloats checkpoints.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.dropout = nn.Dropout(cfg.dropout)
        self.layers = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.norm = RMSNorm(cfg.dim, cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.tok_emb.weight

        cos, sin = precompute_rope_cache(
            cfg.head_dim, cfg.max_seq_len, cfg.rope_theta, cfg.rope_scaling_factor
        )
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self.apply(self._init_weights)
        # GPT-2-style depth scaling: shrink residual-projection init by 1/sqrt(2*L)
        # so the residual stream variance doesn't grow with depth.
        for name, p in self.named_parameters():
            if name.endswith("wo.weight") or name.endswith("w2.weight"):
                nn.init.normal_(p, mean=0.0, std=cfg.initializer_range / math.sqrt(2 * cfg.n_layers))

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=self.cfg.initializer_range)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.cfg.initializer_range)

    # ------------------------------------------------------------------ #
    # Forward
    # ------------------------------------------------------------------ #
    def forward(
        self,
        input_ids: torch.Tensor,                       # (B, T)
        targets: Optional[torch.Tensor] = None,        # (B, T)
        kv_caches: Optional[List[Optional[KVCache]]] = None,
        start_pos: int = 0,
    ) -> Tuple[torch.Tensor, object]:
        """Run the model.

        Training (``targets`` given): returns ``(logits, loss)`` where ``loss``
        includes any MoE auxiliary losses.

        Inference: returns ``(logits, new_kv_caches)``. When ``kv_caches`` is
        provided (incremental decoding) only the final position's logits are
        computed, saving the LM-head matmul over the prefix.
        """
        B, T = input_ids.shape
        x = self.dropout(self.tok_emb(input_ids))

        cos = self.rope_cos[start_pos:start_pos + T]
        sin = self.rope_sin[start_pos:start_pos + T]

        # Recompute blocks in backward only during training (it needs grad, and the
        # KV cache for incremental decoding is incompatible with it).
        use_checkpoint = (
            self.cfg.gradient_checkpointing
            and self.training
            and torch.is_grad_enabled()
            and kv_caches is None
        )

        new_caches: List[KVCache] = []
        total_aux: Optional[torch.Tensor] = None
        for i, layer in enumerate(self.layers):
            cache = kv_caches[i] if kv_caches is not None else None
            if use_checkpoint:
                x, aux = checkpoint(_run_block, layer, x, cos, sin, use_reentrant=False)
            else:
                x, nc, aux = layer(x, cos, sin, cache)
                new_caches.append(nc)
            if aux is not None:
                total_aux = aux if total_aux is None else total_aux + aux

        x = self.norm(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=IGNORE_INDEX,
            )
            if total_aux is not None:
                loss = loss + total_aux
            return logits, loss

        # Inference: only the last position matters when decoding with a cache.
        logits = self.lm_head(x[:, -1:, :]) if kv_caches is not None else self.lm_head(x)
        return logits, new_caches

    # ------------------------------------------------------------------ #
    # Convenience
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def generate(self, *args, **kwargs) -> torch.Tensor:
        """Autoregressively sample tokens. See :func:`rootllm.generation.generate`."""
        from ..generation import generate as _generate  # lazy import avoids a cycle

        return _generate(self, *args, **kwargs)

    def num_params(self, non_embedding: bool = True) -> int:
        """Total parameter count.

        With tied embeddings the LM head shares the embedding matrix, so by
        default it is counted once and the (shared) token-embedding params are
        excluded to report the "compute" parameter count.
        """
        n = sum(p.numel() for p in self.parameters())
        if non_embedding and self.cfg.tie_embeddings:
            n -= self.tok_emb.weight.numel()
        return n
