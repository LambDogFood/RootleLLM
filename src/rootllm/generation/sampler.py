"""Autoregressive decoding with temperature / top-k / top-p sampling.

Kept separate from the model so the architecture stays focused on the forward
pass while decoding strategy can evolve independently. Generation uses the KV
cache, so each new token costs one single-token forward instead of re-running the
whole prefix.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F


def filter_logits(
    logits: torch.Tensor,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    min_p: Optional[float] = None,
) -> torch.Tensor:
    """Mask out improbable tokens by setting their logits to ``-inf``.

    ``top_k`` keeps the k highest-logit tokens. ``top_p`` (nucleus sampling) keeps
    the smallest set of tokens whose cumulative probability exceeds ``p``.
    ``min_p`` keeps tokens whose probability is at least ``min_p`` times the top
    token's probability — a scale-adaptive cutoff that adjusts to how confident
    the model is. They can be combined.
    """
    if top_k is not None:
        k = min(top_k, logits.size(-1))
        kth = logits.topk(k, dim=-1).values[..., -1, None]
        logits = logits.masked_fill(logits < kth, float("-inf"))

    if top_p is not None:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
        cum = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        remove = cum > top_p
        # Shift right so the first token crossing the threshold is kept.
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False
        idx_remove = remove.scatter(-1, sorted_idx, remove)
        logits = logits.masked_fill(idx_remove, float("-inf"))

    if min_p is not None:
        probs = F.softmax(logits, dim=-1)
        top = probs.max(dim=-1, keepdim=True).values
        logits = logits.masked_fill(probs < min_p * top, float("-inf"))

    return logits


def apply_repetition_penalty(
    logits: torch.Tensor,
    generated: torch.Tensor,
    penalty: float,
) -> torch.Tensor:
    """Discourage repeating already-generated tokens (Keskar et al., 2019).

    For each token already in ``generated``, push its logit toward zero by
    dividing positive logits / multiplying negative logits by ``penalty`` (>1).
    ``logits`` is ``(B, V)``; ``generated`` is ``(B, T)``.
    """
    if penalty == 1.0:
        return logits
    score = torch.gather(logits, 1, generated)
    score = torch.where(score < 0, score * penalty, score / penalty)
    return logits.scatter(1, generated, score)


@torch.no_grad()
def generate(
    model,
    input_ids: torch.Tensor,
    max_new_tokens: int = 64,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    min_p: Optional[float] = None,
    repetition_penalty: float = 1.0,
    eos_token_id: Optional[int] = None,
    return_confidence: bool = False,
):
    """Sample up to ``max_new_tokens`` continuations for each row of ``input_ids``.

    ``temperature == 0`` selects greedy (argmax) decoding. ``repetition_penalty``
    (>1) discourages repeating tokens already in the sequence. Returns the full
    sequence including the prompt, shape ``(B, T + generated)``.

    With ``return_confidence=True`` also returns a per-row confidence in ``[0, 1]``
    — the mean, over generated steps, of the model's top raw next-token
    probability. High = peaked/certain distributions; low = the model is guessing.
    """
    was_training = model.training
    model.eval()
    _, T = input_ids.shape

    # Prefill: run the prompt once to populate the KV cache for every layer.
    logits, kv_caches = model.forward(
        input_ids, kv_caches=[None] * model.cfg.n_layers, start_pos=0
    )
    pos = T
    confidences = []

    for _ in range(max_new_tokens):
        raw_logits = logits[:, -1, :]
        if return_confidence:
            # Certainty of the *unmodified* distribution (before temp/penalty/filter).
            confidences.append(F.softmax(raw_logits.float(), dim=-1).max(dim=-1).values)

        next_logits = apply_repetition_penalty(raw_logits, input_ids, repetition_penalty)
        next_logits = next_logits / max(temperature, 1e-6)
        next_logits = filter_logits(next_logits, top_k, top_p, min_p)

        if temperature == 0.0:
            next_tok = next_logits.argmax(dim=-1, keepdim=True)
        else:
            probs = F.softmax(next_logits, dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1)

        input_ids = torch.cat([input_ids, next_tok], dim=1)
        if eos_token_id is not None and (next_tok == eos_token_id).all():
            break

        logits, kv_caches = model.forward(next_tok, kv_caches=kv_caches, start_pos=pos)
        pos += 1

    if was_training:
        model.train()

    if return_confidence:
        if confidences:
            conf = torch.stack(confidences, dim=1).mean(dim=1)
        else:
            conf = torch.ones(input_ids.shape[0], device=input_ids.device)
        return input_ids, conf
    return input_ids
