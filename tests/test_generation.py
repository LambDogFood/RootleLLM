from __future__ import annotations

import torch

from rootllm.generation import apply_repetition_penalty, filter_logits, generate
from rootllm.model import Transformer
from tests.helpers import make_model_config


def test_generate_grows_sequence():
    cfg = make_model_config()
    model = Transformer(cfg).eval()
    prompt = torch.randint(0, cfg.vocab_size, (2, 4))
    out = generate(model, prompt, max_new_tokens=6, temperature=1.0, top_k=10)
    assert out.shape == (2, 10)
    # the prompt is preserved as a prefix
    assert torch.equal(out[:, :4], prompt)


def test_greedy_is_deterministic():
    cfg = make_model_config()
    model = Transformer(cfg).eval()
    prompt = torch.randint(0, cfg.vocab_size, (1, 4))
    a = generate(model, prompt.clone(), max_new_tokens=8, temperature=0.0)
    b = generate(model, prompt.clone(), max_new_tokens=8, temperature=0.0)
    assert torch.equal(a, b)


def test_eos_stops_generation():
    cfg = make_model_config()
    model = Transformer(cfg).eval()
    prompt = torch.randint(0, cfg.vocab_size, (1, 2))
    # Greedy decoding is deterministic, so find the token the model emits first,
    # then make *that* the eos id: generation must stop right after emitting it.
    first = generate(model, prompt.clone(), max_new_tokens=1, temperature=0.0)[0, -1].item()
    out = generate(model, prompt.clone(), max_new_tokens=20, temperature=0.0, eos_token_id=first)
    assert out.shape[1] == prompt.shape[1] + 1  # stopped right after emitting eos


def test_filter_logits_top_k():
    logits = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    filtered = filter_logits(logits, top_k=2)
    # only the top-2 (values 3,4) survive; the rest are -inf
    assert torch.isinf(filtered[0, 0]) and torch.isinf(filtered[0, 1])
    assert filtered[0, 2].item() == 3.0 and filtered[0, 3].item() == 4.0


def test_filter_logits_top_p_keeps_at_least_one():
    logits = torch.tensor([[0.1, 0.2, 0.3, 10.0]])
    filtered = filter_logits(logits, top_p=0.5)
    # the dominating token must always remain selectable
    assert torch.isfinite(filtered[0, 3])


def test_min_p_keeps_dominant_token():
    logits = torch.tensor([[0.1, 0.2, 0.3, 10.0]])
    filtered = filter_logits(logits, min_p=0.5)
    # only the dominant token clears the min-p (0.5 * top prob) bar
    assert torch.isfinite(filtered[0, 3])
    assert torch.isinf(filtered[0, 0])


def test_repetition_penalty_lowers_seen_token_logits():
    logits = torch.tensor([[1.0, 1.0, 1.0, 1.0]])
    seen = torch.tensor([[2]])  # token 2 already generated
    out = apply_repetition_penalty(logits.clone(), seen, penalty=2.0)
    # positive logit for the seen token is divided down; others unchanged
    assert out[0, 2].item() == 0.5
    assert out[0, 0].item() == 1.0


def test_repetition_penalty_identity_at_one():
    logits = torch.randn(2, 10)
    seen = torch.randint(0, 10, (2, 5))
    assert torch.equal(apply_repetition_penalty(logits.clone(), seen, 1.0), logits)


def test_generate_accepts_new_sampling_args():
    cfg = make_model_config()
    model = Transformer(cfg).eval()
    prompt = torch.randint(0, cfg.vocab_size, (1, 4))
    out = generate(model, prompt, max_new_tokens=6, temperature=0.9,
                   top_k=20, min_p=0.05, repetition_penalty=1.2)
    assert out.shape == (1, 10)


def test_generate_returns_confidence():
    cfg = make_model_config()
    model = Transformer(cfg).eval()
    prompt = torch.randint(0, cfg.vocab_size, (1, 4))
    out, conf = generate(model, prompt, max_new_tokens=6, return_confidence=True)
    assert out.shape == (1, 10)
    assert conf.shape == (1,)
    assert 0.0 <= float(conf[0]) <= 1.0


def test_generation_does_not_leave_model_in_eval():
    cfg = make_model_config()
    model = Transformer(cfg)
    model.train()
    generate(model, torch.randint(0, cfg.vocab_size, (1, 3)), max_new_tokens=2)
    assert model.training  # training mode is restored
