from __future__ import annotations

import logging
import math

from rootllm.data.dataset import RandomTokenDataset
from rootllm.data.tokenizer import ByteTokenizer
from rootllm.model import Transformer
from rootllm.training import Trainer
from rootllm.training.trainer import _perplexity
from tests.helpers import make_model_config


def test_perplexity_is_exp_of_loss():
    assert math.isclose(_perplexity(0.0), 1.0)
    assert math.isclose(_perplexity(2.0), math.exp(2.0))
    # clamps to avoid overflow on a wildly untrained model
    assert math.isfinite(_perplexity(1000.0))


def test_sample_logging_emits_a_sample(tiny_config):
    cfg = tiny_config
    cfg.model = make_model_config(vocab_size=258, max_seq_len=64)  # match ByteTokenizer vocab
    cfg.train.sample_tokens = 8
    model = Transformer(cfg.model)
    data = RandomTokenDataset(258, n_tokens=500, seed=0)
    trainer = Trainer(cfg, model, data, data, tokenizer=ByteTokenizer())

    messages = []
    handler = logging.Handler()
    handler.emit = lambda record: messages.append(record.getMessage())
    logging.getLogger("rootllm").addHandler(handler)
    try:
        trainer._log_sample()
    finally:
        logging.getLogger("rootllm").removeHandler(handler)

    assert any("sample:" in m for m in messages)


def test_no_sample_without_tokenizer(tiny_config):
    # sample logging is a no-op when no tokenizer is supplied (e.g. synthetic data)
    model = Transformer(tiny_config.model)
    data = RandomTokenDataset(tiny_config.model.vocab_size, n_tokens=500, seed=0)
    trainer = Trainer(tiny_config, model, data, data, tokenizer=None)
    trainer._log_sample()  # must not raise
