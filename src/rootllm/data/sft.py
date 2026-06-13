"""Supervised fine-tuning (instruction tuning).

Turns the base next-token model into one that *answers* by training on
instruction→response pairs, with the loss computed **only on the response**
tokens. Prompt tokens are masked with ``IGNORE_INDEX`` (-100), which the model's
cross-entropy already ignores — so SFT needs no changes to the model or trainer,
just this dataset.

Input is JSONL, one record per line, accepting either schema::

    {"instruction": "...", "input": "...", "output": "..."}   # Alpaca-style
    {"prompt": "...", "response": "..."}
"""

from __future__ import annotations

import json
from typing import List, Tuple

import torch

from ..model.transformer import IGNORE_INDEX
from .dataset import TokenDataset

# Simple, explicit chat template. The response follows the final marker; loss is
# computed from there on.
_WITH_INPUT = "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n"
_NO_INPUT = "### Instruction:\n{instruction}\n\n### Response:\n"


def format_prompt(instruction: str, input_text: str = "") -> str:
    """Render the prompt portion of an example (everything before the response)."""
    if input_text:
        return _WITH_INPUT.format(instruction=instruction, input=input_text)
    return _NO_INPUT.format(instruction=instruction)


def _record_fields(rec: dict) -> Tuple[str, str, str]:
    instruction = rec.get("instruction") or rec.get("prompt") or ""
    response = rec.get("output")
    if response is None:
        response = rec.get("response", "")
    return instruction, rec.get("input", ""), response


class SFTDataset(TokenDataset):
    """Instruction/response examples with prompt-masked next-token targets."""

    def __init__(self, examples: List[Tuple[List[int], int]], pad_id: int):
        # each example is (full_token_ids, prompt_len)
        self.examples = examples
        self.pad_id = pad_id
        self._n = len(examples)

    @classmethod
    def from_jsonl(cls, path: str, tokenizer, add_bos: bool = True) -> "SFTDataset":
        bos = getattr(tokenizer, "bos_id", -1)
        eos = tokenizer.eos_id
        examples: List[Tuple[List[int], int]] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                instruction, input_text, response = _record_fields(json.loads(line))
                prompt_ids = tokenizer.encode(format_prompt(instruction, input_text))
                if add_bos and isinstance(bos, int) and bos >= 0:
                    prompt_ids = [bos] + prompt_ids
                response_ids = tokenizer.encode(response)
                if isinstance(eos, int) and eos >= 0:
                    response_ids = response_ids + [eos]
                examples.append((prompt_ids + response_ids, len(prompt_ids)))
        if not examples:
            raise ValueError(f"no examples found in {path}")
        pad_id = eos if isinstance(eos, int) and eos >= 0 else 0
        return cls(examples, pad_id)

    def get_batch(
        self,
        batch_size: int,
        seq_len: int,
        device: torch.device,
        generator: torch.Generator = None,
    ):
        idx = torch.randint(self._n, (batch_size,), generator=generator).tolist()
        xs, ys = [], []
        for i in idx:
            full, prompt_len = self.examples[i]
            full = full[: seq_len + 1]
            x = full[:-1]
            # target at position j is full[j+1]; it belongs to the prompt (and is
            # masked) while j+1 < prompt_len. Right-pad to seq_len with masked pad.
            labels = [IGNORE_INDEX if (j + 1) < prompt_len else t for j, t in enumerate(full[1:])]
            pad = seq_len - len(x)
            if pad > 0:
                x = x + [self.pad_id] * pad
                labels = labels + [IGNORE_INDEX] * pad
            xs.append(x)
            ys.append(labels)
        x = torch.tensor(xs, dtype=torch.long)
        y = torch.tensor(ys, dtype=torch.long)
        return x.to(device), y.to(device)
