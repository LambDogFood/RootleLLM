# rootllm

A modern, readable decoder-only transformer (Llama / Mixtral / DeepSeek-class)
packaged for training and inference. The architecture lives in small,
single-responsibility modules; the training loop, data pipeline, tokenizer,
checkpointing, and sampling each have their own home; and a `pytest` suite covers
correctness (including a KV-cache equivalence check and a one-batch overfit test).

## Architecture

All components are current state-of-the-art and fully typed — no placeholders:

- **RMSNorm** with pre-norm placement
- **Rotary Position Embeddings (RoPE)** with optional linear context scaling
- **Grouped-Query Attention (GQA / MQA)** over a FlashAttention path via SDPA
- Optional **QK-Norm** (per-head RMSNorm on q/k) for training stability
- **KV cache** for O(1)-per-token incremental decoding
- **SwiGLU** feed-forward
- Optional **Mixture-of-Experts** (top-k routing + load-balancing aux loss)
- Tied input/output embeddings, GPT-2-style depth-scaled init

Requires **PyTorch ≥ 2.1** (for the memory-efficient / flash SDPA backends).

## Project layout

```
configs/                 # YAML run configs (debug · small · moe · m5pro · shakespeare)
scripts/                 # runnable wrappers: python scripts/train.py ...
src/rootllm/
├── config.py            # typed dataclass configs + YAML / override loading
├── model/               # the transformer, split by concern
│   ├── norm.py          #   RMSNorm
│   ├── rope.py          #   rotary embeddings
│   ├── attention.py     #   grouped-query attention + KV cache
│   ├── feedforward.py   #   SwiGLU + Mixture-of-Experts
│   ├── block.py         #   one pre-norm transformer block
│   └── transformer.py   #   the full model
├── data/                # tokenizer + token datasets (memmap / synthetic)
├── training/            # optimizer, LR schedule, checkpoints, Trainer
├── generation/          # temperature / top-k / top-p sampling
├── utils/               # device & AMP resolution, seeding, logging
└── cli/                 # train · generate · prepare_data entry points
tests/                   # pytest suite
```

## Install

```bash
# Runtime + dev tooling (requires a modern pip / setuptools >= 64)
pip install -e ".[dev]"
```

Without an install you can still run everything from a checkout — `scripts/*.py`
add `src/` to the path, and `pytest` is configured with `pythonpath = ["src"]`.

## Quick start

A full smoke run with **no data files** — the trainer falls back to a
deterministic synthetic token stream:

```bash
python -m rootllm                                   # build + forward + generate sanity check
rootllm-train --config configs/debug.yaml           # ~50 steps, seconds on CPU
```

### Train on real text

A complete run on Shakespeare (downloads the corpus, tokenises with BPE, trains,
and logs generated samples). On an M5 Pro this reaches legible Shakespeare in a
few minutes:

```bash
# 1. download + BPE-tokenise a built-in dataset (or use --input your_files.txt)
rootllm-prepare-data --dataset shakespeare --tokenizer tiktoken --output-dir data/shakespeare

# 2. train (config already points at data/shakespeare and logs a sample each eval)
rootllm-train --config configs/shakespeare.yaml

# 3. sample from the checkpoint
rootllm-generate --ckpt out/shakespeare/ckpt.pt --tokenizer data/shakespeare/tokenizer.json \
  --prompt "ROMEO:" --max-new-tokens 200 --top-k 40 --temperature 0.7
```

Built-in datasets: `shakespeare` (~1 MB), `tinystories` (~22 MB). The default
**byte-level tokenizer** is lossless and dependency-free (vocab 258) so you can
train with zero setup; `--tokenizer tiktoken` uses GPT-2 BPE (vocab 50257).

### Resume a run

Checkpoints store the step, optimiser state, best validation loss, and the
data-sampling RNG, so a long run survives a Ctrl-C or a closed laptop:

```bash
rootllm-train --config configs/shakespeare.yaml --resume out/shakespeare/ckpt.pt \
  --set train.schedule.max_steps=10000          # extend the horizon
```

The model architecture is taken from the checkpoint; `--config`/`--set` control
the (resumable) training settings. Each eval logs **perplexity** and a generated
**sample** so you can watch the model learn.

### Instruction tuning (SFT)

Fine-tune a pretrained base into an instruction-follower. Loss is computed on the
**response only** (prompt tokens are masked), and `--init-from` loads the base
weights with a fresh training state:

```bash
# instructions.jsonl: {"instruction": "...", "response": "..."} per line
rootllm-train --init-from out/shakespeare/ckpt.pt --sft instructions.jsonl \
  --tokenizer data/shakespeare/tokenizer.json --set train.out_dir=out/sft

# chat with it: --chat wraps the prompt in the SFT template and stops at EOS
rootllm-generate --ckpt out/sft/ckpt.pt --tokenizer data/shakespeare/tokenizer.json \
  --chat --prompt "Give a blessing." --repetition-penalty 1.3
```

`--init-from` (load weights, reset optimizer/step/schedule) vs `--resume`
(continue the same run) is the distinction between *fine-tuning* and *continuing*.

### Sampling controls

`rootllm-generate` supports `--temperature`, `--top-k`, `--top-p`, `--min-p`
(scale-adaptive nucleus), and `--repetition-penalty` (>1 discourages loops).

## Configuration

A run is fully described by one `Config` (model + train + data). Configs load
from YAML and accept dotted CLI overrides, which also parse scientific notation:

```bash
rootllm-train --config configs/small.yaml \
  --set train.optimizer.lr=1e-4 train.schedule.max_steps=20000 model.n_experts=8
```

The fully-resolved config and a `metrics.jsonl` log are written to `train.out_dir`
alongside checkpoints (`ckpt.pt`, `best.pt`).

## Testing

```bash
pytest                  # 47 tests, runs on CPU in ~1s
```

Notable coverage: incremental KV-cache decoding matches a full forward pass,
the model overfits a single batch, MoE routing produces gradients and a positive
aux loss, and the prepare→train→generate CLI works end-to-end.

## Device support

Runs on **CUDA**, Apple-Silicon **MPS**, or **CPU** — set `train.device` (default
`auto`). Mixed precision is resolved per device by `dtype: auto`:

- **CUDA** → bf16 (or fp16 + `GradScaler` on older cards)
- **MPS** → bf16 (measured ~1.4× faster than fp32 on M-series, numerically stable)
- **CPU** → fp32

`configs/m5pro.yaml` is tuned for a 64 GB Apple-Silicon machine: a ~180M-param
model that trains at ~0.9 s/step and ~16 GB at batch 8 / seq 512 (bf16 on MPS),
leaving comfortable headroom.

### Gradient checkpointing

MPS has no FlashAttention, so activation memory scales with seq². Set
`model.gradient_checkpointing: true` to recompute each block in the backward pass
instead of storing its activations — mathematically identical, ~30% slower, and a
large memory saving. Measured on an M5 Pro (180M model, batch 8, bf16):

| seq | checkpointing | memory | speed |
|----:|:-------------:|-------:|------:|
| 1024 | off | 40 GB | 2.3 s/step |
| 1024 | **on** | **24 GB** | 3.1 s/step |
| 2048 | on | 35 GB | 9.1 s/step (OOMs without) |

### Memory cap

`train.memory_cap_gb` sets a hard ceiling on device memory: past it the allocator
raises OOM instead of growing (MPS via `set_per_process_memory_fraction`, CUDA via
the equivalent; ignored on CPU). `configs/m5pro.yaml` caps at **32 GB** by default
— change it with `--set train.memory_cap_gb=N`, or `null` to disable. Useful for
leaving headroom for other apps, or catching a config that would otherwise swap.

## License

MIT.
