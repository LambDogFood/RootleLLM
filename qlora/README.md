# QLoRA engine

The smart engine: instead of training a tiny model from scratch (archived at the
`v0.1-from-scratch-poc` tag), adapt a **pretrained code model that already knows
Lua and follows instructions** (`Qwen2.5-Coder-1.5B-Instruct`) with small **LoRA**
adapters on your Luau data.

It reuses everything around it — the same `/generate` HTTP API (so `rootllm-query`
and the confidence/active-learning loop work unchanged), the same `sft/luau.jsonl`,
the same research ingester, the same Docker + GitHub Actions + runner setup. Only
the brain behind `/generate` changes.

A 1.5B model fits a 12 GB GPU in **bf16 — no bitsandbytes, no Blackwell risk**.

## Quickest win: serve the base model (no training)
`Qwen2.5-Coder-1.5B-Instruct` already answers Roblox questions well. Serve it as-is:
```
gh workflow run qlora.yml -f action=serve            # adapter omitted -> base only
```
Then query it exactly like before:
```
curl -s http://<pc-ip>:8000/generate -H "Content-Type: application/json" \
  -d '{"prompt":"How do I make a part that kills the player on touch","max_new_tokens":200}'
```
This alone should be dramatically smarter than the from-scratch model.

## Then sharpen it on Luau with LoRA
```
gh workflow run qlora.yml -f action=finetune \
  -f sft="sft/luau.jsonl data/research/sft.jsonl"     # your curated + researched data
gh workflow run qlora.yml -f action=serve -f adapter=out/qlora-luau -f uncertainty=0.4
```

## Scaling up
- **7B** (better quality): add `bitsandbytes` to `requirements.txt` and pass
  `--four-bit` (QLoRA) — note bitsandbytes on Blackwell is newer.
- The base downloads once and is cached in the `hf_cache` Docker volume.

## Files
- `finetune.py` — LoRA SFT on the instruction data (transformers + peft).
- `serve.py` — serve base + adapter over the same `/generate` API (with confidence + research queue).
- `Dockerfile` / `requirements.txt` — the HuggingFace stack on the CUDA base image.
