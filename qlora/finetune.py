"""LoRA fine-tune a pretrained code model on Luau instruction data.

The new engine. Instead of training from scratch, adapt Qwen2.5-Coder (which
already knows Lua and follows instructions) with small LoRA adapters on your Luau
SFT data. A 1.5B model fits a 12 GB GPU in bf16 with no bitsandbytes (so no
Blackwell-driver risk); pass --four-bit for 7B+ via QLoRA.

    python qlora/finetune.py --sft sft/luau.jsonl data/research/sft.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
from typing import List


def load_examples(paths: List[str]) -> List[dict]:
    rows = []
    for path in paths:
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                instruction = d.get("instruction") or d.get("prompt") or ""
                response = d.get("output") or d.get("response") or ""
                if instruction and response:
                    rows.append({"instruction": instruction, "response": response})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="LoRA fine-tune a code model on Luau.")
    ap.add_argument("--base", default="Qwen/Qwen2.5-Coder-1.5B-Instruct")
    ap.add_argument("--sft", nargs="+", default=["sft/luau.jsonl"], help="instruction JSONL files")
    ap.add_argument("--out", default="out/qlora-luau", help="adapter output dir")
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--four-bit", action="store_true", help="4-bit QLoRA (needs bitsandbytes; for 7B+)")
    args = ap.parse_args()

    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

    tok = AutoTokenizer.from_pretrained(args.base)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model_kwargs = {"torch_dtype": torch.bfloat16}
    if args.four_bit:
        from transformers import BitsAndBytesConfig

        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
        )
        model_kwargs["device_map"] = "auto"
    model = AutoModelForCausalLM.from_pretrained(args.base, **model_kwargs)
    if args.four_bit:
        from peft import prepare_model_for_kbit_training

        model = prepare_model_for_kbit_training(model)  # QLoRA prep (also enables grad reqs)
    else:
        model.enable_input_require_grads()  # required for grad checkpointing on a frozen base

    model = get_peft_model(
        model,
        LoraConfig(
            r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
        ),
    )
    model.print_trainable_parameters()

    rows = load_examples(args.sft)
    if not rows:
        raise SystemExit(f"no instruction examples found in {args.sft}")
    print(f"{len(rows)} instruction examples from {args.sft}")

    def encode(row):
        msgs = [{"role": "user", "content": row["instruction"]}]
        prompt_ids = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True)
        full_ids = tok.apply_chat_template(
            msgs + [{"role": "assistant", "content": row["response"]}], tokenize=True
        )[: args.max_len]
        labels = list(full_ids)
        for i in range(min(len(prompt_ids), len(labels))):  # mask the prompt; learn the response
            labels[i] = -100
        return {"input_ids": full_ids, "labels": labels, "attention_mask": [1] * len(full_ids)}

    ds = Dataset.from_list(rows).map(encode, remove_columns=["instruction", "response"])

    def collate(batch):
        maxlen = max(len(b["input_ids"]) for b in batch)
        pad = tok.pad_token_id
        out = {"input_ids": [], "labels": [], "attention_mask": []}
        for b in batch:
            n = maxlen - len(b["input_ids"])
            out["input_ids"].append(b["input_ids"] + [pad] * n)
            out["labels"].append(b["labels"] + [-100] * n)
            out["attention_mask"].append(b["attention_mask"] + [0] * n)
        return {k: torch.tensor(v) for k, v in out.items()}

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=args.out, num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch, gradient_accumulation_steps=args.grad_accum,
            learning_rate=args.lr, bf16=True, logging_steps=10, save_strategy="epoch",
            warmup_ratio=0.03, lr_scheduler_type="cosine", gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False}, report_to=[],
        ),
        train_dataset=ds,
        data_collator=collate,
    )
    trainer.train()
    model.save_pretrained(args.out)
    tok.save_pretrained(args.out)
    print(f"saved LoRA adapter to {args.out}")


if __name__ == "__main__":
    main()
