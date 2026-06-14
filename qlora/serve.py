"""Serve a (LoRA-fine-tuned) pretrained code model over the SAME HTTP API as the
from-scratch server.

Reuses rootllm's HTTP plumbing, confidence metric, and research-queue, so the
existing query client and the active-learning loop work unchanged on the new
engine. Just a different brain behind /generate.

    python qlora/serve.py --base Qwen/Qwen2.5-Coder-1.5B-Instruct --adapter out/qlora-luau \
        --uncertainty-threshold 0.4
"""

from __future__ import annotations

import argparse
import os
import pathlib
import sys
import threading
from typing import Any, Dict, Optional

# Reuse the from-scratch server's HTTP handler + helpers.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))
from rootllm.serve.server import _opt_float, _opt_int, build_server  # noqa: E402
from rootllm.utils.logging import get_logger  # noqa: E402


class HFModelService:
    """Same .complete()/.info contract as rootllm's ModelService, backed by a
    HuggingFace causal LM (optionally with a LoRA adapter)."""

    def __init__(self, base, adapter=None, uncertainty_threshold=0.0,
                 research_queue=None, default_max_new_tokens=256):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.log = get_logger()
        self._torch = torch
        self.tok = AutoTokenizer.from_pretrained(adapter or base)
        self.model = AutoModelForCausalLM.from_pretrained(
            base, torch_dtype=torch.bfloat16, device_map="auto"
        )
        # Load the LoRA adapter only if it exists, so you can serve the base model
        # directly (smart immediately) before any fine-tuning.
        if adapter and os.path.isdir(adapter):
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, adapter)
            self.log.info("loaded LoRA adapter: %s", adapter)
        else:
            adapter = None
            self.log.info("serving base model (no adapter)")
        self.model.eval()

        self.uncertainty_threshold = uncertainty_threshold
        self.research_queue = research_queue
        self.default_max_new_tokens = default_max_new_tokens
        self._lock = threading.Lock()
        self._queue_lock = threading.Lock()
        self.info = {
            "base": base, "adapter": adapter, "device": str(self.model.device),
            "uncertainty_threshold": uncertainty_threshold,
        }

    def complete(self, req: Dict[str, Any]) -> Dict[str, Any]:
        torch = self._torch
        prompt = req.get("prompt", "")
        messages = [{"role": "user", "content": prompt}]
        input_ids = self.tok.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to(self.model.device)
        temperature = float(req.get("temperature", 0.7))

        with self._lock, torch.no_grad():
            out = self.model.generate(
                input_ids,
                max_new_tokens=int(req.get("max_new_tokens", self.default_max_new_tokens)),
                do_sample=temperature > 0,
                temperature=temperature if temperature > 0 else None,
                top_k=_opt_int(req.get("top_k")),
                top_p=_opt_float(req.get("top_p")),
                repetition_penalty=float(req.get("repetition_penalty", 1.1)),
                output_scores=True, return_dict_in_generate=True,
                pad_token_id=self.tok.eos_token_id,
            )

        gen_ids = out.sequences[0][input_ids.shape[1]:]
        completion = self.tok.decode(gen_ids, skip_special_tokens=True)
        if out.scores:
            confs = [torch.softmax(s[0].float(), dim=-1).max().item() for s in out.scores]
            confidence = sum(confs) / len(confs)
        else:
            confidence = 1.0

        result = {
            "prompt": prompt, "completion": completion,
            "text": prompt + completion, "confidence": round(confidence, 4),
        }
        if self.uncertainty_threshold > 0 and confidence < self.uncertainty_threshold and prompt.strip():
            result["queued_for_research"] = self._queue_topic(prompt.strip())
        return result

    def _queue_topic(self, topic: str) -> bool:
        if not self.research_queue:
            return False
        with self._queue_lock:
            existing = set()
            if os.path.exists(self.research_queue):
                with open(self.research_queue) as f:
                    existing = {line.strip() for line in f}
            if topic in existing:
                return False
            os.makedirs(os.path.dirname(os.path.abspath(self.research_queue)), exist_ok=True)
            with open(self.research_queue, "a", encoding="utf-8") as f:
                f.write(topic + "\n")
            self.log.info("queued for research (confidence low): %r", topic)
            return True


def main(argv: Optional[list] = None) -> None:
    ap = argparse.ArgumentParser(description="Serve a LoRA code model over HTTP.")
    ap.add_argument("--base", default="Qwen/Qwen2.5-Coder-1.5B-Instruct")
    ap.add_argument("--adapter", default=None, help="LoRA adapter dir (omit to serve the base)")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--uncertainty-threshold", type=float, default=0.0)
    ap.add_argument("--research-queue", default="data/research/queue.txt")
    args = ap.parse_args(argv)

    service = HFModelService(
        args.base, args.adapter,
        uncertainty_threshold=args.uncertainty_threshold, research_queue=args.research_queue,
    )
    httpd = build_server(service, args.host, args.port)
    get_logger().info("serving %s + %s on http://%s:%d", args.base, args.adapter, args.host, args.port)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        httpd.shutdown()


if __name__ == "__main__":
    main()
