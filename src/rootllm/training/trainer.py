"""The training loop.

Single class that ties the pieces together: device/AMP setup, optimiser, the
LR schedule, gradient accumulation, gradient clipping, periodic evaluation,
logging, and checkpointing. Device-agnostic — runs on CUDA, MPS, or CPU.
"""

from __future__ import annotations

import os
import time
from typing import Dict, List, Optional

import torch

from ..config import Config
from ..data.dataset import TokenDataset
from ..utils.device import apply_memory_cap, autocast_context, resolve_amp, resolve_device
from ..utils.logging import JsonlLogger, get_logger
from ..utils.seed import set_seed
from .checkpoint import save_checkpoint
from .optim import configure_optimizer, lr_at_step


class Trainer:
    """Drives optimisation of a model on a token dataset."""

    def __init__(
        self,
        config: Config,
        model: torch.nn.Module,
        train_data: TokenDataset,
        val_data: Optional[TokenDataset] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        start_step: int = 0,
        best_val_loss: float = float("inf"),
        tokenizer: Optional[object] = None,
        batch_rng_state: Optional[torch.Tensor] = None,
    ):
        self.cfg = config
        self.tcfg = config.train
        self.log = get_logger()
        self.tokenizer = tokenizer

        set_seed(self.tcfg.seed)
        self.device = resolve_device(self.tcfg.device)
        self.amp = resolve_amp(self.tcfg.dtype, self.device)

        if self.tcfg.memory_cap_gb:
            fraction = None
            try:
                fraction = apply_memory_cap(self.device, self.tcfg.memory_cap_gb)
            except Exception as e:  # a cap problem must never block a run
                self.log.warning("could not apply memory cap (%s) — continuing without it", e)
            if fraction is not None:
                self.log.info(
                    "memory cap: %.1f GB (%s fraction %.3f)",
                    self.tcfg.memory_cap_gb, self.device.type, fraction,
                )
            elif self.device.type not in ("cuda", "mps"):
                self.log.warning(
                    "memory_cap_gb=%.1f set but cannot be enforced on %s",
                    self.tcfg.memory_cap_gb, self.device.type,
                )

        self.model = model.to(self.device)
        if self.tcfg.compile and hasattr(torch, "compile"):
            self.model = torch.compile(self.model)

        self.optimizer = optimizer or configure_optimizer(
            self.model, self.tcfg.optimizer, self.device.type
        )
        self.scaler = torch.amp.GradScaler(self.amp.device_type, enabled=self.amp.use_grad_scaler)

        self.train_data = train_data
        self.val_data = val_data
        self.start_step = start_step
        self.best_val_loss = best_val_loss

        os.makedirs(self.tcfg.out_dir, exist_ok=True)
        self.metrics = JsonlLogger(os.path.join(self.tcfg.out_dir, "metrics.jsonl"))
        # Reproducible batch sampling, independent of model RNG. On resume, restore
        # the generator so we continue the data stream instead of replaying it.
        self._batch_gen = torch.Generator().manual_seed(self.tcfg.seed + 1)
        if batch_rng_state is not None:
            self._batch_gen.set_state(batch_rng_state.to("cpu", torch.uint8))

    # ------------------------------------------------------------------ #
    def _get_batch(self, data: TokenDataset):
        return data.get_batch(
            self.tcfg.batch_size, self.tcfg.seq_len, self.device, self._batch_gen
        )

    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        """Average loss over ``eval_iters`` batches for each available split."""
        self.model.eval()
        out: Dict[str, float] = {}
        splits = {"train": self.train_data}
        if self.val_data is not None:
            splits["val"] = self.val_data
        for name, data in splits.items():
            total = 0.0
            for _ in range(self.tcfg.eval_iters):
                x, y = self._get_batch(data)
                with autocast_context(self.amp):
                    _, loss = self.model(x, targets=y)
                total += loss.item()
            out[name] = total / self.tcfg.eval_iters
        self.model.train()
        return out

    # ------------------------------------------------------------------ #
    def train(self) -> Dict[str, List]:
        """Run optimisation for ``schedule.max_steps`` steps. Returns history."""
        cfg, tcfg = self.cfg, self.tcfg
        sched, opt_cfg = tcfg.schedule, tcfg.optimizer
        history: Dict[str, List] = {"step": [], "loss": [], "lr": [], "eval": []}

        self.model.train()
        if self.start_step >= sched.max_steps:
            self.log.warning(
                "start_step %d >= schedule.max_steps %d — nothing to train "
                "(raise train.schedule.max_steps)", self.start_step, sched.max_steps,
            )
        self.log.info(
            "training on %s (%s) | steps %d->%d | effective batch %d",
            self.device,
            self.amp.dtype if self.amp.enabled else "float32",
            self.start_step, sched.max_steps,
            tcfg.batch_size * tcfg.grad_accum_steps,
        )

        t0 = time.time()
        for step in range(self.start_step, sched.max_steps):
            lr = lr_at_step(step, opt_cfg.lr, sched)
            for group in self.optimizer.param_groups:
                group["lr"] = lr

            # ---- periodic evaluation + checkpointing ----
            if step % tcfg.eval_interval == 0 and (step > 0 or self.start_step == 0):
                metrics = self.evaluate()
                history["eval"].append((step, metrics))
                ppl = {k: _perplexity(v) for k, v in metrics.items()}
                self.log.info("step %d | eval %s | ppl %s", step, _fmt(metrics), _fmt(ppl))
                self.metrics.log({"step": step, "eval": metrics, "perplexity": ppl})
                self._log_sample()
                self._maybe_checkpoint(step, metrics.get("val", metrics["train"]))

            # ---- one optimiser step over grad_accum micro-batches ----
            loss_accum = 0.0
            for _ in range(tcfg.grad_accum_steps):
                x, y = self._get_batch(self.train_data)
                with autocast_context(self.amp):
                    _, loss = self.model(x, targets=y)
                    loss = loss / tcfg.grad_accum_steps
                self.scaler.scale(loss).backward()
                loss_accum += loss.item()

            if opt_cfg.grad_clip > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), opt_cfg.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad(set_to_none=True)

            if step % tcfg.log_interval == 0:
                dt = (time.time() - t0) / max(1, step - self.start_step + 1)
                self.log.info("step %d | loss %.4f | lr %.2e | %.0f ms/step",
                              step, loss_accum, lr, dt * 1000)
                self.metrics.log({"step": step, "loss": loss_accum, "lr": lr})
            history["step"].append(step)
            history["loss"].append(loss_accum)
            history["lr"].append(lr)

            if tcfg.always_save_checkpoint and step > 0 and step % tcfg.ckpt_interval == 0:
                self._save("ckpt.pt", step, self.best_val_loss)

        # Final checkpoint.
        final_step = sched.max_steps
        self._save("ckpt.pt", final_step, self.best_val_loss)
        self.log.info("done. final loss %.4f", history["loss"][-1] if history["loss"] else float("nan"))
        return history

    # ------------------------------------------------------------------ #
    def _maybe_checkpoint(self, step: int, val_loss: float) -> None:
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self._save("best.pt", step, val_loss)
        if self.tcfg.always_save_checkpoint:
            self._save("ckpt.pt", step, self.best_val_loss)

    def _save(self, name: str, step: int, best_val_loss: float) -> None:
        path = os.path.join(self.tcfg.out_dir, name)
        # Unwrap a possible torch.compile wrapper before saving weights.
        model = getattr(self.model, "_orig_mod", self.model)
        # Persist the data-sampling RNG so a resume continues the stream.
        extra = {"batch_rng_state": self._batch_gen.get_state()}
        save_checkpoint(path, model, self.optimizer, self.cfg, step, best_val_loss, extra=extra)
        self.log.info("saved %s (step %d)", path, step)

    @torch.no_grad()
    def _log_sample(self) -> None:
        """Generate and log a short sample so you can watch the model improve."""
        if self.tokenizer is None or self.tcfg.sample_tokens <= 0:
            return
        try:
            bos = getattr(self.tokenizer, "bos_id", -1)
            seed_id = bos if isinstance(bos, int) and bos >= 0 else 0
            ids = torch.tensor([[seed_id]], dtype=torch.long, device=self.device)
            out = self.model.generate(
                ids, max_new_tokens=self.tcfg.sample_tokens, temperature=0.8, top_k=40
            )
            text = self.tokenizer.decode(out[0].tolist()).replace("\n", " ")
            self.log.info("  sample: %s", text[:160])
        except Exception as e:  # never let sampling crash a training run
            self.log.warning("sample generation failed: %s", e)


def _fmt(metrics: Dict[str, float]) -> str:
    return " ".join(f"{k}={v:.4f}" for k, v in metrics.items())


def _perplexity(loss: float) -> float:
    import math

    return math.exp(min(loss, 20.0))  # clamp to avoid overflow on an untrained model
