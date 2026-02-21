from __future__ import annotations

import math
import os
from dataclasses import asdict
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import wandb

from src.losses import Regularizer
from src.evaluate import compute_interpretability_metrics


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler,
        train_loader: DataLoader,
        val_loader: DataLoader,
        regularizers: list[Regularizer],
        config,           # TrainConfig
        device: torch.device,
        output_dir: str,
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.regularizers = regularizers
        self.config = config
        self.device = device
        self.output_dir = output_dir
        self.global_step = 0

        self.scaler = torch.amp.GradScaler("cuda", enabled=config.use_amp)

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def train(self):
        self.model.train()
        train_iter = iter(self.train_loader)

        while self.global_step < self.config.max_steps:
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(self.train_loader)
                batch = next(train_iter)

            metrics = self._train_step(batch)

            if self.global_step % self.config.log_every == 0:
                metrics["train/lr"] = self.scheduler.get_last_lr()[0]
                metrics["step"] = self.global_step
                wandb.log(metrics, step=self.global_step)

            if self.global_step % self.config.eval_every == 0:
                val_metrics = self.evaluate()
                val_metrics["step"] = self.global_step
                wandb.log(val_metrics, step=self.global_step)
                self.model.train()

            if self.global_step % self.config.save_every == 0 and self.global_step > 0:
                self._save_checkpoint()

            self.global_step += 1

        # Final eval and checkpoint
        val_metrics = self.evaluate()
        wandb.log({**val_metrics, "step": self.global_step})
        self._save_checkpoint(final=True)

    # ------------------------------------------------------------------
    # Single training step
    # ------------------------------------------------------------------

    def _train_step(self, batch: dict) -> dict:
        input_ids = batch["input_ids"].to(self.device)
        targets = batch["targets"].to(self.device)
        needs_activations = len(self.regularizers) > 0

        with torch.amp.autocast("cuda", enabled=self.config.use_amp):
            _, ce_loss, activations = self.model(
                input_ids,
                targets=targets,
                return_activations=needs_activations,
            )

            total_loss = ce_loss
            metrics = {"train/ce_loss": ce_loss.item()}

            for reg in self.regularizers:
                reg_loss = reg.compute(activations, input_ids=input_ids)
                weighted = reg.weight * reg_loss
                total_loss = total_loss + weighted
                metrics[f"train/{reg.name}_loss"] = reg_loss.item()
                metrics[f"train/{reg.name}_weighted"] = weighted.item()

            if self.regularizers:
                reg_total = total_loss.item() - ce_loss.item()
                metrics["train/reg_to_ce_ratio"] = reg_total / (ce_loss.item() + 1e-8)

        metrics["train/total_loss"] = total_loss.item()

        self.scaler.scale(total_loss).backward()
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad(set_to_none=True)
        self.scheduler.step()

        return metrics

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(self) -> dict:
        self.model.eval()
        total_loss = 0.0
        total_tokens = 0
        collected_activations = []
        collected_input_ids = []

        max_batches = self.config.max_eval_batches
        with torch.no_grad():
            for i, batch in enumerate(self.val_loader):
                if max_batches > 0 and i >= max_batches:
                    break
                input_ids = batch["input_ids"].to(self.device)
                targets = batch["targets"].to(self.device)
                collect = i < self.config.num_eval_activation_batches

                _, ce_loss, activations = self.model(
                    input_ids,
                    targets=targets,
                    return_activations=collect,
                )

                n_tokens = (targets != -100).sum().item()
                total_loss += ce_loss.item() * n_tokens
                total_tokens += n_tokens

                if collect:
                    collected_activations.append(
                        {k: {ak: av.cpu() for ak, av in v.items()} for k, v in activations.items()}
                    )
                    # Flatten to 1D now to avoid shape mismatch when batches have
                    # different sequence lengths due to dynamic padding.
                    collected_input_ids.append(input_ids.cpu().reshape(-1))

        avg_loss = total_loss / max(total_tokens, 1)
        perplexity = math.exp(min(avg_loss, 20))  # cap to avoid overflow in logging

        metrics = {
            "val/loss": avg_loss,
            "val/perplexity": perplexity,
        }

        if collected_activations:
            all_ids = torch.cat(collected_input_ids, dim=0)  # [total_tokens]
            interp = compute_interpretability_metrics(collected_activations, all_ids)
            metrics.update(interp)

        return metrics

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def _save_checkpoint(self, final: bool = False):
        tag = "final" if final else f"step_{self.global_step}"
        ckpt_dir = os.path.join(self.output_dir, tag)
        os.makedirs(ckpt_dir, exist_ok=True)
        torch.save(
            {
                "step": self.global_step,
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
            },
            os.path.join(ckpt_dir, "checkpoint.pt"),
        )
        print(f"Saved checkpoint to {ckpt_dir}")
