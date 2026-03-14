"""
Sparse Autoencoder (SAE) for post-hoc interpretability evaluation.

Architecture:
    x             [B*S, d_model]
    x - pre_bias  subtract learned input bias
    latents = ReLU(encoder(x - pre_bias))   [B*S, d_sae]
    x_hat = decoder(latents) + pre_bias     [B*S, d_model]
    loss  = MSE(x, x_hat) + lambda_l1 * mean(|latents|)

Decoder columns are kept unit-norm after every optimizer step.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class SAEConfig:
    d_model: int = 384
    expansion_factor: int = 8          # d_sae = expansion_factor * d_model
    lambda_l1: float = 1e-3
    learning_rate: float = 1e-3
    batch_size: int = 4096             # tokens per SAE update step
    n_steps: int = 50_000
    warmup_steps: int = 1_000
    log_every: int = 200
    normalize_decoder: bool = True
    dead_threshold: float = 1e-3       # latent fires < this fraction of tokens → dead

    @property
    def d_sae(self) -> int:
        return self.expansion_factor * self.d_model


class SparseAutoencoder(nn.Module):
    def __init__(self, config: SAEConfig):
        super().__init__()
        self.config = config
        d_model, d_sae = config.d_model, config.d_sae

        self.pre_bias = nn.Parameter(torch.zeros(d_model))
        self.encoder = nn.Linear(d_model, d_sae, bias=True)
        # decoder bias is pre_bias (added back after decoding)
        self.decoder = nn.Linear(d_sae, d_model, bias=False)

        # Initialise decoder columns to unit norm
        nn.init.normal_(self.decoder.weight, std=1.0 / math.sqrt(d_model))
        self._normalize_decoder()

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """x: [*, d_model] → latents: [*, d_sae]"""
        return F.relu(self.encoder(x - self.pre_bias))

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """latents: [*, d_sae] → x_hat: [*, d_model]"""
        return self.decoder(latents) + self.pre_bias

    def forward(self, x: torch.Tensor):
        latents = self.encode(x)
        x_hat = self.decode(latents)
        return x_hat, latents

    @torch.no_grad()
    def _normalize_decoder(self):
        """Project decoder columns to unit norm."""
        self.decoder.weight.data = F.normalize(self.decoder.weight.data, dim=0)


class SAETrainer:
    def __init__(self, sae: SparseAutoencoder, config: SAEConfig, device: torch.device):
        self.sae = sae
        self.config = config
        self.device = device
        self.optimizer = torch.optim.Adam(sae.parameters(), lr=config.learning_rate)
        self.scheduler = torch.optim.lr_scheduler.LinearLR(
            self.optimizer,
            start_factor=1e-3,
            end_factor=1.0,
            total_iters=config.warmup_steps,
        )
        self.step_count = 0

    def step(self, acts: torch.Tensor) -> dict:
        """
        One SAE training step.

        Args:
            acts: [N, d_model] token activations (already on device)

        Returns:
            dict with mse_loss, l1_loss, total_loss, l0
        """
        # Subsample if needed
        N = acts.shape[0]
        bs = self.config.batch_size
        if N > bs:
            idx = torch.randperm(N, device=self.device)[:bs]
            acts = acts[idx]

        acts = acts.float()
        x_hat, latents = self.sae(acts)

        mse_loss = F.mse_loss(x_hat, acts)
        l1_loss = latents.abs().mean()
        total_loss = mse_loss + self.config.lambda_l1 * l1_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        if self.step_count < self.config.warmup_steps:
            self.scheduler.step()

        if self.config.normalize_decoder:
            self.sae._normalize_decoder()

        self.step_count += 1

        with torch.no_grad():
            l0 = (latents > 0).float().sum(dim=-1).mean().item()

        return {
            "mse_loss": mse_loss.item(),
            "l1_loss": l1_loss.item(),
            "total_loss": total_loss.item(),
            "l0": l0,
        }


@torch.no_grad()
def compute_sae_metrics(
    sae: SparseAutoencoder,
    model,
    val_loader,
    layer_idx: int,
    device: torch.device,
) -> dict:
    """
    Evaluate a trained SAE on the val set.

    Returns:
        reconstruction_mse, explained_variance, l0_mean, l1_mean,
        dead_latent_fraction
    """
    sae.eval()
    model.eval()

    all_x, all_latents, all_x_hat = [], [], []

    for batch in val_loader:
        input_ids = batch["input_ids"].to(device)
        _, _, activations = model(input_ids, return_activations=True)

        acts = activations[layer_idx]["mlp_out"]           # [B, S, d_model]
        B, S, D = acts.shape
        acts_flat = acts.reshape(B * S, D).float()

        x_hat, latents = sae(acts_flat)
        all_x.append(acts_flat.cpu())
        all_latents.append(latents.cpu())
        all_x_hat.append(x_hat.cpu())

    x = torch.cat(all_x, dim=0)            # [N, d_model]
    latents = torch.cat(all_latents, dim=0) # [N, d_sae]
    x_hat = torch.cat(all_x_hat, dim=0)    # [N, d_model]

    # Reconstruction MSE
    mse = F.mse_loss(x_hat, x).item()

    # Explained variance: 1 - Var(residual) / Var(x)
    residual = x - x_hat
    var_x = x.var().item()
    var_res = residual.var().item()
    explained_var = 1.0 - var_res / (var_x + 1e-10)

    # L0 / L1
    l0_mean = (latents > 0).float().sum(dim=-1).mean().item()
    l1_mean = latents.abs().mean().item()

    # Dead latent fraction
    frac_active = (latents > 0).float().mean(dim=0)  # [d_sae]
    dead_frac = (frac_active < sae.config.dead_threshold).float().mean().item()

    sae.train()
    return {
        "reconstruction_mse": mse,
        "explained_variance": explained_var,
        "l0_mean": l0_mean,
        "l1_mean": l1_mean,
        "dead_latent_fraction": dead_frac,
    }
