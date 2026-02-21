"""
Entry point for Experiments 1, 2, and 5 (shared LM architecture, different losses).

Usage:
    python train.py --config configs/expt1_baseline.yaml
    python train.py --config configs/expt1_orthogonal_1e-2.yaml --train.seed 1
"""

import argparse
import os
from dataclasses import asdict

import torch
import wandb
from torch.utils.data import DataLoader

from src.utils import load_config, apply_cli_overrides, set_seed
from src.model import GPT, GPTConfig
from src.data import get_tokenizer, get_tinystories_dataset, make_dataloaders
from src.losses import build_regularizers
from src.trainer import Trainer


def get_cosine_schedule_with_warmup(optimizer, warmup_steps: int, total_steps: int):
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return max(0.0, 0.5 * (1.0 + torch.cos(torch.tensor(3.14159265 * progress)).item()))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args, unknown = parser.parse_known_args()

    config = load_config(args.config)
    apply_cli_overrides(config, unknown)

    set_seed(config.train.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # W&B
    wandb.init(
        project=config.wandb_project,
        group=config.wandb_group,
        name=f"{config.name}_seed{config.train.seed}",
        config=asdict(config),
    )

    # Data
    tokenizer = get_tokenizer(config.data.tokenizer)
    train_dataset, val_dataset = get_tinystories_dataset(
        tokenizer,
        max_length=config.data.max_length,
        num_workers=config.data.num_workers,
        cache_dir=config.data.data_cache_dir,
    )
    train_loader, val_loader = make_dataloaders(
        train_dataset, val_dataset, tokenizer,
        batch_size=config.train.batch_size,
        num_workers=config.data.num_workers,
    )

    # Model
    model_cfg = GPTConfig(
        vocab_size=config.model.vocab_size,
        max_seq_len=config.model.max_seq_len,
        n_layers=config.model.n_layers,
        n_heads=config.model.n_heads,
        d_model=config.model.d_model,
        d_ff=config.model.d_ff,
        dropout=config.model.dropout,
        bias=config.model.bias,
    )
    model = GPT(model_cfg).to(device)
    print(f"Parameters: {model.num_params():,} total, {model.num_params(non_embedding=True):,} non-embedding")
    wandb.config.update({"model/total_params": model.num_params(),
                          "model/non_embedding_params": model.num_params(non_embedding=True)})

    # Optimizer + scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.train.learning_rate,
        weight_decay=config.train.weight_decay,
        betas=(0.9, 0.95),
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        warmup_steps=config.train.warmup_steps,
        total_steps=config.train.max_steps,
    )

    # Regularizers
    regularizers = build_regularizers(config.regularizers)
    if regularizers:
        print(f"Regularizers: {[r.name for r in regularizers]}")

    # Output dir (includes run name + seed for disambiguation)
    output_dir = os.path.join(
        config.output_dir, f"{config.name}_seed{config.train.seed}"
    )
    os.makedirs(output_dir, exist_ok=True)

    # Train
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=train_loader,
        val_loader=val_loader,
        regularizers=regularizers,
        config=config.train,
        device=device,
        output_dir=output_dir,
    )
    trainer.train()

    wandb.finish()


if __name__ == "__main__":
    main()
