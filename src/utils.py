import random
import os
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import yaml


# ---------------------------------------------------------------------------
# Config dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ModelConfig:
    vocab_size: int = 50257
    max_seq_len: int = 512
    n_layers: int = 6
    n_heads: int = 6
    d_model: int = 384
    d_ff: int = 1536
    dropout: float = 0.1
    bias: bool = False
    mlp_topk_ratio: float = 0.0


@dataclass
class DataConfig:
    dataset: str = "skeskinen/TinyStories-hf"
    tokenizer: str = "gpt2"
    max_length: int = 512
    num_workers: int = 4
    data_cache_dir: Optional[str] = None  # set to Modal Volume path when running on Modal


@dataclass
class TrainConfig:
    batch_size: int = 64
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    warmup_steps: int = 1000
    max_steps: int = 50000
    max_grad_norm: float = 1.0
    use_amp: bool = True
    log_every: int = 50
    eval_every: int = 1000
    save_every: int = 5000
    num_eval_activation_batches: int = 10
    max_eval_batches: int = 200  # cap val batches per eval; -1 = full val set
    seed: int = 42
    gradient_accumulation_steps: int = 1


@dataclass
class RegularizerConfig:
    name: str = "none"        # "none" | "orthogonality" | "polysemanticity"
    weight: float = 0.0
    max_tokens: int = 4096    # max tokens to use per layer in orthogonality penalty
    layers: str = "all"       # "all" or comma-separated layer indices e.g. "0,2,4"


@dataclass
class ExperimentConfig:
    name: str = "baseline"
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    regularizers: list = field(default_factory=list)  # list[RegularizerConfig]
    wandb_project: str = "training-time-interpretability"
    wandb_group: str = "expt1"
    output_dir: str = "outputs"


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def _deep_update(base: dict, override: dict) -> dict:
    """Recursively merge override into base."""
    result = base.copy()
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_update(result[k], v)
        else:
            result[k] = v
    return result


def _dict_to_config(d: dict) -> ExperimentConfig:
    model = ModelConfig(**{k: v for k, v in d.get("model", {}).items()})
    data = DataConfig(**{k: v for k, v in d.get("data", {}).items()})
    train = TrainConfig(**{k: v for k, v in d.get("train", {}).items()})
    regularizers = [
        RegularizerConfig(**{k: v for k, v in r.items()})
        for r in d.get("regularizers", [])
    ]
    top_level = {
        k: v for k, v in d.items()
        if k not in ("model", "data", "train", "regularizers", "_base_")
    }
    return ExperimentConfig(
        model=model,
        data=data,
        train=train,
        regularizers=regularizers,
        **top_level,
    )


def load_config(path: str) -> ExperimentConfig:
    """Load a YAML config, recursively merging any _base_ parent."""
    with open(path) as f:
        raw = yaml.safe_load(f)

    if "_base_" in raw:
        base_path = raw["_base_"]
        # Resolve relative to the config file's directory
        base_path = os.path.join(os.path.dirname(path), os.path.basename(base_path))
        with open(base_path) as f:
            base_raw = yaml.safe_load(f)
        raw = _deep_update(base_raw, {k: v for k, v in raw.items() if k != "_base_"})

    return _dict_to_config(raw)


def apply_cli_overrides(config: ExperimentConfig, overrides: list[str]) -> None:
    """
    Apply dot-notation CLI overrides to a config in-place.

    Example: ["--train.seed", "123", "--train.learning_rate", "1e-4"]
    Supports: train.*, model.*, data.*, top-level fields.
    Does not support regularizers override (use separate config files).
    """
    i = 0
    while i < len(overrides):
        arg = overrides[i]
        if not arg.startswith("--"):
            i += 1
            continue
        key = arg.lstrip("-")
        value = overrides[i + 1] if i + 1 < len(overrides) else None
        i += 2

        parts = key.split(".", 1)
        if len(parts) == 2:
            section, attr = parts
            target = getattr(config, section, None)
            if target is not None and hasattr(target, attr):
                orig = getattr(target, attr)
                setattr(target, attr, _cast(value, type(orig)))
        else:
            if hasattr(config, key):
                orig = getattr(config, key)
                setattr(config, key, _cast(value, type(orig)))


def _cast(value: str, typ: type):
    if typ == bool:
        return value.lower() in ("true", "1", "yes")
    try:
        return typ(value)
    except (ValueError, TypeError):
        return value


# ---------------------------------------------------------------------------
# Seeding
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
