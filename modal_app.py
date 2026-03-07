"""
Modal app for running training experiments on cloud GPUs.

Setup (one-time):
    modal secret create wandb-secret WANDB_API_KEY=<your-key>
    modal run modal_app.py::cache_dataset

Running an experiment:
    modal run modal_app.py --config configs/expt1_baseline.yaml
    modal run modal_app.py --config configs/expt1_orthogonal_1e-2.yaml --extra "--train.seed 1"

Running evaluation:
    modal run --detach modal_app.py::eval_main --run-name expt1_baseline_seed42
    modal run --detach modal_app.py::eval_main --run-name expt1_orthogonal_1e-2_seed42
"""

import modal

# ---------------------------------------------------------------------------
# Volumes
# ---------------------------------------------------------------------------

data_volume = modal.Volume.from_name("tti-data", create_if_missing=True)
checkpoint_volume = modal.Volume.from_name("tti-checkpoints", create_if_missing=True)

DATA_DIR = "/data"
CHECKPOINT_DIR = "/checkpoints"

# ---------------------------------------------------------------------------
# Images
# ---------------------------------------------------------------------------

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install_from_requirements("requirements.txt")
    .add_local_dir("src", "/root/project/src")
    .add_local_dir("configs", "/root/project/configs")
    .add_local_file("train.py", "/root/project/train.py")
)

eval_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install_from_requirements("requirements.txt")
    .pip_install("scikit-learn>=1.3", "spacy>=3.7")
    .run_commands("python -m spacy download en_core_web_sm")
    .add_local_dir("src", "/root/project/src")
    .add_local_dir("configs", "/root/project/configs")
)

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = modal.App("training-time-interpretability", image=image)


@app.function(
    gpu="A100",
    timeout=60 * 60 * 12,  # 12 hours
    volumes={
        DATA_DIR: data_volume,
        CHECKPOINT_DIR: checkpoint_volume,
    },
    secrets=[modal.Secret.from_name("wandb-secret")],
)
def train(config: str, extra: str = ""):
    """
    Run a training experiment.

    Args:
        config: path to YAML config relative to project root
        extra:  additional CLI args as a string, e.g. "--train.seed 1"
    """
    import subprocess
    import sys
    import os

    os.chdir("/root/project")

    # Inject Modal Volume paths into config via CLI overrides
    cmd = [
        sys.executable, "train.py",
        "--config", config,
        "--data.data_cache_dir", DATA_DIR,
        "--output_dir", CHECKPOINT_DIR,
    ]
    if extra:
        cmd.extend(extra.split())

    result = subprocess.run(cmd, check=True)
    checkpoint_volume.commit()
    return result.returncode


@app.function(
    gpu="A100",
    timeout=60 * 60 * 4,  # 4 hours
    image=eval_image,
    volumes={
        DATA_DIR: data_volume,
        CHECKPOINT_DIR: checkpoint_volume,
    },
    secrets=[modal.Secret.from_name("wandb-secret")],
)
def evaluate(
    run_name: str,
    checkpoint_path: str = "",
    n_probe_texts: int = 5000,
    topk_k: int = 20,
    mlp_topk_ratio: float = 0.0,
    wandb_group: str = "expt1_eval",
):
    """
    Run the full evaluation pipeline for a trained model.

    Saves to /checkpoints/eval/{run_name}/:
        perplexity.json, cosine_sim_heatmaps.npz,
        topk_contexts.json, probe_results.json

    Args:
        run_name:         name of the run (used for output path and W&B)
        checkpoint_path:  optional override; default is
                          /checkpoints/{run_name}/final/checkpoint.pt
        n_probe_texts:    number of raw val texts for POS probing
        topk_k:           top-k contexts to collect per neuron
        mlp_topk_ratio:   top-k ratio for Top-K MLP models (0.0 = disabled)
        wandb_group:      W&B group name for this evaluation run
    """
    import json
    import os
    import sys

    import numpy as np
    import spacy
    import torch
    import torch.nn.functional as F
    import wandb
    from datasets import load_dataset
    from torch.utils.data import DataLoader

    os.chdir("/root/project")
    sys.path.insert(0, "/root/project")

    from src.data import get_tokenizer, get_tinystories_dataset
    from src.model import GPT, GPTConfig
    from src.eval_probing import (
        TopKContextCollector,
        collect_probe_dataset,
        train_linear_probes,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # -----------------------------------------------------------------------
    # Output directory
    # -----------------------------------------------------------------------
    out_dir = os.path.join(CHECKPOINT_DIR, "eval", run_name)
    os.makedirs(out_dir, exist_ok=True)

    # -----------------------------------------------------------------------
    # Load model
    # -----------------------------------------------------------------------
    if not checkpoint_path:
        checkpoint_path = os.path.join(CHECKPOINT_DIR, run_name, "final", "checkpoint.pt")

    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)

    model_cfg = GPTConfig(
        vocab_size=50257,
        max_seq_len=512,
        n_layers=6,
        n_heads=6,
        d_model=384,
        d_ff=1536,
        dropout=0.0,
        bias=False,
        mlp_topk_ratio=mlp_topk_ratio,
    )
    model = GPT(model_cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"Loaded model ({model.num_params():,} params) from step {ckpt.get('step', '?')}")

    # -----------------------------------------------------------------------
    # Data
    # -----------------------------------------------------------------------
    tokenizer = get_tokenizer("gpt2")
    _, val_dataset = get_tinystories_dataset(
        tokenizer,
        max_length=512,
        num_workers=4,
        cache_dir=DATA_DIR,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        collate_fn=_make_lm_collator(tokenizer.pad_token_id),
        num_workers=4,
        pin_memory=True,
    )

    # -----------------------------------------------------------------------
    # W&B
    # -----------------------------------------------------------------------
    wandb.init(
        project="training-time-interpretability",
        group=wandb_group,
        name=run_name,
    )

    # -----------------------------------------------------------------------
    # Pass 1: perplexity + cosine sim heatmaps + top-k contexts simultaneously
    # -----------------------------------------------------------------------
    print("Pass 1: perplexity + heatmaps + top-k contexts...")
    model.eval()

    topk_collector = TopKContextCollector(k=topk_k, n_layers=model_cfg.n_layers, d_model=model_cfg.d_model)

    # Perplexity accumulators
    total_loss = 0.0
    total_tokens = 0
    n_batches = 0

    # Cosine sim: reservoir per layer
    max_tokens_heatmap = 32768
    reservoirs: dict[int, list] = {i: [] for i in range(model_cfg.n_layers)}
    res_counts: dict[int, int] = {i: 0 for i in range(model_cfg.n_layers)}

    # Sparsity accumulators (mlp_hidden preferred; fallback to mlp_out)
    sparsity_threshold = 0.01
    sparsity_sums: dict[int, float] = {i: 0.0 for i in range(model_cfg.n_layers)}
    sparsity_batch_counts: dict[int, int] = {i: 0 for i in range(model_cfg.n_layers)}

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            targets = batch["targets"].to(device)

            _, ce_loss, activations = model(
                input_ids, targets=targets, return_activations=True
            )

            n_tokens = (targets != -100).sum().item()
            total_loss += ce_loss.item() * n_tokens
            total_tokens += n_tokens
            n_batches += 1

            # Top-k update
            topk_collector.update(activations, input_ids, tokenizer)

            # Sparsity
            for layer_idx, acts_dict in activations.items():
                hidden = acts_dict.get("mlp_hidden") or acts_dict["mlp_out"]
                sparsity_sums[layer_idx] += (hidden.abs() < sparsity_threshold).float().mean().item()
                sparsity_batch_counts[layer_idx] += 1

            # Reservoir sampling for heatmaps
            for layer_idx, acts_dict in activations.items():
                acts = acts_dict["mlp_out"]
                B, S, D = acts.shape
                rows = acts.reshape(B * S, D).cpu().float()
                N = rows.shape[0]
                reservoir = reservoirs[layer_idx]
                k = res_counts[layer_idx]
                for j in range(N):
                    k += 1
                    if len(reservoir) < max_tokens_heatmap:
                        reservoir.append(rows[j])
                    else:
                        r = int(torch.randint(0, k, (1,)).item())
                        if r < max_tokens_heatmap:
                            reservoir[r] = rows[j]
                res_counts[layer_idx] = k

    # Compute perplexity
    val_loss = total_loss / max(total_tokens, 1)
    perplexity = float(torch.exp(torch.tensor(val_loss)).item())
    perplexity_result = {
        "val_loss": val_loss,
        "perplexity": perplexity,
        "n_tokens": total_tokens,
        "n_batches": n_batches,
    }
    print(f"  val_loss={val_loss:.4f}  perplexity={perplexity:.2f}")

    # Save perplexity
    with open(os.path.join(out_dir, "perplexity.json"), "w") as f:
        json.dump(perplexity_result, f, indent=2)

    # Compute and save sparsity
    sparsity_result = {
        f"layer_{i}": sparsity_sums[i] / max(sparsity_batch_counts[i], 1)
        for i in range(model_cfg.n_layers)
    }
    sparsity_result["mean"] = sum(sparsity_result.values()) / model_cfg.n_layers
    with open(os.path.join(out_dir, "sparsity.json"), "w") as f:
        json.dump(sparsity_result, f, indent=2)
    print(f"  Saved sparsity.json  mean={sparsity_result['mean']:.4f}")

    # Compute heatmaps
    sim_matrices = {}
    mean_off_diag = {}
    for layer_idx in range(model_cfg.n_layers):
        reservoir = reservoirs[layer_idx]
        if not reservoir:
            continue
        mat = torch.stack(reservoir, dim=0)
        D = mat.shape[1]
        neuron_vecs = F.normalize(mat.T, dim=1)
        sim = torch.mm(neuron_vecs, neuron_vecs.T).numpy()
        sim_matrices[layer_idx] = sim
        eye = np.eye(D)
        mean_off_diag[layer_idx] = float(np.abs(sim[~eye.astype(bool)]).mean())

    # Save heatmaps as npz
    np.savez(
        os.path.join(out_dir, "cosine_sim_heatmaps.npz"),
        **{f"layer_{k}": v for k, v in sim_matrices.items()},
    )
    print(f"  Saved cosine_sim_heatmaps.npz  mean_off_diag={mean_off_diag}")

    # Save top-k contexts
    topk_data = topk_collector.finalize()
    with open(os.path.join(out_dir, "topk_contexts.json"), "w") as f:
        json.dump(topk_data, f)
    print("  Saved topk_contexts.json")

    # -----------------------------------------------------------------------
    # Pass 2: collect probe dataset → train linear probes
    # -----------------------------------------------------------------------
    print(f"Pass 2: probing with {n_probe_texts} texts...")
    nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])

    # Re-load raw texts from HuggingFace (avoids modifying cache_dataset)
    raw_dataset = load_dataset("skeskinen/TinyStories-hf", split="validation")
    val_texts = [ex["text"] for ex in raw_dataset.select(range(min(n_probe_texts, len(raw_dataset))))]

    probe_dataset = collect_probe_dataset(
        model=model,
        texts=val_texts,
        tokenizer=tokenizer,
        nlp=nlp,
        device=device,
        batch_size=32,
    )

    probe_results = train_linear_probes(probe_dataset)

    # Make confusion matrices JSON-serializable
    probe_results_json = {}
    for layer_idx, r in probe_results.items():
        probe_results_json[str(layer_idx)] = {
            "accuracy": r["accuracy"],
            "macro_f1": r["macro_f1"],
            "per_class_f1": r["per_class_f1"],
            "confusion_matrix": r["confusion_matrix"],
        }
    probe_results_json["label_names"] = probe_dataset["label_names"]

    with open(os.path.join(out_dir, "probe_results.json"), "w") as f:
        json.dump(probe_results_json, f, indent=2)
    print("  Saved probe_results.json")

    # -----------------------------------------------------------------------
    # Log summary to W&B
    # -----------------------------------------------------------------------
    wandb_metrics = {
        "eval/val_loss": val_loss,
        "eval/perplexity": perplexity,
    }
    for layer_idx, v in mean_off_diag.items():
        wandb_metrics[f"eval/mean_off_diag/layer_{layer_idx}"] = v
    for layer_idx, r in probe_results.items():
        wandb_metrics[f"eval/probe_accuracy/layer_{layer_idx}"] = r["accuracy"]
        wandb_metrics[f"eval/probe_macro_f1/layer_{layer_idx}"] = r["macro_f1"]
    for i in range(model_cfg.n_layers):
        wandb_metrics[f"eval/sparsity/layer_{i}"] = sparsity_result[f"layer_{i}"]
    wandb_metrics["eval/sparsity/mean"] = sparsity_result["mean"]

    wandb.log(wandb_metrics)
    wandb.finish()

    checkpoint_volume.commit()
    print(f"Done. Results saved to {out_dir}")


def _make_lm_collator(pad_token_id: int):
    """Return an LMDataCollator instance (avoids importing at module level)."""
    from src.data import LMDataCollator
    return LMDataCollator(pad_token_id=pad_token_id)


@app.function(
    volumes={DATA_DIR: data_volume},
    timeout=60 * 60 * 2,
)
def cache_dataset():
    """
    One-time function to download and tokenize TinyStories into the data volume.
    Run with: modal run modal_app.py::cache_dataset
    """
    import os
    import sys

    os.chdir("/root/project")
    sys.path.insert(0, "/root/project")

    from src.data import get_tokenizer, get_tinystories_dataset

    tokenizer = get_tokenizer("gpt2")
    get_tinystories_dataset(
        tokenizer,
        max_length=512,
        num_workers=4,
        cache_dir=DATA_DIR,
    )
    data_volume.commit()
    print("Dataset cached successfully.")


# ---------------------------------------------------------------------------
# Local entrypoints
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def main(config: str, extra: str = ""):
    train.remote(config=config, extra=extra)


@app.local_entrypoint()
def eval_main(
    run_name: str,
    checkpoint_path: str = "",
    n_probe_texts: int = 5000,
    topk_k: int = 20,
    mlp_topk_ratio: float = 0.0,
    wandb_group: str = "expt1_eval",
):
    """
    Launch evaluation for a trained run.

    Usage:
        modal run --detach modal_app.py::eval_main --run-name expt1_baseline_seed42
        modal run --detach modal_app.py::eval_main --run-name expt2_topk_25pct_seed42 --mlp-topk-ratio 0.25 --wandb-group expt2_eval
    """
    evaluate.remote(
        run_name=run_name,
        checkpoint_path=checkpoint_path,
        n_probe_texts=n_probe_texts,
        topk_k=topk_k,
        mlp_topk_ratio=mlp_topk_ratio,
        wandb_group=wandb_group,
    )
