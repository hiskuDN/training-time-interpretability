"""
Modal app for running training experiments on cloud GPUs.

Setup (one-time):
    modal secret create wandb-secret WANDB_API_KEY=<your-key>
    modal run modal_app.py::cache_dataset

Running an experiment:
    modal run modal_app.py --config configs/expt1_baseline.yaml
    modal run modal_app.py --config configs/expt1_orthogonal_1e-2.yaml --extra "--train.seed 1"
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
# Image
# ---------------------------------------------------------------------------

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install_from_requirements("requirements.txt")
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
    # Mount the local source code into the container
    mounts=[modal.Mount.from_local_dir(".", remote_path="/root/project")],
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
    volumes={DATA_DIR: data_volume},
    mounts=[modal.Mount.from_local_dir(".", remote_path="/root/project")],
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
# Local entrypoint
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def main(config: str, extra: str = ""):
    train.remote(config=config, extra=extra)
