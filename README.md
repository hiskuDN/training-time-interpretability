# Training-Time Interpretability

Most mechanistic interpretability research works post-hoc: train a model, then analyze it. This project asks a different question — can we build interpretability directly into the training process?

Rather than applying Sparse Autoencoders after the fact, we test whether explicit training-time objectives (orthogonality constraints, sparsity penalties, SAE-inspired losses) produce models that are intrinsically easier to interpret, and whether that comes at a meaningful cost to performance.

The project runs 5 experiments on small-scale GPT-2-style models (~10M parameters) trained on TinyStories, with a meta-experiment comparing approaches on an interpretability/performance Pareto frontier. See [`docs/expt-suite.md`](docs/expt-suite.md) for the full experimental spec.

---

## Setup

**Requirements:** Python 3.12, a [Modal](https://modal.com) account, and a [Weights & Biases](https://wandb.ai) account. Training runs on Modal A100 GPUs; the local environment is only used to launch runs.

```bash
# Clone and create the virtual environment
git clone <repo>
cd training-time-interpretability
python3.12 -m venv .venv
source .venv/bin/activate
pip install modal wandb pyyaml datasets transformers tqdm
```

**Authenticate:**
```bash
modal setup          # authenticate with Modal
wandb login          # authenticate with W&B
```

**Create the W&B secret in Modal:**
```bash
modal secret create wandb-secret WANDB_API_KEY=<your-key>
```

**Cache the dataset (one-time):**
```bash
modal run modal_app.py::cache_dataset
```
This downloads and tokenizes TinyStories into a persistent Modal Volume (`tti-data`), so it never re-downloads between runs.

---

## Experiment 1: Anti-Superposition via Orthogonality Constraints

The core question: does penalizing neuron overlap during training produce more monosemantic representations?

**Three training configurations:**

| Config | Loss |
|---|---|
| Baseline | CE only |
| Orthogonal | CE + λ · cosine similarity between neuron activation vectors |
| Anti-polysemantic | CE + λ · entropy of per-neuron token-type distribution |

For each, λ is swept over `[0.001, 0.01, 0.1]` — 7 configs total, run with 3 seeds each.

**Model:** 6-layer GPT-2 style transformer, 384 hidden dim, 6 heads, ~30M total params (~10.6M non-embedding). Trained on TinyStories for 50k steps.

**Metrics tracked during training:**
- `val/perplexity` — primary performance metric
- `train/reg_to_ce_ratio` — monitors whether the regularizer dominates
- `val/sparsity/layer_*` — fraction of inactive neurons per layer
- `val/polysemanticity/layer_*` — entropy of per-neuron token-type activation distributions

**Running an experiment:**
```bash
# Baseline
modal run --detach modal_app.py --config configs/expt1_baseline.yaml

# Orthogonality penalty, λ=0.01
modal run --detach modal_app.py --config configs/expt1_orthogonal_1e-2.yaml

# Different seed
modal run --detach modal_app.py --config configs/expt1_orthogonal_1e-2.yaml --extra "--train.seed 1"
```

Checkpoints are saved to the `tti-checkpoints` Modal Volume. Results are logged to the `expt1` group in W&B under the `training-time-interpretability` project.

**Implementation notes:**
- The orthogonality penalty computes neuron-neuron cosine similarity: activations are reshaped to `[batch*seq_len, d_model]`, transposed to give one vector per neuron, and a `[d_model, d_model]` similarity matrix is computed. Off-diagonal elements are penalized. See `src/losses.py`.
- The polysemanticity penalty aggregates activation magnitudes by token type via `scatter_add`, normalizes per neuron to a distribution, and penalizes the entropy. See `src/losses.py`.
