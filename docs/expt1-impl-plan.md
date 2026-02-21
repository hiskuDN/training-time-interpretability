# Implementation Plan: Experiment 1 — Anti-Superposition Training

## Context
No code exists yet. We need to build the training infrastructure for Experiment 1 (orthogonality constraints / anti-polysemanticity) and design it so Experiments 2-5 can reuse it later.

## Project Structure
```
training_time_interpretability/
├── configs/
│   ├── base.yaml                    # Shared defaults
│   ├── expt1_baseline.yaml          # CE-only
│   ├── expt1_orthogonal.yaml        # Orthogonality penalty (3 lambdas)
│   └── expt1_antipolysemantic.yaml  # Polysemanticity penalty (3 lambdas)
├── src/
│   ├── __init__.py
│   ├── model.py          # GPT-2 transformer with activation capture
│   ├── data.py           # TinyStories loading + causal LM collation
│   ├── losses.py         # Regularizers (orthogonality, polysemanticity)
│   ├── trainer.py        # Training loop with pluggable losses + W&B
│   ├── evaluate.py       # Perplexity, sparsity, polysemanticity metrics
│   └── utils.py          # Config dataclasses, YAML loading, seeding
├── train.py              # CLI entry point
└── requirements.txt
```

## Key Design Decisions

### Model (`src/model.py`)
- GPT-2 style: 6 layers, 384 hidden dim, 6 heads, 1536 FFN dim, pre-norm, GELU, no bias
- ~30M total params (19.3M in embeddings), ~10.6M non-embedding params
- Weight tying between token embedding and LM head
- **`return_activations` flag** on forward pass — when True, returns dict of per-layer `mlp_out` and `attn_out` tensors. This is how regularizers access intermediate activations (explicit, no hooks).

### Data (`src/data.py`)
- `skeskinen/TinyStories-hf` from HuggingFace, GPT-2 tokenizer
- Dynamic padding per batch via custom `LMDataCollator`
- Collator creates causal LM targets (input_ids shifted left by 1, padding masked with -100)

### Losses (`src/losses.py`) — the core novel code

**Corrected Orthogonality Penalty:**

> **Bug in original spec**: The pseudocode uses `torch.eye(hidden_dim)` against a `[batch, seq_len, seq_len]` similarity matrix, which computes *token-token* similarity — not neuron-neuron overlap. This penalizes different positions attending to each other, not superposition.

The correct approach computes *neuron-neuron* cosine similarity:
1. Reshape MLP activations `[batch, seq_len, d_model]` -> `[batch*seq_len, d_model]`
2. Transpose to `[d_model, batch*seq_len]` — each row is one neuron's activation pattern across all tokens
3. Cosine similarity matrix `[d_model, d_model]` — neuron-neuron overlap
4. Penalize mean absolute off-diagonal values (`torch.eye(d_model)` to mask diagonal)
5. Subsample to 4096 tokens max for efficiency (O(d^2 * N) cost otherwise)

**Polysemanticity Penalty:**
1. For each neuron, aggregate activation magnitudes by token type via `scatter_add_`
2. Normalize to get per-neuron distribution over token types
3. Compute entropy — high entropy = polysemantic
4. Penalize mean entropy across neurons
5. Needs `input_ids` in addition to activations — regularizer interface uses `**context`

### Trainer (`src/trainer.py`)
- Takes list of `Regularizer` objects; calls each on activations during forward pass
- Logs CE loss, each regularizer loss, and reg-to-CE ratio separately to W&B
- AdamW (lr=3e-4, wd=0.1, betas 0.9/0.95), cosine schedule with 1000-step warmup
- AMP (fp16), gradient clipping at 1.0
- Eval every 1000 steps: perplexity + interpretability metrics on subset of val batches
- Checkpointing every 5000 steps

### Configs
- YAML with `_base_` inheritance from `base.yaml`
- One config file per regularizer type, lambda specified inline
- CLI overrides via `--train.seed 123` etc.
- 7 configs total: 1 baseline + 3 orthogonal (lambda 0.001/0.01/0.1) + 3 polysemantic (lambda 0.001/0.01/0.1)

## Implementation Order
1. `requirements.txt`
2. `src/utils.py` — config dataclasses + loading
3. `src/model.py` — GPT model with activation capture
4. `src/data.py` — TinyStories pipeline
5. `src/losses.py` — orthogonality + polysemanticity regularizers
6. `src/trainer.py` — training loop
7. `src/evaluate.py` — interpretability metrics
8. `train.py` — CLI entry point
9. Config YAML files

## What We're NOT Building Yet
- Linear probe evaluation (post-training analysis script, add later)
- SAE training/evaluation (same)
- Visualization (do in notebooks)
- Distributed training (single GPU sufficient)
- Hyperparameter sweep framework (just separate configs)

## Verification
1. Run `python train.py --config configs/expt1_baseline.yaml` with `max_steps=100` to verify the full pipeline works
2. Check W&B dashboard shows CE loss decreasing
3. Run orthogonality config, verify regularization loss is logged and non-zero
4. Run polysemanticity config, verify entropy-based loss is logged
5. Confirm validation perplexity is computed at eval steps
