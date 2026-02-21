# Review & Analysis: Experimental Suite

## Experiment 1: Orthogonality Constraints

- **Bug in pseudocode**: The similarity matrix computed is `[batch, seq_len, seq_len]` (token-token similarity), not neuron-neuron overlap. To penalize neuron overlap, reshape activations to `[batch * seq_len, hidden_dim]` and compute a `[hidden_dim, hidden_dim]` cosine similarity matrix. The `torch.eye` dimension should match accordingly.
- The pairwise penalty is O(d^2) — fine at 384 hidden dim, but worth noting for scaling.
- Lambda range `[0.001, 0.01, 0.1]` is reasonable, but 0.1 may dominate the CE loss. Log the ratio of regularization loss to CE loss during training.

## Experiment 2: Sparse Constraints

- **L0 Gate temperature**: `temperature=0.1` is very sharp. Anneal from ~1.0 down to 0.1 during training to avoid premature neuron death.
- **Top-K activation**: "Differentiable approximation" is underspecified. Use straight-through estimator (STE) in the backward pass, as in the TopK SAE literature.
- Dead neuron metric is critical — aggressive sparsity can easily kill 50%+ of neurons.

## Experiment 3: MoE Specialization

- **Weak baselines**: A standard multi-task transformer differs architecturally from MoE, making it hard to attribute improvements to specialization vs. architecture. Add a "Dense + Multi-Head" baseline with the same parameter count but no routing to isolate the effect.
- Mixing classification (SST-2, SNLI) with extraction (SQuAD) complicates analysis. Consider simpler multi-task setups first.

## Experiment 4: Disentanglement

- Most disconnected from the other experiments. Experiments 1/2/5 use TinyStories LM; Experiment 3 uses multi-task classification; Experiment 4 uses a synthetic VAE setup. The Meta-Experiment Pareto comparison will be difficult across different architectures and tasks.
- **Suggestion**: Either (a) also run a VAE variant on TinyStories for comparability, or (b) scope this as a standalone proof of concept outside the Pareto frontier analysis.

## Experiment 5: SAE-Inspired

- **Online SAE needs clarification**: Does the SAE gradient flow back into the model? If yes, the model's representations are actively shaped by the SAE objective (intended). If no, it is just periodic SAE retraining with a reconstruction auxiliary loss. These produce very different outcomes — specify which is intended.
- The "Direct Sparsity" variant is essentially a wide MLP with L1, similar to sparse probing approaches. Worth citing prior work.

## Meta-Experiment

- "Pareto frontier: interpretability vs. performance" requires a single scalar interpretability score, but the suite defines ~6 metrics. Either define an aggregation strategy or report per-metric Pareto frontiers.

## Practical Concerns

1. **Compute estimate is low**: 50-100 A100-hours for ~15+ configs with hyperparameter sweeps, SAE training, and evaluation is optimistic. Budget 150-200 hours including failed runs and debugging.
2. **Missing seeds**: No mention of multiple random seeds. Need at least 3 seeds per config for meaningful comparisons.
3. **No early stopping criteria**: Define when to stop runs that are clearly diverging or producing dead neurons.
4. **Shared infrastructure**: Experiments 1, 2, and 5 share the same base model and task. Build a solid baseline first and branch from there.

## Recommended Priority Order

**Exp 2 > Exp 1 > Exp 5 > Exp 3 > Exp 4**

Experiments 2 (sparsity) and 1 (orthogonality) share the most infrastructure and give quick wins. Experiment 5 builds naturally on top of them. Experiment 4 is the most speculative.
