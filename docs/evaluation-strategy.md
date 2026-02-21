# Evaluation Strategy — Experiment 1

The goal is to determine whether orthogonality/polysemanticity training objectives produce models that are intrinsically more interpretable, and at what cost to performance.

## 1. Performance (the cost side)

- **Val perplexity** at end of training across all configs
- Plot perplexity vs λ to quantify sensitivity to regularization strength

## 2. Representation Geometry

Direct check that the constraint did what it was supposed to:

- **Neuron-neuron cosine similarity heatmap** — visualize the `[d_model, d_model]` similarity matrix for each model. Off-diagonal values should be closer to zero for orthogonality-trained models.
- **Effective dimensionality** — participation ratio from the eigenspectrum of the MLP activation covariance matrix. More orthogonal models should use more of the available dimensions rather than collapsing into a low-rank subspace.

## 3. Monosemanticity

- **Top-k activating contexts per neuron** — find the token contexts that maximally activate each neuron. Do orthogonality-trained models have neurons with more coherent themes (e.g., consistently activates on verbs) vs. mixed activation patterns?
- **Polysemanticity score by layer** — compare layer-by-layer across models, not just averaged. Already computed during training (`val/polysemanticity/layer_*`).

## 4. Downstream Interpretability Quality

Strongest tests of whether the training objective actually helps in practice:

- **Linear probes** — train simple linear classifiers on frozen layer activations to predict token properties (POS tags, named entities, syntactic role). More disentangled representations should yield higher probe accuracy with less data.
- **SAE quality** — train a small Sparse Autoencoder on MLP activations from each model. Compare reconstruction MSE at fixed L0 sparsity, and number of dead SAE features. Hypothesis: a more orthogonal base model needs fewer SAE features to reconstruct well.

## 5. Pareto Frontier

Plot **perplexity vs polysemanticity score** for all 7 configs on a single scatter plot. This is the meta-result — does orthogonality training move the Pareto frontier or just trade one for the other linearly?

---

## Priority Order

| Priority | Evaluation | Effort |
|---|---|---|
| 1 | Perplexity comparison | Low — already logged in W&B |
| 2 | Cosine similarity heatmaps | Low — one forward pass per model |
| 3 | Top-k activating contexts per neuron | Medium — qualitative, good for writeup |
| 4 | SAE quality comparison | High — strongest quantitative signal |

The SAE comparison is the most interesting result: if orthogonality training makes SAEs work better, that's a concrete, practical finding beyond this project.
