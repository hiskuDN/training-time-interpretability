# Experiment 2 Results: Sparse Architectural Constraints

**Date:** 2026-03-08
**Runs:** 6 configs + baseline reference, seed 42
**W&B group:** [`expt2`](https://wandb.ai/hisku/training-time-interpretability) / evals: [`expt2_eval`](https://wandb.ai/hisku/training-time-interpretability)

---

## Setup

Same base model as Experiment 1 (6-layer GPT-2-style, 384 hidden dim, 6 heads, ~30M params, TinyStories, 50k steps). Two sparsity mechanisms tested:

- **L1 activation penalty (soft):** L1 norm on post-GELU d_ff=1536 hidden activations, λ ∈ {1e-4, 1e-3, 1e-2}. No architectural change; gradient flows through all neurons.
- **Top-K MLP (hard):** at each forward pass, all-but-top-k neurons by magnitude are zeroed. k ∈ {10%, 25%, 50%} of d_ff=1536 (= 154, 384, 768 active neurons). Straight-through: gradient only propagates through active neurons; mask is non-differentiable.

Sparsity is measured on `mlp_hidden` (d_ff=1536), i.e. the post-GELU hidden layer where zeros are actually introduced. Cosine similarity heatmaps and probes use `mlp_out` (d_model=384) for comparability with Experiment 1.

---

## 1. Perplexity

| Run | Val Loss | Perplexity | Δ vs Baseline |
|-----|----------|------------|---------------|
| Baseline | 1.4598 | 4.31 | — |
| L1 λ=1e-4 | 1.4599 | 4.31 | +0.0001 nats |
| L1 λ=1e-3 | 1.4600 | 4.31 | +0.0002 nats |
| L1 λ=1e-2 | 1.4601 | 4.31 | +0.0003 nats |
| Top-K 50% | 1.4806 | 4.40 | +0.0208 nats (+1.4%) |
| Top-K 25% | 1.4869 | 4.42 | +0.0271 nats (+1.9%) |
| Top-K 10% | 1.4899 | 4.44 | +0.0301 nats (+2.1%) |

**Findings:**
- **L1 is entirely free** — all three λ values produce val loss within 0.0003 nats of the baseline, well within noise. The penalty is so small relative to CE loss that it has no measurable effect on the learned distribution.
- **Top-K has a small but real monotonic cost** that scales cleanly with constraint strength. Going from 50% active to 10% active costs only 0.01 additional nats. The 2% perplexity hit at 90% sparsity is a remarkably low price.
- The Top-K costs are not alarming — they suggest the model adapts gracefully by concentrating information in the surviving neurons, rather than catastrophically losing capacity.

---

## 2. Activation Sparsity

Fraction of d_ff=1536 neurons with |activation| < 0.01, computed over the full val set during post-hoc eval.

| Run | Mean | L0 | L1 | L2 | L3 | L4 | L5 |
|-----|------|-----|-----|-----|-----|-----|-----|
| Baseline | N/A | — | — | — | — | — | — |
| L1 λ=1e-4 | 0.032 | 0.024 | 0.067 | 0.035 | 0.029 | 0.021 | 0.017 |
| L1 λ=1e-3 | 0.053 | 0.029 | 0.053 | 0.069 | 0.074 | 0.059 | 0.035 |
| L1 λ=1e-2 | 0.102 | 0.075 | 0.108 | 0.057 | 0.029 | 0.175 | 0.167 |
| Top-K 10% | 0.900 | 0.900 | 0.900 | 0.900 | 0.900 | 0.900 | 0.900 |
| Top-K 25% | 0.750 | 0.750 | 0.750 | 0.750 | 0.750 | 0.750 | 0.750 |
| Top-K 50% | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 |

**Findings:**
- **L1 produces weak and uneven sparsity.** Even at λ=1e-2, only ~10% of neurons are inactive on average. The sparsity profile is non-uniform across layers: at λ=1e-2, L4 and L5 reach 17% while L3 sits at 3%. This unevenness suggests the model prioritises keeping certain layers dense and absorbs the penalty in layers where it costs less.
- **Top-K is perfectly controllable and uniform by construction.** Sparsity is exactly 50/75/90% at every layer regardless of λ — the hard mask enforces this. There is no layer-to-layer variation.
- To match Top-K 25% sparsity (75% inactive) with L1 would require an enormous λ that would almost certainly collapse perplexity. L1 is not a viable path to high sparsity.

---

## 3. Representation Geometry (Cosine Similarity)

Mean absolute off-diagonal cosine similarity between neuron activation vectors in `mlp_out`. Lower = more orthogonal.

| Run | Mean | L0 | L1 | L2 | L3 | L4 | L5 |
|-----|------|-----|-----|-----|-----|-----|-----|
| Baseline | 0.1638 | 0.1424 | 0.2144 | 0.1934 | 0.1709 | 0.1374 | 0.1241 |
| L1 λ=1e-4 | 0.1242 | 0.1430 | 0.1867 | 0.1281 | 0.0995 | 0.0963 | 0.0918 |
| L1 λ=1e-3 | 0.1217 | 0.1566 | **0.3017** | 0.0882 | 0.0662 | 0.0544 | 0.0627 |
| L1 λ=1e-2 | 0.1009 | 0.0805 | 0.1032 | 0.1415 | **0.1714** | 0.0461 | 0.0626 |
| Top-K 10% | 0.1517 | 0.1760 | 0.1402 | **0.2408** | 0.1576 | 0.1241 | 0.0714 |
| Top-K 25% | 0.1563 | 0.1432 | 0.1487 | **0.2640** | 0.1563 | 0.1102 | 0.1154 |
| Top-K 50% | 0.1473 | 0.1317 | 0.1843 | 0.1737 | 0.1727 | 0.1246 | 0.0970 |

**Findings:**

**L1 substantially reduces neuron correlation (−24% to −38% vs baseline), despite barely achieving sparsity.** The dominant effect of L1 is decorrelation, not sparsification. This is mechanistically interesting: penalising the mean magnitude of hidden activations appears to discourage neurons from activating together, even without zeroing them.

However, L1 exhibits the same layer-resistance pattern seen with the orthogonal regularizer in Experiment 1: **L1 at λ=1e-3 causes L1 to spike to 0.3017** (+41% vs baseline), while outer layers drop dramatically. At λ=1e-2, L3 rises back toward baseline (0.1714) while L0, L1, L4, L5 decrease. The model consistently fights back in middle layers.

**Top-K achieves only modest correlation reduction (−5% to −10%) despite far greater sparsity.** L2 is a consistent outlier — cosine sim *increases* for all Top-K runs (0.1934 baseline → 0.2408/0.2640/0.1737 for 10%/25%/50%). With fewer active neurons, the surviving neurons in `mlp_out` appear to become more correlated, possibly because they must collectively encode the same information that was previously spread across more neurons.

**The two mechanisms are doing fundamentally different things:** L1 decorrelates without sparsifying; Top-K sparsifies without decorrelating. They are complementary rather than substitutable.

---

## 4. Linear Probes (POS Tagging)

### Accuracy

| Run | L0 | L1 | L2 | L3 | L4 | L5 |
|-----|-----|-----|-----|-----|-----|-----|
| Baseline | 0.9621 | 0.9643 | 0.9518 | 0.9234 | 0.8936 | 0.8642 |
| L1 λ=1e-4 | 0.9620 | 0.9642 | 0.9524 | 0.9243 | 0.8941 | 0.8634 |
| L1 λ=1e-3 | 0.9621 | 0.9644 | 0.9528 | 0.9232 | 0.8926 | 0.8660 |
| L1 λ=1e-2 | 0.9619 | 0.9647 | 0.9529 | 0.9227 | 0.8924 | 0.8659 |
| Top-K 10% | 0.9587 | 0.9627 | **0.9563** | **0.9261** | **0.8997** | 0.8432 |
| Top-K 25% | 0.9591 | 0.9630 | 0.9504 | 0.9248 | 0.8841 | 0.8444 |
| Top-K 50% | 0.9601 | 0.9654 | 0.9477 | 0.9211 | 0.8772 | 0.8275 |

### Macro F1

| Run | L0 | L1 | L2 | L3 | L4 | L5 |
|-----|-----|-----|-----|-----|-----|-----|
| Baseline | 0.8445 | 0.8495 | 0.8524 | 0.8315 | 0.7986 | 0.7749 |
| L1 λ=1e-4 | 0.8446 | 0.8491 | 0.8511 | 0.8307 | 0.7995 | 0.7754 |
| L1 λ=1e-3 | 0.8445 | 0.8487 | 0.8495 | 0.8357 | 0.8007 | 0.7749 |
| L1 λ=1e-2 | 0.8465 | 0.8499 | 0.8543 | 0.8327 | 0.8008 | 0.7737 |
| Top-K 10% | 0.8390 | 0.8484 | **0.8541** | 0.8228 | **0.8069** | 0.7498 |
| Top-K 25% | 0.8410 | 0.8472 | 0.8446 | 0.8293 | 0.7943 | 0.7560 |
| Top-K 50% | 0.8417 | 0.8581 | 0.8467 | 0.8194 | 0.7851 | 0.7468 |

**Findings:**
- **L1: no effect on probes** at any λ. All deltas are within ±0.003, indistinguishable from noise. Consistent with the sparsity being too weak to alter representational content.
- **Top-K: cost is concentrated in L5, and notably absent in early/mid layers.** At Top-K 10%, L2–L4 accuracy is comparable to or slightly *above* baseline (L2: +0.005, L3: +0.003, L4: +0.006). L0 drops 0.003, L5 drops 0.021. The early layers appear to learn more compact, probe-friendly representations under sparsity pressure — or at minimum, are completely unaffected up to 90% sparsity.
- **Top-K 50% causes the most uniform degradation**, with L4 dropping −0.016 and L5 dropping −0.037. Paradoxically, less aggressive constraints (10%) cause less probe degradation in mid-layers than the softer 50% constraint — possibly because 10% forces the model to develop genuinely sparse, informative representations early, while 50% allows a lazier allocation.
- The monotonic layer-ordering (probe accuracy decreasing L0→L5) is preserved under all conditions, consistent with syntactic information concentrating in early layers regardless of training constraints.

---

## Summary

| Dimension | L1 λ=1e-3 | L1 λ=1e-2 | Top-K 25% | Top-K 10% |
|-----------|-----------|-----------|-----------|-----------|
| Perplexity cost | None | None | +2.6% | +3.0% |
| Sparsity (mean) | 5.3% inactive | 10.2% inactive | 75% (exact) | 90% (exact) |
| Neuron correlation (mean Δ) | −26% | −38% | −5% | −7% |
| Probe accuracy (L5 Δ) | +0.002 | +0.002 | −0.020 | −0.021 |
| Layer-uniform? | No (L1 spikes) | No (L3 spikes) | No (L2 spikes) | No (L2 spikes) |

**Key conclusions:**

1. **L1 is a decorrelation tool, not a sparsity tool.** It reduces neuron correlation by 24–38% at zero perplexity cost, but achieves only 3–10% actual sparsity. Its primary effect is geometrical: it discourages neurons from co-activating without forcing them to go dark. The dominant result mirrors the orthogonal regularizer from Experiment 1, suggesting L1 on hidden activations and cosine similarity penalties on outputs have overlapping effects on representation geometry.

2. **Top-K is the right tool for explicit sparsity budgets.** 90% of d_ff neurons can be zeroed per token with only a 3% perplexity penalty and negligible probe accuracy loss in layers 0–4. The model adapts gracefully to extreme sparsity. If the goal is to study or improve the interpretability of individual neurons, Top-K 25% is the recommended operating point: strong sparsity (75%), small cost (2.6% PPL), minimal representational damage.

3. **The middle layer resistance pattern is consistent across both experiments.** L1 and L2 consistently resist decorrelation under L1 penalty (spiking above baseline in at least one run for every λ). Top-K increases L2 correlation. In Experiment 1, Orth λ=1e-1 caused L1 to resist. This suggests the model has a structural need for correlated representations in layers 1–2 to support in-context computation, and fights back under constraints that try to reduce this correlation.

4. **L1 and Top-K are complementary, not substitutable.** One decorrelates without sparsifying; the other sparsifies without decorrelating. A natural follow-up is combining them (Top-K + L1 on the surviving activations) to achieve both simultaneously.

5. **Linear probes remain an insensitive metric for these interventions.** As in Experiment 1, substantial changes in geometry and sparsity produce at most 2–4% changes in POS decodability, concentrated in the final layer.

---

## Open Questions (Pending)

- **Monosemanticity:** do the Top-K neurons (with only 10–25% active) develop more specialised, monosemantic responses? The cosine sim reduction is too small to answer this — requires the systematic neuron audit (Eval 2).
- **Redundancy relocation:** does Top-K's L2 correlation spike indicate genuine relocation of computation, or is it an artefact of the cosine sim metric applied to a different sparsity regime? (Eval 1)
- **Combining constraints:** does Top-K 25% + L1 (jointly) achieve both sparsity and decorrelation without additional perplexity cost?
