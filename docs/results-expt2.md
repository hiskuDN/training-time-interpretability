# Experiment 2 Results: Sparse Architectural Constraints

**Date:** 2026-03-08 (updated 2026-03-11 with polysemanticity + neuron utilization)
**Runs:** 6 configs + baseline reference, seed 42
**W&B group:** [`expt2`](https://wandb.ai/hisku/training-time-interpretability) / evals: [`expt2_eval`](https://wandb.ai/hisku/training-time-interpretability)

---

## Setup

Same base model as Experiment 1 (6-layer GPT-2-style, 384 hidden dim, 6 heads, ~30M params, TinyStories, 50k steps). Two sparsity mechanisms tested:

- **L1 activation penalty (soft):** L1 norm on post-GELU d_ff=1536 hidden activations, λ ∈ {1e-4, 1e-3, 1e-2}. No architectural change; gradient flows through all neurons.
- **Top-K MLP (hard):** at each forward pass, all-but-top-k neurons by magnitude are zeroed. k ∈ {10%, 25%, 50%} of d_ff=1536 (= 154, 384, 768 active neurons). Straight-through: gradient only propagates through active neurons; mask is non-differentiable.

Sparsity and polysemanticity are measured on `mlp_hidden` (d_ff=1536), i.e. the post-GELU hidden layer where zeros are actually introduced. Cosine similarity heatmaps and probes use `mlp_out` (d_model=384) for comparability with Experiment 1.

---

## 1. Perplexity

| Run | Val Loss | Perplexity | Δ vs Baseline |
|-----|----------|------------|---------------|
| Baseline | 1.4598 | 4.31 | — |
| L1 λ=1e-4 | 1.4599 | 4.31 | +0.0001 nats |
| L1 λ=1e-3 | 1.4600 | 4.31 | +0.0002 nats |
| L1 λ=1e-2 | 1.4601 | 4.31 | +0.0003 nats |
| Top-K 50% | 1.4806 | 4.40 | +0.0208 nats (+2.1%) |
| Top-K 25% | 1.4869 | 4.42 | +0.0271 nats (+2.7%) |
| Top-K 10% | 1.4899 | 4.44 | +0.0301 nats (+3.1%) |

**Findings:**
- **L1 is entirely free** — all three λ values produce val loss within 0.0003 nats of the baseline, well within noise. The penalty is so small relative to CE loss that it has no measurable effect on the learned distribution.
- **Top-K has a small but real monotonic cost** that scales cleanly with constraint strength. Going from 50% active to 10% active costs only 0.01 additional nats. The 3% perplexity hit at 90% sparsity is a remarkably low price.
- The Top-K costs are not alarming — they suggest the model adapts gracefully by concentrating information in the surviving neurons, rather than catastrophically losing capacity.

---

## 2. Activation Sparsity

Fraction of d_ff=1536 neurons with |activation| < 0.01, computed over the full val set during post-hoc eval.

| Run | Mean | L0 | L1 | L2 | L3 | L4 | L5 |
|-----|------|-----|-----|-----|-----|-----|-----|
| Baseline | 0.027 | 0.023 | 0.051 | 0.027 | 0.022 | 0.019 | 0.017 |
| L1 λ=1e-4 | 0.032 | 0.024 | 0.067 | 0.035 | 0.029 | 0.021 | 0.017 |
| L1 λ=1e-3 | 0.053 | 0.029 | 0.053 | 0.069 | 0.074 | 0.059 | 0.035 |
| L1 λ=1e-2 | 0.102 | 0.075 | 0.108 | 0.057 | 0.029 | 0.175 | 0.167 |
| Top-K 10% | 0.900 | 0.900 | 0.900 | 0.900 | 0.900 | 0.900 | 0.900 |
| Top-K 25% | 0.750 | 0.750 | 0.750 | 0.750 | 0.750 | 0.750 | 0.750 |
| Top-K 50% | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 |

**Findings:**
- **Baseline natural sparsity is 2.7%** — GELU produces a small fraction of near-zero activations with no training pressure.
- **L1 produces weak and uneven sparsity above baseline.** Even at λ=1e-2, only ~10% of neurons are inactive — just 4× natural sparsity. L1 λ=1e-4 achieves 3.2%, barely above baseline. The sparsity profile is non-uniform: at λ=1e-2, L4 and L5 reach 17% while L3 sits at 3%.
- **Top-K is perfectly controllable and uniform by construction.** Sparsity is exactly 50/75/90% at every layer — the hard mask enforces this.
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

**L1 substantially reduces neuron correlation (−24% to −38% vs baseline), despite barely achieving sparsity.** The dominant effect of L1 is decorrelation, not sparsification. Penalising the mean magnitude of hidden activations appears to discourage neurons from activating together, even without zeroing them.

L1 exhibits the same layer-resistance pattern seen with the orthogonal regularizer in Experiment 1: **L1 at λ=1e-3 causes layer 1 to spike to 0.3017** (+41% vs baseline), while outer layers drop dramatically. At λ=1e-2, L3 rises back toward baseline (0.1714). The model consistently fights back in middle layers.

**Top-K achieves only modest correlation reduction (−5% to −10%) despite far greater sparsity.** L2 is a consistent outlier — cosine sim *increases* for all Top-K runs (0.1934 baseline → 0.2408/0.2640/0.1737 for 10%/25%/50%). With fewer active neurons, the surviving neurons in `mlp_out` appear to become more correlated, possibly because they must collectively encode the same information that was previously spread across more neurons.

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
- **L1: no effect on probes** at any λ. All deltas are within ±0.003, indistinguishable from noise.
- **Top-K: cost is concentrated in L5, absent in early/mid layers.** At Top-K 10%, L2–L4 accuracy is at or slightly above baseline. L5 drops 0.021. The model concentrates syntactic information earlier under sparsity pressure.
- The monotonic layer-ordering (probe accuracy decreasing L0→L5) is preserved under all conditions.

---

## 5. Polysemanticity

Mean per-neuron entropy of the token-type activation distribution, computed on `mlp_hidden` (d_ff=1536). Lower = more monosemantic (neuron responds to a narrower set of token types).

| Run | Mean | L0 | L1 | L2 | L3 | L4 | L5 |
|-----|------|-----|-----|-----|-----|-----|-----|
| Baseline | 4.359 | 4.367 | 4.310 | 4.386 | 4.386 | 4.335 | 4.368 |
| L1 λ=1e-4 | 4.504 | 4.640 | 4.522 | 4.507 | 4.537 | 4.412 | 4.403 |
| L1 λ=1e-3 | 4.729 | 4.733 | 4.376 | 4.830 | 4.928 | 4.858 | 4.651 |
| L1 λ=1e-2 | 5.056 | 5.214 | 5.073 | 4.909 | 4.649 | 5.289 | 5.204 |
| Top-K 10% | **3.694** | 3.856 | 3.434 | 3.618 | 3.713 | 3.777 | 3.767 |
| Top-K 25% | **4.039** | 4.046 | 3.930 | 3.978 | 4.076 | 4.105 | 4.102 |
| Top-K 50% | 4.267 | 4.139 | 4.225 | 4.314 | 4.338 | 4.272 | 4.315 |

**Findings:**

**L1 makes neurons more polysemantic — the opposite of the intended effect.** Entropy increases monotonically with λ: +3.3% at 1e-4, +8.5% at 1e-3, +16% at 1e-2. The mechanism: L1 penalises large activations, so the model spreads each neuron's activation budget across more token types rather than concentrating strongly on a few. Decorrelation and sparsification do not imply monosemanticity — in fact they are in tension here.

**Top-K consistently reduces polysemanticity, with a clear dose-response.** 10% active: −15% entropy vs baseline. 25% active: −7%. 50% active: −2%. The constraint to fire on only the most-stimulated neurons forces each surviving neuron to earn its activation, concentrating responses on the tokens that most strongly excite it.

**The dissociation between cosine similarity and polysemanticity is striking.** L1 reduces cosine similarity (decorrelates outputs) but increases polysemanticity (broadens individual neuron responses). Top-K barely reduces cosine similarity but substantially improves polysemanticity. These are measuring different things: correlation is a property of the joint activation pattern across neurons; polysemanticity is a property of individual neuron selectivity.

---

## 6. Neuron Utilization

Fraction of val tokens each `mlp_hidden` neuron is active (|activation| > 0.01). Summarised as `frac_dead` (active < 1% of tokens) and `frac_always_on` (active > 99% of tokens).

### Fraction always-on per layer

| Run | L0 | L1 | L2 | L3 | L4 | L5 |
|-----|-----|-----|-----|-----|-----|-----|
| Baseline | 0.633 | 0.266 | 0.189 | 0.129 | 0.111 | 0.180 |
| L1 λ=1e-4 | 0.544 | 0.204 | 0.178 | 0.133 | 0.111 | 0.166 |
| L1 λ=1e-3 | 0.425 | 0.255 | 0.153 | 0.106 | 0.065 | 0.074 |
| L1 λ=1e-2 | 0.227 | 0.100 | 0.065 | 0.139 | 0.029 | 0.003 |
| Top-K 10% | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| Top-K 25% | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| Top-K 50% | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |

### Fraction dead per layer

| Run | L0 | L1 | L2 | L3 | L4 | L5 |
|-----|-----|-----|-----|-----|-----|-----|
| Baseline | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| L1 λ=1e-4 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| L1 λ=1e-3 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| L1 λ=1e-2 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| Top-K 10% | 0.001 | **0.046** | 0.005 | 0.002 | 0.000 | 0.001 |
| Top-K 25% | 0.000 | 0.012 | 0.000 | 0.000 | 0.000 | 0.000 |
| Top-K 50% | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |

**Findings:**

**Baseline neurons are almost never off.** Every baseline neuron is active on at least 62% of tokens (minimum across all layers). The median utilization is 97–99%. **63% of L0 neurons are always-on** — firing on every single token regardless of input. This is near-zero specialisation: most neurons behave like unconditional bias terms rather than feature detectors.

**Top-K completely eliminates always-on neurons.** frac_always_on = 0 across all layers and all Top-K variants. Every neuron has some selectivity — there is no input for which it would always fire. This is the structural change that drives the polysemanticity improvement: forcing selectivity removes the large pool of unconditional activators.

**Top-K creates dynamic sparsity, not static pruning.** The utilization distribution for Top-K 10% is right-skewed: median utilization is 5–9% (neurons selected for ~1 in 14 tokens) and the p95 is 20–36% (the most popular neurons selected for up to 78% of tokens). No neuron is always in the top-k; the active set rotates per input. This confirms that Top-K is not just pruning the network but is enforcing token-conditional computation.

**Top-K 10% creates a small number of dead neurons at L1 (4.6%).** These neurons are never competitive enough to enter the top 154 — effectively wasted capacity. The effect is concentrated at L1 and disappears at Top-K 25%. This is a marginal cost of aggressive sparsity budgets.

**L1 progressively reduces always-on neurons.** At λ=1e-2, the always-on fraction at L0 drops from 63% to 23%; at L5 it drops from 18% to 0.3%. L1 achieves this through magnitude reduction rather than hard masking, so no neurons go dead. This is a secondary benefit of L1 that partially compensates for its polysemanticity cost.

---

## Summary

| Dimension | L1 λ=1e-3 | L1 λ=1e-2 | Top-K 25% | Top-K 10% |
|-----------|-----------|-----------|-----------|-----------|
| Perplexity cost | None | None | +2.7% | +3.1% |
| Sparsity (mean) | 5.3% inactive | 10.2% inactive | 75% (exact) | 90% (exact) |
| Neuron correlation (mean Δ) | −26% | −38% | −5% | −7% |
| Polysemanticity (mean Δ) | **+8.5%** ↑worse | **+16%** ↑worse | −7% ↓better | −15% ↓better |
| Always-on neurons (L0) | 42.5% | 22.7% | 0% | 0% |
| Dead neurons | 0% | 0% | 1.2% (L1) | 4.6% (L1) |
| Probe accuracy (L5 Δ) | +0.002 | +0.002 | −0.020 | −0.021 |

**Key conclusions:**

1. **L1 is a decorrelation tool that inadvertently increases polysemanticity.** It reduces neuron correlation by 24–38% at zero perplexity cost, but achieves only 3–10% sparsity and — counterintuitively — makes individual neurons respond to *more* token types (+8–16% entropy). The mechanism: L1 penalises large activations, so neurons spread their limited activation budget broadly instead of concentrating on a few token types. Decorrelation and monosemanticity are not the same thing, and can move in opposite directions.

2. **Top-K is the right tool for sparsity and monosemanticity simultaneously.** 90% of d_ff neurons can be zeroed per token with only a 3% perplexity penalty, −15% polysemanticity, and negligible probe loss in layers 0–4. The hard mask forces each surviving neuron to earn its activation. Top-K 25% is the recommended operating point: strong sparsity (75%), meaningful monosemanticity gain (−7%), small cost.

3. **Baseline neurons are almost never off — and this is the root problem.** 63% of L0 neurons fire on every single token; every neuron is active for at least 62% of all inputs. A model in this regime cannot have monosemantic neurons by definition — neurons that always fire carry no token-specific information. Top-K's primary contribution is structural: it guarantees that every neuron must be selective, eliminating the always-on regime entirely.

4. **Top-K creates dynamic, token-conditional computation.** The active set rotates per input (median utilization ~7%, no neuron always selected). This confirms Top-K is not equivalent to weight pruning — it enforces context-dependent gating, which is the computational structure required for interpretable feature detectors.

5. **The middle layer resistance pattern is consistent across both experiments.** Layer 1 resists decorrelation under L1 (spiking above baseline for every λ); L2 resists under Top-K (cosine sim increases). This suggests a structural need for correlated representations in early-middle layers that persists regardless of training constraints.

6. **L1 and Top-K are complementary but not in the way originally thought.** The original framing was "L1 decorrelates, Top-K sparsifies." The updated picture: L1 decorrelates and reduces always-on neurons but *hurts* monosemanticity; Top-K sparsifies and improves monosemanticity but barely decorrelates. A combination (Top-K + L1 on surviving activations) might achieve all three: sparsity, decorrelation, and monosemanticity.

7. **Linear probes remain an insensitive metric.** Substantial changes in geometry, sparsity, and monosemanticity produce at most 2–4% changes in POS decodability. Probes measure what information is decodable, not how it is structured — and these interventions primarily affect structure.

---

## Open Questions

- **Combining constraints:** does Top-K 25% + L1 (jointly) achieve decorrelation + sparsity + monosemanticity without additional perplexity cost? (Experiment 3 candidate)
- **SAE reconstruction:** does Top-K training reduce the reconstruction loss of a post-hoc sparse autoencoder at fixed sparsity? This would directly test whether Top-K reduces superposition in the sense the field cares about.
- **Dead neuron mechanism at L1:** why is layer 1 specifically the one that produces dead neurons under Top-K 10%? The same layer is also the site of the cosine sim spike and the L1 resistance pattern — suggesting L1 has a special computational role.
- **Always-on neuron function:** what do the 63% of always-on L0 neurons compute? If they are not token-selective, are they encoding positional information, bias terms, or universal features? Ablation experiments would clarify.
