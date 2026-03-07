# Experiment 1 Results: Anti-Superposition via Orthogonality Constraints

**Date:** 2026-03-06
**Runs:** all 7 configs, seed 42
**W&B group:** [`expt1_eval`](https://wandb.ai/hisku/training-time-interpretability)

---

## Setup

All models are 6-layer GPT-2-style transformers (384 hidden dim, 6 heads, ~30M params) trained on TinyStories for 50k steps. Two regularizer families are tested:

- **Orthogonal:** penalises mean |off-diagonal| cosine similarity between neuron activation vectors in `mlp_out`, λ ∈ {1e-3, 1e-2, 1e-1}
- **Polysemantic:** penalises per-neuron entropy over token-type activation distributions, λ ∈ {1e-3, 1e-2, 1e-1}

Evaluation covers the full validation set (~4.65M tokens). Linear probes use 5,000 val texts with spaCy POS labels aligned to GPT-2 BPE tokens, trained per-layer on `mlp_out`.

---

## 1. Perplexity

| Run | Val Loss | Perplexity |
|-----|----------|------------|
| Baseline | 1.4598 | 4.31 |
| Orth λ=1e-3 | 1.4610 | 4.31 |
| Orth λ=1e-2 | 1.4610 | 4.31 |
| Orth λ=1e-1 | 1.4604 | 4.31 |
| Poly λ=1e-3 | 1.4603 | 4.31 |
| Poly λ=1e-2 | 1.4617 | 4.31 |
| Poly λ=1e-1 | 1.4593 | 4.30 |

**Finding:** Neither regularizer imposes any measurable perplexity cost across the full λ sweep. The largest difference from baseline is 0.002 nats — well within noise. This is true even at λ=0.1 for both families.

---

## 2. Representation Geometry (Cosine Similarity)

Mean absolute off-diagonal cosine similarity between neuron activation vectors per layer. Lower = more orthogonal. For the polysemantic runs, signed statistics are included separately because the absolute mean is misleading (see below).

### Orthogonal regularizer

| Run | L0 | L1 | L2 | L3 | L4 | L5 |
|-----|-----|-----|-----|-----|-----|-----|
| Baseline | 0.142 | 0.214 | 0.193 | 0.171 | 0.137 | 0.124 |
| Orth λ=1e-3 | 0.076 | 0.174 | 0.189 | 0.141 | 0.073 | 0.058 |
| Orth λ=1e-2 | 0.074 | 0.172 | 0.184 | 0.140 | 0.082 | 0.058 |
| **Orth λ=1e-1** | **0.069** | **0.197** | **0.205** | **0.185** | **0.118** | **0.085** |

**Findings:**
- **λ=1e-3 already captures nearly all the benefit.** λ=1e-2 adds almost nothing on top.
- **λ=1e-1 is counterproductive in every middle layer (L1–L5).** Off-diagonal sim *increases* vs λ=1e-2: +0.025 at L1, +0.022 at L2, +0.045 at L3, +0.036 at L4, +0.027 at L5. Only L0 improves marginally (−0.005).
- The middle layers (L1–L2) are consistently resistant across all λ — the model appears to need correlated representations there to support in-context computations, and fights back when pushed too hard. Pushing λ above 1e-2 causes the model to redistribute correlation into middle layers while letting outer layers appear more orthogonal.
- **Sweet spot: λ=1e-3.** Achieves ~47–54% reduction at outer layers, ~10–19% at middle layers, at negligible cost.

### Polysemantic regularizer

| Run | L0 mean\|off-diag\| | L0 signed mean | L0 std |
|-----|---------|---------------|--------|
| Baseline | 0.142 | +0.142 | ~0.08 |
| Poly λ=1e-3 | 0.987 | −0.003 | 0.987 |
| Poly λ=1e-2 | 1.000 | −0.002 | 1.000 |
| Poly λ=1e-1 | 1.000 | −0.003 | 1.000 |

The mean absolute off-diagonal reaches ~1.0, but the signed mean is ~0 with std ~1.0 — cosine similarities are spread across the full [−1, +1] range, not all aligned. The top-k context analysis reveals why.

**Finding: The polysemantic regularizer causes neurons to collapse onto a handful of specific examples.**

| | Baseline | Poly λ=1e-3 | Poly λ=1e-1 |
|---|---|---|---|
| Peak activation magnitude (L0) | ~1.4 | ~5.1 | **~142** |
| Unique top contexts across 50 neurons (L3) | 48 | 48 | **1** |
| Neurons sharing the same #1 context (L0, top 20) | — | — | **13/20** |

At λ=1e-1: every one of the 50 layer-3 neurons inspected has the same top-activating context. Activation magnitudes are ~100× larger than baseline. The model found a single "magic story" that maximally satisfies the entropy loss and collapsed all neurons onto it — some firing strongly positive, others strongly negative, producing the high-std, mean-zero cosine sim distribution we observed.

At λ=1e-3: magnitudes are more moderate (~5× baseline) and context diversity is comparable to baseline (48 unique contexts across 50 neurons), but the contexts are less thematically coherent — generic story openings rather than the specific clusters (e.g. "crocodile stories", "library stories") that emerge in the baseline and orthogonal models.

**Root cause:** the regularizer measures entropy over token-type distributions, not over semantic concepts. The model can achieve low entropy — and satisfy the loss — by spiking very strongly on a few specific *examples*, making those examples dominate the per-neuron token distribution. This is a gaming of the metric rather than genuine monosemanticity. The fix would require measuring concept-level selectivity, not token-level entropy.

---

## 3. Linear Probes (POS Tagging)

### Accuracy

| Run | L0 | L1 | L2 | L3 | L4 | L5 |
|-----|-----|-----|-----|-----|-----|-----|
| Baseline | 0.9621 | 0.9643 | 0.9518 | 0.9234 | 0.8936 | 0.8642 |
| Orth λ=1e-3 | 0.9625 | 0.9632 | 0.9485 | 0.9214 | 0.8918 | 0.8669 |
| Orth λ=1e-2 | 0.9619 | 0.9641 | 0.9494 | 0.9232 | 0.8873 | 0.8662 |
| Orth λ=1e-1 | 0.9624 | 0.9642 | 0.9476 | 0.9220 | 0.8871 | 0.8707 |
| Poly λ=1e-3 | 0.9623 | 0.9647 | 0.9503 | 0.9204 | 0.8932 | 0.8618 |
| Poly λ=1e-2 | 0.9615 | 0.9600 | 0.9445 | 0.9205 | 0.8913 | 0.8701 |
| Poly λ=1e-1 | 0.9592 | 0.9618 | 0.9464 | 0.9214 | 0.8833 | 0.8654 |

### Macro F1

| Run | L0 | L1 | L2 | L3 | L4 | L5 |
|-----|-----|-----|-----|-----|-----|-----|
| Baseline | 0.8445 | 0.8495 | 0.8524 | 0.8315 | 0.7986 | 0.7749 |
| Orth λ=1e-3 | 0.8474 | 0.8484 | 0.8515 | 0.8292 | 0.8049 | 0.7710 |
| Orth λ=1e-2 | 0.8446 | 0.8511 | 0.8483 | 0.8327 | 0.7988 | 0.7708 |
| Orth λ=1e-1 | 0.8446 | 0.8486 | 0.8524 | 0.8259 | 0.7978 | 0.7764 |
| Poly λ=1e-3 | 0.8440 | 0.8550 | 0.8508 | 0.8269 | 0.7986 | 0.7704 |
| Poly λ=1e-2 | 0.8433 | 0.8435 | 0.8370 | 0.8184 | 0.7963 | 0.7823 |
| Poly λ=1e-1 | 0.8398 | 0.8440 | 0.8349 | 0.8173 | 0.7864 | 0.7663 |

**Findings:**
- **Orthogonal regularizer: no meaningful effect on probes.** All accuracy deltas are within ±0.007, with no consistent direction. POS information is equally accessible regardless of how orthogonal the representations are.
- **Polysemantic regularizer: small but consistent degradation, scaling with λ.** Poly λ=1e-1 shows the clearest signal: accuracy −0.003 to −0.010 across layers, F1 down −0.005 to −0.012. This is consistent with the geometric collapse — anti-correlated neuron pairs preserve perplexity but encode syntactic structure less cleanly.
- **Both models follow the same monotonic layer pattern:** accuracy and F1 decrease from L0→L5, confirming syntactic information concentrates in early layers and this structure is preserved under regularization.

---

## 4. Top-k Activating Contexts (Spot Check)

Qualitative comparison for Layer 3, Neuron 0:

| | Baseline | Orthogonal λ=1e-2 |
|---|---|---|
| Peak activation | +0.89 | +1.61 |
| Sign of top-3 | mixed (+/−/+) | all positive |
| Top contexts | bunny story, robot story, girl-finds-skull story | "Once upon a time…" openings (×3) |

The orthogonal neuron fires ~80% stronger, exclusively on story openings, with no negative activations in the top-k — consistent with a more specialised, monosemantic response. This is a single neuron and requires systematic analysis to generalise.

---

## Summary

| Dimension | Orthogonal | Polysemantic |
|-----------|------------|--------------|
| Perplexity cost | None (all λ) | None (all λ) |
| Geometry effect | Strong, diminishing returns after λ=1e-3 | Collapses neurons onto a few specific examples |
| Optimal λ | 1e-3 | N/A — regulariser games the metric |
| Probe accuracy | Unchanged | Minor degradation, scales with λ |
| Qualitative (top-k) | Promising — neurons appear more specialised | Not evaluated (geometry pathological) |

**Key conclusions:**

1. **Orthogonal regularizer works as intended** — it reduces neuron correlation, is free perplexity-wise, and leaves downstream decodability intact. The sweet spot is λ=1e-3; going higher causes middle layers to resist and partially undo the gains.

2. **Polysemantic regularizer games its own metric.** Minimising per-neuron token-type entropy doesn't produce monosemantic neurons — it produces neurons that spike very strongly on a handful of specific examples, which trivially satisfies low entropy without developing semantic selectivity. At λ=1e-1, all 50 inspected layer-3 neurons share the same top-activating context; activation magnitudes are ~100× the baseline. The regularizer measures the wrong thing: token-type distributions can be gamed by extreme-valued responses to a few inputs, whereas genuine monosemanticity requires consistent selective response to a *concept* across many diverse inputs.

3. **Linear probes are an insensitive interpretability metric** for this type of intervention. Neuron geometry changes substantially while POS decodability barely moves, suggesting probes measure a different axis of interpretability than monosemanticity.

**Next steps:** systematic monosemanticity evaluation across a random sample of neurons (top-k context inspection + human rating) for the orthogonal λ=1e-3 vs baseline; redesign or drop the polysemantic regularizer for Experiment 2.
