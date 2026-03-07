from __future__ import annotations

import torch


def compute_interpretability_metrics(
    activations: list[dict],
    input_ids: torch.Tensor,
) -> dict:
    """
    Compute interpretability metrics from collected validation activations.

    Args:
        activations: list of activation dicts, each {layer_idx: {"mlp_out": ..., "attn_out": ...}}
                     tensors are on CPU, shape [batch, seq_len, d_model]
        input_ids:   all input_ids concatenated, shape [total_batch, seq_len]

    Returns:
        dict of metric_name -> float
    """
    metrics = {}
    metrics.update(_activation_sparsity(activations))
    metrics.update(_polysemanticity_score(activations, input_ids))
    return metrics


def _activation_sparsity(activations: list[dict], threshold: float = 0.01) -> dict:
    """
    Fraction of neurons that are inactive (|act| < threshold) per layer,
    averaged over all token positions.

    Prefers mlp_hidden (d_ff) when available — this is where Top-K zeros are
    introduced and where L1 acts. Falls back to mlp_out (d_model) otherwise.
    """
    if not activations:
        return {}

    layer_indices = list(activations[0].keys())
    metrics = {}

    for layer_idx in layer_indices:
        key = "mlp_hidden" if activations[0][layer_idx].get("mlp_hidden") is not None else "mlp_out"
        all_acts = torch.cat(
            [batch[layer_idx][key].reshape(-1, batch[layer_idx][key].shape[-1])
             for batch in activations],
            dim=0,
        )  # [total_tokens, d_ff or d_model]

        inactive = (all_acts.abs() < threshold).float().mean().item()
        metrics[f"val/sparsity/layer_{layer_idx}"] = inactive

    mean_sparsity = sum(
        v for k, v in metrics.items() if k.startswith("val/sparsity/layer_")
    ) / max(len(layer_indices), 1)
    metrics["val/sparsity/mean"] = mean_sparsity

    return metrics


def _polysemanticity_score(activations: list[dict], input_ids: torch.Tensor) -> dict:
    """
    Average entropy of per-neuron token-type activation distributions.

    For each neuron, we compute a distribution over token types based on
    which tokens activate it most strongly, then measure the entropy.
    High entropy = polysemantic. Reported per layer and as a mean.
    """
    if not activations:
        return {}

    layer_indices = list(activations[0].keys())
    # input_ids is already flattened to [total_tokens] by the caller
    ids_flat = input_ids
    metrics = {}

    for layer_idx in layer_indices:
        all_acts = torch.cat(
            [batch[layer_idx]["mlp_out"].reshape(-1, batch[layer_idx]["mlp_out"].shape[-1])
             for batch in activations],
            dim=0,
        ).abs()  # [total_tokens, d_model]

        N, D = all_acts.shape
        # Trim ids to match (sequence lengths may differ slightly due to last batch)
        n = min(N, ids_flat.shape[0])
        acts = all_acts[:n]
        ids = ids_flat[:n]

        max_id = int(ids.max().item()) + 1
        token_acts = torch.zeros(max_id, D, dtype=acts.dtype)
        token_acts.scatter_add_(0, ids.unsqueeze(1).expand_as(acts), acts)

        unique_ids = ids.unique()
        token_dist = token_acts[unique_ids]  # [V, D]
        token_dist = token_dist / (token_dist.sum(dim=0, keepdim=True) + 1e-10)

        entropy = -(token_dist * torch.log(token_dist + 1e-10)).sum(dim=0).mean().item()
        metrics[f"val/polysemanticity/layer_{layer_idx}"] = entropy

    mean_poly = sum(
        v for k, v in metrics.items() if k.startswith("val/polysemanticity/layer_")
    ) / max(len(layer_indices), 1)
    metrics["val/polysemanticity/mean"] = mean_poly

    return metrics
