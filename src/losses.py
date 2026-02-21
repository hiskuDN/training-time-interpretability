from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn.functional as F


class Regularizer(ABC):
    """
    Base class for training-time interpretability regularizers.

    Each regularizer receives the model's intermediate activations and
    optionally other context (e.g. input_ids), and returns a scalar loss.
    The Trainer multiplies by `self.weight` before adding to the CE loss.
    """

    name: str
    weight: float

    @abstractmethod
    def compute(self, activations: dict, **context) -> torch.Tensor:
        """
        Args:
            activations: {layer_idx: {"attn_out": Tensor, "mlp_out": Tensor}}
                         each tensor shape [batch, seq_len, d_model]
            **context:   additional tensors (e.g. input_ids=[batch, seq_len])

        Returns:
            Scalar loss tensor (unweighted).
        """
        ...


# ---------------------------------------------------------------------------
# Orthogonality regularizer
# ---------------------------------------------------------------------------

class OrthogonalityRegularizer(Regularizer):
    """
    Penalize cosine similarity between neuron activation vectors.

    For each layer, each neuron's activation pattern across all tokens in the
    batch is treated as a vector. We penalize pairwise cosine similarity
    between these vectors, encouraging neurons to activate on different tokens.

    Corrects the bug in the original spec pseudocode (which computed
    token-token similarity [batch, seq_len, seq_len] instead of
    neuron-neuron similarity [d_model, d_model]).
    """

    def __init__(self, weight: float, max_tokens: int = 4096, layers: str = "all"):
        self.name = "orthogonality"
        self.weight = weight
        self.max_tokens = max_tokens
        self._layers = layers  # "all" or comma-separated indices e.g. "0,2,4"

    def _target_layers(self, activations: dict) -> list[int]:
        if self._layers == "all":
            return list(activations.keys())
        return [int(i) for i in self._layers.split(",")]

    def compute(self, activations: dict, **context) -> torch.Tensor:
        penalty = torch.tensor(0.0, device=next(iter(activations.values()))["mlp_out"].device)
        count = 0

        for layer_idx in self._target_layers(activations):
            if layer_idx not in activations:
                continue
            acts = activations[layer_idx]["mlp_out"]  # [B, S, D]
            B, S, D = acts.shape

            # Flatten tokens: [B*S, D]
            acts_flat = acts.reshape(B * S, D)

            # Subsample for efficiency
            N = acts_flat.shape[0]
            if N > self.max_tokens:
                idx = torch.randperm(N, device=acts_flat.device)[: self.max_tokens]
                acts_flat = acts_flat[idx]

            # Each row of neuron_vecs is one neuron's activation pattern: [D, N']
            neuron_vecs = F.normalize(acts_flat.T, dim=1)  # [D, N']

            # Neuron-neuron cosine similarity: [D, D]
            sim = torch.mm(neuron_vecs, neuron_vecs.T)

            # Penalize off-diagonal elements
            eye = torch.eye(D, device=sim.device)
            penalty = penalty + (sim - eye).abs().mean()
            count += 1

        return penalty / max(count, 1)


# ---------------------------------------------------------------------------
# Polysemanticity regularizer
# ---------------------------------------------------------------------------

class PolysemanticitRegularizer(Regularizer):
    """
    Penalize neurons that activate on many different token types.

    For each neuron, we build a distribution over token types by aggregating
    activation magnitudes per token type. The entropy of this distribution
    measures polysemanticity: high entropy = the neuron responds to many
    different tokens = polysemantic. We minimize average entropy across neurons.

    Requires input_ids to be passed via context.
    """

    def __init__(self, weight: float, layers: str = "all"):
        self.name = "polysemanticity"
        self.weight = weight
        self._layers = layers

    def _target_layers(self, activations: dict) -> list[int]:
        if self._layers == "all":
            return list(activations.keys())
        return [int(i) for i in self._layers.split(",")]

    def compute(self, activations: dict, **context) -> torch.Tensor:
        input_ids: torch.Tensor = context["input_ids"]  # [B, S]

        first_acts = next(iter(activations.values()))["mlp_out"]
        device = first_acts.device

        penalty = torch.tensor(0.0, device=device)
        count = 0

        for layer_idx in self._target_layers(activations):
            if layer_idx not in activations:
                continue
            acts = activations[layer_idx]["mlp_out"]  # [B, S, D]
            B, S, D = acts.shape

            acts_flat = acts.reshape(B * S, D).abs()  # [N, D] — use magnitudes
            ids_flat = input_ids.reshape(B * S)       # [N]

            # Aggregate activation magnitudes by token type via scatter_add.
            # Accumulate in float32 to avoid fp16 overflow under AMP.
            max_id = int(ids_flat.max().item()) + 1
            token_acts = torch.zeros(max_id, D, device=device, dtype=torch.float32)
            token_acts.scatter_add_(
                0,
                ids_flat.unsqueeze(1).expand_as(acts_flat),
                acts_flat.float(),
            )

            # Keep only rows for tokens that actually appear
            unique_ids = ids_flat.unique()
            token_dist = token_acts[unique_ids]  # [V, D]

            # Normalize per neuron -> distribution over token types
            token_dist = token_dist / (token_dist.sum(dim=0, keepdim=True) + 1e-10)

            # Entropy per neuron: H = -sum(p * log(p))
            log_dist = torch.log(token_dist + 1e-10)
            entropy = -(token_dist * log_dist).sum(dim=0).mean()  # scalar

            penalty = penalty + entropy
            count += 1

        return penalty / max(count, 1)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_regularizers(reg_configs: list) -> list[Regularizer]:
    """Build regularizer instances from a list of RegularizerConfig objects."""
    registry = {
        "orthogonality": lambda rc: OrthogonalityRegularizer(
            weight=rc.weight,
            max_tokens=rc.max_tokens,
            layers=rc.layers,
        ),
        "polysemanticity": lambda rc: PolysemanticitRegularizer(
            weight=rc.weight,
            layers=rc.layers,
        ),
    }
    regularizers = []
    for rc in reg_configs:
        if rc.name in registry:
            regularizers.append(registry[rc.name](rc))
    return regularizers
