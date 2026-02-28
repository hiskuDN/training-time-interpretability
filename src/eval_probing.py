"""
Evaluation functions for Experiment 1: perplexity, cosine-similarity heatmaps,
top-k activating contexts per neuron, and linear probes with POS tags.
"""
from __future__ import annotations

import heapq
import json
import math
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Perplexity
# ---------------------------------------------------------------------------

def compute_perplexity(model, val_loader, device, max_batches: int = -1) -> dict:
    """
    Full val-set perplexity with weighted token averaging.

    Returns:
        {val_loss, perplexity, n_tokens, n_batches}
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    n_batches = 0

    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if max_batches > 0 and i >= max_batches:
                break
            input_ids = batch["input_ids"].to(device)
            targets = batch["targets"].to(device)

            _, ce_loss, _ = model(input_ids, targets=targets, return_activations=False)

            n_tokens = (targets != -100).sum().item()
            total_loss += ce_loss.item() * n_tokens
            total_tokens += n_tokens
            n_batches += 1

    val_loss = total_loss / max(total_tokens, 1)
    perplexity = math.exp(val_loss)

    return {
        "val_loss": val_loss,
        "perplexity": perplexity,
        "n_tokens": total_tokens,
        "n_batches": n_batches,
    }


# ---------------------------------------------------------------------------
# Cosine similarity heatmaps
# ---------------------------------------------------------------------------

def compute_cosine_sim_heatmaps(
    model,
    val_loader,
    device,
    max_tokens: int = 32768,
) -> dict:
    """
    Compute per-layer [d_model, d_model] neuron-neuron cosine similarity matrices
    using reservoir sampling over up to max_tokens token rows.

    Returns:
        {
            sim_matrices: {layer_idx: ndarray[d_model, d_model]},
            mean_off_diag: {layer_idx: float},
        }
    """
    model.eval()
    n_layers = len(model.blocks)

    # Reservoir: list of (layer_idx -> list of row tensors on CPU)
    reservoirs: dict[int, list] = {i: [] for i in range(n_layers)}
    counts: dict[int, int] = {i: 0 for i in range(n_layers)}

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            _, _, activations = model(input_ids, return_activations=True)

            for layer_idx, acts_dict in activations.items():
                acts = acts_dict["mlp_out"]  # [B, S, D]
                B, S, D = acts.shape
                rows = acts.reshape(B * S, D).cpu().float()  # move to CPU immediately
                N = rows.shape[0]

                reservoir = reservoirs[layer_idx]
                k = counts[layer_idx]

                for j in range(N):
                    k += 1
                    if len(reservoir) < max_tokens:
                        reservoir.append(rows[j])
                    else:
                        # Reservoir sampling: replace with probability max_tokens/k
                        r = int(torch.randint(0, k, (1,)).item())
                        if r < max_tokens:
                            reservoir[r] = rows[j]
                counts[layer_idx] = k

    sim_matrices = {}
    mean_off_diag = {}

    for layer_idx in range(n_layers):
        reservoir = reservoirs[layer_idx]
        if not reservoir:
            continue
        # Stack: [N', D]
        mat = torch.stack(reservoir, dim=0)  # [N', D]
        D = mat.shape[1]

        # Neuron vectors: normalize each neuron's activation pattern
        neuron_vecs = F.normalize(mat.T, dim=1)  # [D, N']
        sim = torch.mm(neuron_vecs, neuron_vecs.T).numpy()  # [D, D]

        sim_matrices[layer_idx] = sim

        # Mean off-diagonal absolute value
        eye = np.eye(D)
        off_diag = sim[~eye.astype(bool)]
        mean_off_diag[layer_idx] = float(np.abs(off_diag).mean())

    return {
        "sim_matrices": sim_matrices,
        "mean_off_diag": mean_off_diag,
    }


# ---------------------------------------------------------------------------
# Top-k activating contexts per neuron
# ---------------------------------------------------------------------------

class TopKContextCollector:
    """
    Streaming min-heap (size k) per neuron per layer.

    Call .update(activations, input_ids, tokenizer) per batch.
    finalize() returns a JSON-serializable dict.
    """

    def __init__(self, k: int = 20, n_layers: int = 6, d_model: int = 384):
        self.k = k
        self.n_layers = n_layers
        self.d_model = d_model
        self._counter = 0  # tiebreaker: prevents dict comparison in heap tuples
        # heaps[layer][neuron] = list of (abs_activation, counter, entry_dict) min-heap
        self.heaps: dict[int, dict[int, list]] = {
            layer: {neuron: [] for neuron in range(d_model)}
            for layer in range(n_layers)
        }

    def update(self, activations: dict, input_ids: torch.Tensor, tokenizer):
        """
        Args:
            activations: {layer_idx: {"mlp_out": Tensor[B,S,D]}}
            input_ids:   Tensor[B, S]
            tokenizer:   HuggingFace tokenizer (for decoding context text)
        """
        for layer_idx, acts_dict in activations.items():
            if layer_idx not in self.heaps:
                continue
            acts = acts_dict["mlp_out"]  # [B, S, D]
            B, S, D = acts.shape

            # Per-neuron max across all tokens in this batch for pre-filtering
            layer_max = acts.reshape(B * S, D).abs().max(dim=0).values  # [D]

            for neuron_idx in range(min(D, self.d_model)):
                heap = self.heaps[layer_idx][neuron_idx]
                # Pre-filter: skip if max activation < current heap minimum
                if len(heap) >= self.k:
                    heap_min = heap[0][0]
                    if layer_max[neuron_idx].item() <= heap_min:
                        continue

                # Find top activation position in this batch
                neuron_acts = acts[:, :, neuron_idx]  # [B, S]
                flat_idx = neuron_acts.abs().argmax().item()
                b_idx = flat_idx // S
                s_idx = flat_idx % S
                act_val = neuron_acts[b_idx, s_idx].item()

                context_ids = input_ids[b_idx].tolist()
                context_text = tokenizer.decode(context_ids, skip_special_tokens=True)

                entry = {
                    "activation": act_val,
                    "token_idx": s_idx,
                    "input_ids": context_ids,
                    "context_text": context_text,
                }

                self._counter += 1
                heap_entry = (abs(act_val), self._counter, entry)
                if len(heap) < self.k:
                    heapq.heappush(heap, heap_entry)
                elif abs(act_val) > heap[0][0]:
                    heapq.heapreplace(heap, heap_entry)

    def finalize(self) -> dict:
        """
        Returns JSON-serializable:
            {layer_i: {neuron_j: [{rank, activation, context_text, input_ids}]}}
        """
        result = {}
        for layer_idx, neurons in self.heaps.items():
            result[str(layer_idx)] = {}
            for neuron_idx, heap in neurons.items():
                # Sort descending by |activation|
                sorted_entries = sorted(heap, key=lambda x: x[0], reverse=True)
                ranked = []
                for rank, (act_abs, _counter, entry) in enumerate(sorted_entries):
                    ranked.append({
                        "rank": rank,
                        "activation": entry["activation"],
                        "context_text": entry["context_text"],
                        "input_ids": entry["input_ids"],
                    })
                result[str(layer_idx)][str(neuron_idx)] = ranked
        return result


# ---------------------------------------------------------------------------
# POS alignment
# ---------------------------------------------------------------------------

def align_pos_to_gpt2_tokens(
    text: str,
    gpt2_offset_mapping: list[tuple[int, int]],
    spacy_annotations,
) -> list[str]:
    """
    Align spaCy word-level POS to GPT-2 BPE tokens via character span majority-overlap.

    BOS/EOS tokens (offset (0,0)) are labelled "IGNORE".
    Subwords within the same word all receive the same POS tag.

    Args:
        text:                  raw text string
        gpt2_offset_mapping:   list of (char_start, char_end) per GPT-2 token
        spacy_annotations:     spaCy Doc object

    Returns:
        List of POS tag strings, one per GPT-2 token.
    """
    IGNORE = "IGNORE"

    # Build list of (char_start, char_end, pos) for each spaCy token
    spacy_spans = [(tok.idx, tok.idx + len(tok.text), tok.pos_) for tok in spacy_annotations]

    labels = []
    for char_start, char_end in gpt2_offset_mapping:
        if char_start == char_end:
            # BOS/EOS/special token
            labels.append(IGNORE)
            continue

        # Find the spaCy token with the most overlap
        best_pos = IGNORE
        best_overlap = 0
        for sp_start, sp_end, pos in spacy_spans:
            overlap = max(0, min(char_end, sp_end) - max(char_start, sp_start))
            if overlap > best_overlap:
                best_overlap = overlap
                best_pos = pos

        labels.append(best_pos)

    return labels


# ---------------------------------------------------------------------------
# Probe dataset collection
# ---------------------------------------------------------------------------

def collect_probe_dataset(
    model,
    texts: list[str],
    tokenizer,
    nlp,
    device,
    batch_size: int = 32,
) -> dict:
    """
    Collect (mlp_out, POS label) pairs per layer for linear probing.

    Args:
        model:      GPT model in eval mode
        texts:      raw text strings from val split
        tokenizer:  HuggingFace tokenizer (must support return_offsets_mapping)
        nlp:        spaCy language model
        device:     torch device
        batch_size: number of texts per forward pass

    Returns:
        {
            layer_i: {activations: ndarray[N,384], labels: ndarray[N]},
            label_names: [str],
        }
    """
    IGNORE = "IGNORE"

    # Collect all unique POS tags for label encoding (pass 0: scan)
    # We'll build label_names from data
    all_layer_acts: dict[int, list] = {}
    all_layer_labels: dict[int, list] = {}
    label_set: set[str] = set()

    model.eval()

    # First, determine n_layers and d_model from model
    n_layers = len(model.blocks)
    d_model = model.config.d_model

    for layer in range(n_layers):
        all_layer_acts[layer] = []
        all_layer_labels[layer] = []

    # Process texts in batches
    for batch_start in range(0, len(texts), batch_size):
        batch_texts = texts[batch_start: batch_start + batch_size]

        # (1) spaCy annotations
        spacy_docs = list(nlp.pipe(batch_texts))

        # (2) Tokenize with offset mapping
        encodings = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=model.config.max_seq_len + 1,
            return_offsets_mapping=True,
        )
        offset_mappings = encodings.pop("offset_mapping")  # [B, S]
        input_ids = encodings["input_ids"].to(device)      # [B, S]

        # (3) Align POS labels per text
        batch_labels_list = []
        for i, (doc, offsets) in enumerate(zip(spacy_docs, offset_mappings)):
            # offsets is [S, 2] tensor; convert to list of tuples
            offsets_list = [(int(s), int(e)) for s, e in offsets]
            labels = align_pos_to_gpt2_tokens(batch_texts[i], offsets_list, doc)
            batch_labels_list.append(labels)

        # Collect unique POS tags
        for labels in batch_labels_list:
            for lbl in labels:
                if lbl != IGNORE:
                    label_set.add(lbl)

        # (4) Forward pass with return_activations=True
        # input_ids shape: [B, full_seq]; LM needs input[:-1] -> target[1:]
        # But for probing we want activations at all positions, so use the
        # padded input directly (no target needed)
        with torch.no_grad():
            _, _, activations = model(
                input_ids[:, :-1],  # drop last token to stay within max_seq_len
                return_activations=True,
            )

        # (5) Collect (mlp_out, label) per layer, excluding IGNORE
        for layer_idx, acts_dict in activations.items():
            acts = acts_dict["mlp_out"]  # [B, S', D]  where S' = S - 1
            B, S_prime, D = acts.shape

            for b in range(B):
                text_labels = batch_labels_list[b]
                # text_labels corresponds to S tokens; activations has S-1 positions
                # (we dropped the last token). Align to S' positions.
                for s in range(min(S_prime, len(text_labels) - 1)):
                    lbl = text_labels[s]
                    if lbl == IGNORE:
                        continue
                    all_layer_acts[layer_idx].append(acts[b, s].cpu().float().numpy())
                    all_layer_labels[layer_idx].append(lbl)

    # Build integer label mapping
    label_names = sorted(label_set)
    label_to_idx = {lbl: i for i, lbl in enumerate(label_names)}

    result = {"label_names": label_names}
    for layer in range(n_layers):
        acts_list = all_layer_acts[layer]
        labels_list = all_layer_labels[layer]
        if not acts_list:
            continue
        result[layer] = {
            "activations": np.stack(acts_list, axis=0),
            "labels": np.array([label_to_idx[l] for l in labels_list], dtype=np.int32),
        }

    return result


# ---------------------------------------------------------------------------
# Linear probes
# ---------------------------------------------------------------------------

def train_linear_probes(probe_dataset: dict, test_fraction: float = 0.2) -> dict:
    """
    Train LogisticRegression probes on per-layer activations.

    Args:
        probe_dataset: output of collect_probe_dataset()
        test_fraction: fraction of data to hold out for evaluation

    Returns:
        {layer_i: {accuracy, macro_f1, per_class_f1: {tag: f1}, confusion_matrix}}
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

    label_names = probe_dataset["label_names"]
    results = {}

    for key, data in probe_dataset.items():
        if key == "label_names":
            continue
        layer_idx = int(key)
        X = data["activations"]   # [N, D]
        y = data["labels"]        # [N]

        if len(np.unique(y)) < 2:
            continue

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_fraction, random_state=42, stratify=y
        )

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        clf = LogisticRegression(
            solver="lbfgs",
            max_iter=1000,
            random_state=42,
        )
        clf.fit(X_train, y_test if len(y_train) == 0 else y_train)
        y_pred = clf.predict(X_test)

        acc = float(accuracy_score(y_test, y_pred))
        macro_f1 = float(f1_score(y_test, y_pred, average="macro", zero_division=0))

        per_class = f1_score(
            y_test, y_pred, average=None, zero_division=0,
            labels=list(range(len(label_names)))
        )
        per_class_f1 = {label_names[i]: float(per_class[i]) for i in range(len(label_names))}

        cm = confusion_matrix(y_test, y_pred, labels=list(range(len(label_names))))

        results[layer_idx] = {
            "accuracy": acc,
            "macro_f1": macro_f1,
            "per_class_f1": per_class_f1,
            "confusion_matrix": cm.tolist(),
        }

    return results
