# Experimental Suite: Training-Time Interpretability

## Overview
This document specifies 5 core experiments to test training-time interpretability approaches. Each experiment compares a baseline model against models trained with interpretability constraints. All experiments use small-scale models (~10M parameters) for feasibility.

---

## Experiment 1: Anti-Superposition Training via Orthogonality Constraints

### Goal
Test whether explicitly penalizing feature overlap during training produces more monosemantic neurons without sacrificing performance.

### Setup
- **Task**: Language modeling on TinyStories or WikiText-2
- **Model**: Small GPT-2-style transformer (6 layers, 384 hidden dim, 6 heads)
- **Dataset**: TinyStories (smaller, faster to train)

### Training Configurations
1. **Baseline**: Standard cross-entropy loss
2. **Orthogonal Features (Soft)**: Add regularization term that encourages orthogonality of neuron activation vectors
   - Loss = CE_loss + λ₁ * orthogonality_penalty
   - orthogonality_penalty = mean(|⟨aᵢ, aⱼ⟩| for all i≠j neuron pairs)
   - Try λ₁ ∈ [0.001, 0.01, 0.1]
3. **Anti-Polysemantic (Hard)**: Explicit penalty for neurons activating on diverse contexts
   - Loss = CE_loss + λ₂ * polysemanticity_score
   - polysemanticity_score = entropy of which tokens activate each neuron
   - Try λ₂ ∈ [0.001, 0.01, 0.1]

### Metrics to Track
**Performance**:
- Perplexity on validation set
- Training loss curve
- Training time/compute

**Interpretability**:
- Neuron polysemanticity score: For each neuron, compute entropy over top-activating token types
- Activation sparsity: What % of neurons activate per token
- Linear probe accuracy: Train linear probes on intermediate activations to predict syntactic/semantic features
- SAE reconstruction quality: Train SAEs on each model, compare reconstruction fidelity (L0 and L2 reconstruction loss)

**Post-training analysis**:
- Manual inspection: Sample 50 random neurons, examine top-10 activating examples, human-rate as monosemantic/polysemantic
- Feature redundancy: Measure CKA/representational similarity between layers

### Expected Outcomes
- Orthogonality constraints should reduce neuron overlap
- May see capability tax (higher perplexity)
- Should see lower polysemanticity scores

### Implementation Notes
```python
# Pseudocode for orthogonality penalty
def orthogonality_loss(activations):
    # activations: [batch, seq_len, hidden_dim]
    # Compute pairwise cosine similarities
    normalized = F.normalize(activations, dim=-1)
    similarity_matrix = torch.matmul(normalized, normalized.transpose(-2, -1))
    # Penalize off-diagonal elements
    penalty = torch.mean(torch.abs(similarity_matrix - torch.eye(hidden_dim)))
    return penalty
```

---

## Experiment 2: Sparse Architectural Constraints via L0 Regularization

### Goal
Test whether enforcing sparsity during training (instead of post-hoc) improves interpretability while maintaining performance.

### Setup
- **Task**: Same as Experiment 1 (TinyStories language modeling)
- **Model**: Same base architecture

### Training Configurations
1. **Baseline**: Standard training
2. **L1 Sparsity**: Add L1 penalty on activations
   - Loss = CE_loss + λ * ||activations||₁
   - Try λ ∈ [0.0001, 0.001, 0.01]
3. **L0 Regularization (Concrete Relaxation)**: Stochastic gates that learn which neurons to use
   - Each neuron has learnable gate
   - Loss = CE_loss + β * expected_L0_norm
   - Use concrete/Gumbel-Softmax relaxation
   - Try β ∈ [0.001, 0.01, 0.1]
4. **Top-K Activation**: Hard constraint - only top-k% of neurons can activate
   - Replace ReLU with TopK activation (differentiable approximation)
   - Try k ∈ [10%, 25%, 50%]

### Metrics to Track
Same as Experiment 1, plus:
- **Sparsity metrics**:
  - Average % of active neurons per forward pass
  - Distribution of neuron activation frequencies
  - Dead neurons: % of neurons that never activate

### Expected Outcomes
- Sparsity should be explicitly controllable
- Trade-off curve between sparsity and performance
- Question: Are sparse neurons more interpretable?

### Implementation Notes
```python
# L0 regularization with concrete relaxation
class L0Gate(nn.Module):
    def __init__(self, dim, temperature=0.1):
        super().__init__()
        self.logits = nn.Parameter(torch.randn(dim))
        self.temperature = temperature
    
    def forward(self, x):
        if self.training:
            # Concrete relaxation for differentiability
            u = torch.rand_like(self.logits)
            s = torch.sigmoid((torch.log(u) - torch.log(1-u) + self.logits) / self.temperature)
            z = s.clamp(0, 1)
        else:
            z = (self.logits > 0).float()
        return x * z
    
    def expected_l0(self):
        return torch.sigmoid(self.logits).sum()
```

---

## Experiment 3: Modular Architecture with Enforced Specialization

### Goal
Test whether forcing architectural modularity (MoE-style) improves interpretability through functional specialization.

### Setup
- **Task**: Multi-task: Combine sentiment classification (SST-2) + NLI (SNLI subset) + QA (SQuAD tiny)
- **Model**: Transformer with MoE layers

### Training Configurations
1. **Baseline**: Standard multi-task transformer
2. **Vanilla MoE**: Standard mixture of experts (top-2 routing)
3. **Specialized MoE**: Force experts to specialize on tasks
   - Add auxiliary loss: Load balancing + task-expert affinity
   - Loss = task_losses + λ₁ * load_balance + λ₂ * specialization_loss
   - specialization_loss = entropy of expert-task assignment distribution
   - Try λ₂ ∈ [0.01, 0.1, 1.0]
4. **Hard-Routed MoE**: Each task gets dedicated experts (oracle routing during training)

### Metrics to Track
**Performance**:
- Per-task accuracy/F1
- Multi-task performance vs. single-task baselines

**Interpretability/Modularity**:
- Expert utilization: Which experts activate for which tasks?
- Specialization score: Mutual information between expert activation and task type
- Cross-task interference: Does ablating expert X hurt task A more than task B?
- Router entropy: How deterministic is expert selection?

**Ablation studies**:
- Remove each expert, measure per-task performance drop
- Experts should show task-specific importance

### Expected Outcomes
- Specialized MoE should have clear expert-task associations
- May sacrifice some performance vs. baseline
- Question: Does specialization = interpretability?

---

## Experiment 4: Disentanglement Objectives (β-VAE Style)

### Goal
Test whether disentanglement objectives from representation learning improve interpretability of transformer representations.

### Setup
- **Task**: Synthetic task where ground-truth factors are known
  - Example: Generate sentences with controlled factors (tense, sentiment, topic, number)
  - "The happy cat jumps" → tense=present, sentiment=positive, topic=animal, number=singular
- **Model**: Small encoder-decoder with VAE-style bottleneck

### Training Configurations
1. **Baseline**: Standard autoencoder
2. **β-VAE**: Add KL divergence term with β > 1
   - Loss = reconstruction_loss + β * KL(q(z|x) || p(z))
   - Try β ∈ [1, 4, 10, 40]
3. **Factor-VAE**: Add discriminator to encourage factorized representations
4. **Supervised Disentanglement**: If using synthetic data, add auxiliary loss to predict known factors from latent codes

### Metrics to Track
**Performance**:
- Reconstruction quality (BLEU for text)

**Disentanglement**:
- MIG (Mutual Information Gap): Measure whether individual latent dims encode individual factors
- SAP (Separated Attribute Predictability): Train linear predictors for each factor
- Disentanglement score: Independence of latent dimensions
- Interventional robustness: Change one latent dim, does it change one factor?

**If using synthetic data**:
- Ground-truth factor recovery: Do latent dims align with known factors?
- Linear probe accuracy for each factor

### Expected Outcomes
- Higher β should increase disentanglement
- May hurt reconstruction quality
- Question: Does disentanglement in latent space transfer to interpretability of internal representations?

### Implementation Notes
```python
# β-VAE loss
def beta_vae_loss(x, x_recon, mu, logvar, beta=4.0):
    recon_loss = F.mse_loss(x_recon, x, reduction='sum')
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kl_loss
```

---

## Experiment 5: Training with SAE-Inspired Objectives

### Goal
Instead of training SAEs post-hoc, integrate SAE-like objectives directly into training.

### Setup
- **Task**: TinyStories language modeling
- **Model**: GPT-2 style with modified training

### Training Configurations
1. **Baseline**: Standard training
2. **Online SAE**: Train model and SAE jointly
   - Every N steps, update SAE on current activations
   - Add SAE reconstruction loss as auxiliary objective
   - Loss = CE_loss + λ * SAE_reconstruction_error + μ * SAE_sparsity
3. **Direct Sparsity**: Force model activations to be sparse in overcomplete basis
   - Add trainable overcomplete projection: hidden_dim → expanded_dim
   - Penalize L1 in expanded space
   - Project back: expanded_dim → hidden_dim
   - Loss = CE_loss + sparsity_loss_in_expanded_space

### Metrics to Track
Same as Experiment 1, plus:
- SAE reconstruction quality (should be better for Online SAE)
- Feature activation frequency distribution
- Computational overhead

### Expected Outcomes
- Online SAE should produce activations that are easier to interpret
- Direct sparsity might be simpler but less effective
- Question: Is this better than post-hoc SAE?

---

## Meta-Experiment: Comparative Evaluation Framework

### Goal
After running all experiments, systematically compare approaches.

### Procedure
1. **Select best hyperparameters** from each experiment based on interpretability-performance Pareto frontier
2. **Unified evaluation**:
   - Train all best configs on same seed
   - Evaluate on same test set
   - Apply same interpretability metrics to all

3. **Cross-method analysis**:
   - Plot Pareto frontiers: interpretability vs. performance
   - Correlation analysis: Which interpretability metrics correlate?
   - Human evaluation: Sample neurons from each method, blind rating

4. **Adversarial testing**:
   - Train adversarial examples for each model
   - Check if "interpretable" models are actually more robust
   - Test if models can game interpretability metrics

### Visualization Outputs
- Pareto frontier plot: perplexity vs. interpretability score
- Radar charts: multiple interpretability dimensions per method
- Heatmaps: neuron specialization patterns
- t-SNE/UMAP: feature space geometry

---

## Implementation Recommendations

### Codebase Structure
```
experiments/
├── configs/
│   ├── baseline.yaml
│   ├── orthogonal.yaml
│   ├── sparse.yaml
│   ├── moe.yaml
│   └── disentangled.yaml
├── models/
│   ├── transformer.py
│   ├── sparse_transformer.py
│   ├── moe_transformer.py
│   └── vae_transformer.py
├── training/
│   ├── losses.py (custom losses)
│   ├── trainer.py
│   └── regularizers.py
├── evaluation/
│   ├── interpretability_metrics.py
│   ├── sae_analysis.py
│   └── visualization.py
└── run_experiment.py
```

### Compute Requirements
- Each experiment: ~4-8 GPU hours on single A100
- Total for all configs: ~50-100 GPU hours
- Can parallelize across multiple GPUs

### Datasets
- **TinyStories**: ~2GB, fast iteration
- **WikiText-2**: Standard LM benchmark
- **SST-2**: ~70k sentences, sentiment
- **SNLI**: Use small subset (~50k examples)

### Suggested Tooling
- **Framework**: PyTorch + HuggingFace Transformers
- **Experiment tracking**: Weights & Biases
- **Interpretability**: TransformerLens or custom analysis tools
- **SAE training**: Use existing SAE implementations (e.g., from Anthropic's repo)

---

## Timeline Suggestion

**Week 1-2**: 
- Set up codebase infrastructure
- Implement Experiment 1 (Anti-Superposition)
- Run baseline + 3 variants

**Week 3-4**:
- Implement Experiment 2 (Sparse Constraints)
- Run all sparse variants

**Week 5**:
- Implement Experiments 3 & 4 (MoE and Disentanglement)
- These can run in parallel if you have multiple GPUs

**Week 6**:
- Implement Experiment 5 (SAE-Inspired)
- Begin Meta-Experiment analysis

**Week 7-8**:
- Complete all runs
- Comprehensive evaluation
- Generate visualizations
- Write up results

---

## Key Questions Each Experiment Answers

1. **Exp 1**: Does penalizing overlap reduce polysemanticity?
2. **Exp 2**: Is enforced sparsity better than emergent sparsity?
3. **Exp 3**: Does modularity lead to functional specialization?
4. **Exp 4**: Do representation learning techniques transfer to LLMs?
5. **Exp 5**: Is online training better than post-hoc SAE?
6. **Meta**: Which approach has best interpretability/performance trade-off?