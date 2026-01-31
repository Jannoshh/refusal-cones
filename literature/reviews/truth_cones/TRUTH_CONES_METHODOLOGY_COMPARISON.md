# Methodology Comparison: Refusal Cones vs. Truth Cones

## Overview

Both papers use nearly identical methodology with domain-specific adaptations.

---

## 1. Initial Direction Discovery

### Refusal (Wollschläger 2502.17420)
```
Method: Difference-in-means (DIM)
v_init = mean(activations_harmful) - mean(activations_harmless)
Purpose: Find single "refusal direction" through linear probe
Domain: Safety-critical behavior (refuse harmful requests)
```

### Truth (Yu et al. 2505.21800)
```
Method: Linear probe / Difference-in-means (DIM)
v_init = mean(activations_true) - mean(activations_false)
Purpose: Find single "truth direction" through linear classification
Domain: Factuality behavior (distinguish true/false statements)
```

### Similarity
- Both use linear probe as baseline
- Both normalize by supervision signal (explicit labels)
- Both use middle-to-late layers (60-75% depth)

---

## 2. Multi-Dimensional Extension: Concept Cones

### Common Framework

**Definition (Identical Structure):**

```
Given orthonormal basis vectors V = [v₁, v₂, ..., vₖ],
the concept cone is:

C = {Σᵢ λᵢ·vᵢ | λᵢ ≥ 0} \ {0}

All directions within C exhibit the target property.
```

### Properties to Satisfy

| Property | Definition | Refusal Example | Truth Example |
|----------|-----------|-----------------|---------------|
| **Monotonic Scaling** | Effect scales with α | α ↑ → more compliant | α ↑ → more truthful |
| **Surgical Ablation** | v can be surgically removed | Projection removes refusal signal | Projection removes truth signal |
| **Positivity Constraint** | λᵢ ≥ 0 (conic structure) | All basis vectors strengthen refusal | All basis vectors strengthen truth |

---

## 3. Three-Term Loss Optimization

### Refusal (Wollschläger 2502.17420)

```
L_total = λ₁·L_add + λ₂·L_ablate + λ₃·L_retain

L_add: Force HARMFUL completion when v added
      (want: high perplexity on harmful requests with jailbreak)

L_ablate: Force SAFETY when v ablated
         (want: low perplexity on harmful requests without vector)

L_retain: Preserve helpfulness on benign requests
         (want: minimal divergence from base model on helpful prompts)
```

### Truth (Yu et al. 2505.21800)

```
L_total = λ₁·L_add + λ₂·L_ablate + λ₃·L_retain

L_add = -(1/|D_false|) Σ log ŷ_add(x + v)  [target: y = Yes/1]
        Force TRUE answer on false statements when v added

L_ablate = -(1/|D_true|) Σ log(1 - ŷ_ablate(x - vv^T·x))  [target: y = No/0]
           Force FALSE answer on true statements when v ablated

L_retain = D_KL[p₀(y | x) || p_v(y | x)]
           Preserve unrelated model behavior on Alpaca instructions
```

### Key Differences

| Component | Refusal | Truth |
|-----------|---------|-------|
| **L_add objective** | Maximize harm (perplexity) | Force "Yes" answer (binary CE) |
| **L_ablate objective** | Preserve safety (low perplexity) | Force "No" answer (binary CE) |
| **Generation scope** | Multi-token continuations | Restrict to {Yes, No} logits |
| **Retention data** | Generic instructions | 30-token Alpaca continuations |
| **Loss type** | Perplexity-based | Cross-entropy based |

**Why the Difference:** Refusal is continuous (harm level), while truth is binary (correct/incorrect answer).

---

## 4. Causal Interventions

### Activation Addition

**Refusal:**
```python
x'[l] = x[l] + α·v[l]

On harmful prompts, adding v should:
- Reduce refusal (α ↑ → more jailbroken)
- Maximize harmful completions
```

**Truth:**
```python
x'[l] = x[l] + α·v[l]

On false statements, adding v should:
- Increase truthfulness perception (α ↑ → more "Yes" answers)
- Make false statement appear true to model
```

**Causal Test:** Does magnitude of α correlate with effect strength?

### Directional Ablation

**Refusal:**
```python
x̃[l] = x[l] - v[l]·v[l]^T·x[l]

Removes refusal signal from activation space.
On harmful prompts, model should refuse (high safety).
Applied globally: all layers, all tokens.
```

**Truth:**
```python
x̃[l] = x[l] - v[l]·v[l]^T·x[l]

Removes truth signal from activation space.
On true statements, model should answer "No" (false).
Applied globally: all layers, all tokens.
```

**Causal Test:** Does ablation flip binary answer? (Yes → No, harmful → refuse)

---

## 5. Evaluation Metrics

### Refusal Domain

| Metric | Definition | Interpretation |
|--------|-----------|-----------------|
| **ASR** | Compliance rate after ablation | Higher ASR = v captures refusal |
| **KL Divergence** | Divergence on benign prompts | Lower KL = surgical precision |
| **Perplexity** | Completion quality on harmful prompts | Lower PPL with jailbreak = effective |

### Truth Domain

| Metric | Definition | Interpretation |
|--------|-----------|-----------------|
| **ASR** | Answer switching rate (Yes → No when ablated) | Higher ASR = v captures truth |
| **KL Divergence** | Divergence on Alpaca prompts | Lower KL = surgical precision |
| **Accuracy** | Correct answer on test statements | Higher accuracy with v added = effective |

### Direct Mapping

```
Refusal ASR     ↔  Truth ASR
(jailbreak rate)   (answer switching rate)

Refusal KL      ↔  Truth KL
(on benign)        (on Alpaca)

Refusal PPL     ↔  Truth CE Loss
(on harmful)       (on {true, false})
```

---

## 6. Monte Carlo Evaluation of Cone Space

### Refusal (Implicit in Wollschläger)
```python
# Sample directions from cone interior
for trial in range(N_trials):
    λ = sample_nonnegative_uniform(k)
    v_sample = Σᵢ λᵢ·bᵢ / ||...||

    # Test if v_sample satisfies refusal property
    test_causal_interventions(v_sample)
```

### Truth (Explicit in Yu et al.)
```python
# Section 3.3: "Monte-Carlo sampling for Testing"
# Sample 64 directions uniformly from cone

for trial in range(64):
    λ = [λ₁, λ₂, ..., λₖ] ~ Uniform[0,1]^k (normalized)
    v_sample = Σᵢ λᵢ·bᵢ

    # Measure ASR on all prompts
    ASR_sample = compute_answer_switching_rate(v_sample)
```

**Purpose:** Validate that **entire cone**, not just basis vectors, mediates the property.

---

## 7. Retention / Fidelity Mechanism

### Refusal
```
L_retain = D_KL[p_base(tokens | generic_prompts) ||
                p_jailbreak(tokens | generic_prompts)]

Goal: Ensure refusal vector doesn't break helpful behavior
Data: Generic instruction-following prompts
Scope: Full continuation (allows more variance)
Threshold: KL < 0.1 (Arditi et al. standard)
```

### Truth
```
L_retain = D_KL[p_base(y₁:₃₀ | alpaca_prompt) ||
                p_truth(y₁:₃₀ | alpaca_prompt)]

Goal: Ensure truth vector doesn't break general capabilities
Data: Alpaca instruction-following dataset
Scope: 30-token continuations (wider scope than single-token)
Threshold: KL < 0.1 (inherited from refusal papers)
```

**Key Difference:** Truth uses wider scope (30 tokens vs. unspecified) to ensure robustness.

---

## 8. Search Algorithm: Gradient-Based with GP Prior

### Both Papers Use Similar Structure

**Refusal (Wollschläger 2502.17420):**
```
1. Initialize with v_init (DIM direction)
2. Perform gradient ascent on refusal property
3. Use Gaussian Process to model landscape
4. Combine local (gradient) + global (GP) exploration
5. Identify multiple modes (multi-dimensional cone basis)
```

**Truth (Yu et al. 2505.21800):**
```
1. Initialize with v_init (DIM/linear probe)
2. Optimize three-term loss with gradient descent
3. Search orthogonal space for additional basis vectors
4. Ensure orthonormality constraint (QR factorization)
5. Identify k basis vectors spanning truth cone
```

**Conceptual Alignment:** Both use gradient information + prior to efficiently discover multi-dimensional structure.

---

## 9. Discovered Dimensionality

### Refusal
- **Typical range:** 2-5 dimensions
- **Depends on:** Model size, architecture, layer choice
- **Scale pattern:** Larger models → higher-dimensional cones

### Truth
- **Typical range:** 2-5 dimensions
- **Depends on:** Model size (scale-dependent)
- **Observed pattern:**
  - Qwen-7B: up to 5D with 100% ASR
  - Gemma-9B: up to 5D with 97.3% ASR
  - Qwen-3B: drops at 3D+ (45.1% at 2D)
  - Gemma-2B: drops at 3D+ (43.1% at 4D)

### Interpretation
Both domains show:
- **Not 1D:** Single linear direction insufficient
- **Not arbitrary:** Plateaus above 4-5D
- **Scale-dependent:** Larger models support higher dimensions
- **Efficiency:** Most utility captured in 2-3D

---

## 10. Layer Localization

### Refusal (Implicit in Wollschläger)
```
Effective layers: Middle to late layers
Not specified exactly, but uses layer-specific optimization
```

### Truth (Explicit in Yu et al. - Experiment 1)

```
PEAK effectiveness: 60-75% normalized layer depth

Pattern:
├─ Early layers (0-60%): Weak ASR
├─ Middle layers (60-75%): Sharp ASR increase → PEAK
├─ Late layers (75-100%): Sharp ASR decrease
└─ Final token position: Strongest interventions

Interpretation: High-level semantic properties
               concentrate in middle-to-late layers
```

**Why This Matters:**
- Reduces search space (only probe 60-75% layers)
- Consistent with "feature accumulation" in residual stream
- Same pattern in both domains suggests universal principle

---

## 11. Framework Boundary & Limitations

### What Works (Supported by Experiments)

| Property | Refusal | Truth | Evidence |
|----------|---------|-------|----------|
| Boolean/Binary concepts | ✓ | ✓ | Both show clear cones |
| Multi-dimensional | ✓ | ✓ | 5D cones in both |
| Surgical intervention | ✓ | ✓ | KL < 0.05 in both |
| Layer-localized | ✓ | ✓ | 60-75% depth |
| Model-generalizable | ✓ | ✓ | Multiple models tested |

### What Doesn't Work (Negative Results)

| Property | Status | Reason |
|----------|--------|--------|
| Sentiment | ✗ (Truth) | Gradient property, not boolean |
| Toxicity | ✗ (Truth) | No valid baseline direction |
| Bias | ? | Not tested |

**Hypothesis:** Concept cones generalize to **discrete/boolean properties** but NOT **gradient properties** (semantic spectrum).

---

## 12. Comparison Matrix: Key Metrics

| Aspect | Refusal | Truth | Correspondence |
|--------|---------|-------|-----------------|
| **Framework** | Concept cones (Wollschläger) | Concept cones (Yu et al. / Wollschläger) | 100% |
| **Init direction** | DIM (harmful vs. harmless) | DIM (true vs. false) | Analogous |
| **Search method** | Gradient-based + GP | Gradient-based optimization | Similar |
| **Dimensionality** | 2-5D | 2-5D | Same range |
| **Layer depth** | Implicit | 60-75% | Validated in truth |
| **KL threshold** | 0.1 | 0.1 | Inherited |
| **ASR on 1D** | ~100% (jailbreak rate) | 100% (answer switching) | High fidelity |
| **ASR on 5D** | 90%+ | 97%+ (Gemma-9B) | Maintained |
| **Basis orthogonality** | Required | Required + validated | Structural requirement |

---

## 13. Implementation Template for Truth Cones

### For Your Project

```python
# Adapted from refusal discovery code
# File: src/discovery/truth_discovery.py

class TruthConeDiscovery(GradientGeometryDiscovery):
    """Extends refusal cone framework to truth domain."""

    def measure_truth_with_grad(self, v: Tensor, layer: int, token_pos: int):
        """
        Measure truth strength when steering with vector v.

        Args:
            v: Direction vector [hidden_dim]
            layer: Which layer to intervene
            token_pos: Token position (typically -1 for last token)

        Returns:
            R: Truth rate (0-1, higher = more truthful)
            grad: Gradient of R w.r.t. v
        """
        v.requires_grad = True

        # Get true/false statement activations
        true_acts = self.get_activations(self.true_statements, layer, token_pos)
        false_acts = self.get_activations(self.false_statements, layer, token_pos)

        # Compute L_add: add v on false statements → force "Yes"
        x_added = false_acts + v.unsqueeze(0)  # [batch, hidden_dim]
        logits_added = self.model.unembed(x_added)
        logits_restricted = logits_added[:, [YES_TOKEN, NO_TOKEN]]
        L_add = self.binary_ce_loss(logits_restricted, target=1)

        # Compute L_ablate: ablate v on true statements → force "No"
        x_ablated = true_acts - torch.einsum('...d,d->...', true_acts, v) * v
        logits_ablated = self.model.unembed(x_ablated)
        logits_restricted = logits_ablated[:, [YES_TOKEN, NO_TOKEN]]
        L_ablate = self.binary_ce_loss(logits_restricted, target=0)

        # Compute L_retain: preserve Alpaca performance
        alpaca_acts = self.get_activations(self.alpaca_prompts, layer, token_pos)
        x_alpaca_base = alpaca_acts
        x_alpaca_ablated = alpaca_acts - torch.einsum('...d,d->...', alpaca_acts, v) * v
        L_retain = self.kl_divergence(x_alpaca_base, x_alpaca_ablated)

        # Multi-objective optimization
        loss = 1.0 * L_add + 1.0 * L_ablate + 0.5 * L_retain
        R = (-loss).item()  # Convert loss to rate

        loss.backward()
        return R, v.grad.clone()

    def discover_truth_cones(self, n_dimensions=5):
        """Discover k-dimensional truth cone."""
        basis_vectors = []

        for k in range(1, n_dimensions + 1):
            # Initialize orthogonal to previous basis
            v_init = self.get_init_direction()
            for prev_v in basis_vectors:
                # Gram-Schmidt orthogonalization
                v_init = v_init - torch.dot(v_init, prev_v) * prev_v
            v_init = v_init / v_init.norm()

            # Run discovery (inherited from base class)
            results = self.discover(
                v_init=v_init,
                n_steps=50,
                layer_range=(0.6, 0.75),  # 60-75% depth
                token_pos=-1  # Last token
            )

            basis_vectors.append(results['best_vector'])

        return {
            'basis': torch.stack(basis_vectors),
            'dimensionality': len(basis_vectors),
            'asr': self.evaluate_cone_asr(basis_vectors)
        }
```

---

## Summary

**Yu et al. (2505.21800) is a direct application of Wollschläger et al.'s concept cone framework to truth, with:**

1. ✓ Same three-term loss structure
2. ✓ Same causal intervention methodology (add/ablate)
3. ✓ Same Monte Carlo evaluation strategy
4. ✓ Same multi-dimensional discovery process
5. ✗ Different output space (binary vs. continuous)
6. ✗ Different dataset (true/false vs. harmful/harmless)
7. ✗ Explicit layer localization (60-75% depth)

**Key Innovation:** Not a new method, but **validation that concept cones are a general framework** applicable across safety-critical properties.

**Implication for Your Project:** Your refusal cone work is strengthened by this paper, as it demonstrates the framework generalizes beyond refusal to other domains.
