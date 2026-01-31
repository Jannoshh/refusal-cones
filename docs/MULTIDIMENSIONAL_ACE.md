# Multi-Dimensional ACE (Affine Concept Editing)

**Extension of ACE to k-dimensional refusal subspaces with non-orthogonal vectors**

## Overview

This document describes the generalized multi-dimensional ACE formula that supports arbitrary (non-orthogonal) subspace vectors.

## Mathematical Formulation

### Single-Direction ACE (Original)

```
h' = h - proj_v(h - v⁻) + α·v
```

Where:
- `v ∈ R^d` - Single refusal direction
- `v⁻ ∈ R^d` - Harmless baseline (mean harmless activations)
- `α ∈ R` - Steering coefficient (scalar)
- `proj_v(x) = (x·v̂)v̂` - Orthogonal projection onto v

### Multi-Dimensional ACE (Generalized)

```
h' = h - P_V(h - v⁻) + V·α
```

Where:
- `V ∈ R^{k×d}` - k refusal directions (arbitrary, not necessarily orthogonal)
- `v⁻ ∈ R^d` - Harmless baseline (same for all directions)
- `α ∈ R^k` - Per-direction steering coefficients (vector)
- `P_V` - Projection operator (see below)

### Projection Operators

#### Orthogonal Case (`force_orthogonal=True`)

When `V` is orthonormalized via Gram-Schmidt:

```
P_V(h) = VV^T h
```

- Complexity: O(kd)
- Always stable (condition number = 1)
- Less expressive (forces independence)

#### Non-Orthogonal Case (`force_orthogonal=False`)

When `V` is arbitrary (possibly correlated):

```
P_V(h) = V(V^T V)^{-1}V^T h
```

Where `(V^T V)^{-1}` is the inverse Gram matrix (k×k).

- Complexity: O(k²d + k³)
- Requires stability monitoring (condition number)
- More expressive (preserves discovered correlations)

## Uniqueness Properties

### Question: Is the baseline restoration unique for non-orthogonal vectors?

**Answer: YES**

The projection `P_V(v⁻)` is unique because:

1. **Geometric uniqueness**: Orthogonal projection onto a subspace has a unique solution (the closest point in the subspace)
2. **Algebraic uniqueness**: Given fixed `V` and `v⁻`, the formula `V(V^T V)^{-1}V^T v⁻` is deterministic
3. **Pseudoinverse property**: The Moore-Penrose pseudoinverse gives the minimum-norm solution

### What's NOT unique?

The **coefficient representation** is not unique for non-orthogonal vectors:

```python
# Same geometric projection, different coefficients
point = 2.0*v_1 + 3.0*v_2
point = 1.5*v_1 + 3.5*v_2  # Different, if v_1, v_2 not orthogonal

# The pseudoinverse chooses the minimum-norm coefficients
coeffs = (V^T V)^{-1} V^T · point  # Unique choice
```

But this doesn't affect the **geometric projection**, which is what matters for ACE.

## Implementation

### Configuration

```python
from src.training.adapters.unified_rdo_adapter import UnifiedRDOConfig

config = UnifiedRDOConfig(
    enable_rank_k=True,
    rank_k=3,                         # 3-dimensional subspace
    force_orthogonal=False,            # Allow non-orthogonal vectors
    use_baseline=True,                 # Use ACE baseline restoration
    per_direction_steering=True,       # α ∈ R^k (vector)
    projection_alpha=1.0,              # β for ablation
    addition_alpha=1.0                 # Scaling for steering
)
```

### Usage

```python
from transformers import AutoModelForCausalLM
from src.training.adapters import get_unified_rdo_model

# Load model
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B")

# Wrap with multi-dimensional ACE
model = get_unified_rdo_model(model, config)

# Initialize from mean difference
model.initialize_rank_k_from_mean_diff(
    tokenizer=tokenizer,
    harmless_prompts=harmless_examples,
    harmful_prompts=harmful_examples,
    noise_std=0.01  # Small perturbations around mean diff
)

# Monitor numerical stability
cond_numbers = model.get_all_condition_numbers()
print(f"Max condition number: {max(cond_numbers):.2f}")
```

### Layer-Level API

```python
from src.training.adapters.unified_rdo_adapter import RankKUnifiedLayer

layer = RankKUnifiedLayer(
    base_layer=transformer_layer,
    dim=1024,
    rank_k=3,
    force_orthogonal=False,
    use_baseline=True,
    per_direction_steering=True
)

# Initialize from data
layer.initialize_from_mean_diff(
    harmful_activations,  # [n_samples, dim]
    harmless_activations, # [n_samples, dim]
    noise_std=0.01
)

# Check stability
cond_number = layer.get_condition_number()
print(f"Condition number: {cond_number:.2f}")
```

## Numerical Stability

### Monitoring

The condition number of the Gram matrix `V^T V` indicates numerical stability:

```python
# Condition number = λ_max / λ_min
cond = layer.get_condition_number()

if cond < 10:
    # Excellent stability
elif cond < 100:
    # Good stability
else:
    # Warning: near-singular, consider orthogonalization
```

### Guidelines

| Condition Number | Interpretation | Action |
|-----------------|----------------|--------|
| < 10 | Excellent | No action needed |
| 10-100 | Good | Monitor during training |
| 100-1000 | Moderate | Consider reducing rank_k |
| > 1000 | Poor | Use force_orthogonal=True |

### Regularization

The pseudoinverse projection includes automatic regularization:

```python
G_reg = G + eps * I  # eps = 1e-6
```

This prevents numerical issues even when vectors are nearly collinear.

## When to Use Non-Orthogonal?

### Use Non-Orthogonal When:

1. **Preserving discovered structure**
   - Refusal directions found via gradient-based discovery may be naturally correlated
   - Orthogonalization destroys this discovered structure

2. **Exploring correlated concepts**
   - Related safety concepts (e.g., violence + harm) may share components
   - Non-orthogonal vectors can represent these correlations

3. **Maximum expressiveness**
   - Training can optimize freely without orthogonality constraints
   - May find better solutions in high-dimensional activation space

### Use Orthogonal When:

1. **Computational efficiency**
   - O(kd) vs O(k²d + k³)
   - Matters for large models or frequent inference

2. **Numerical stability guaranteed**
   - No condition number monitoring needed
   - Robust to all inputs

3. **Interpretability**
   - Independent directions easier to analyze
   - Clear separation of concepts

## Comparison Table

| Feature | Orthogonal | Non-Orthogonal |
|---------|-----------|----------------|
| **Projection formula** | `VV^T` | `V(V^T V)^{-1}V^T` |
| **Complexity** | O(kd) | O(k²d + k³) |
| **Uniqueness** | ✓ Projection unique<br>✓ Coefficients unique | ✓ Projection unique<br>✗ Coefficients non-unique |
| **Stability** | Always stable | Needs monitoring |
| **Expressiveness** | Less expressive | More expressive |
| **Interpretability** | Clear (independent) | Harder (correlated) |
| **Best for** | Production, efficiency | Research, discovery |

## Advanced Features

### Per-Direction Steering

```python
# Scalar α (same for all directions)
per_direction_steering=False
alpha ∈ R  # Single value

# Vector α (independent control)
per_direction_steering=True
alpha ∈ R^k  # One per direction
```

Per-direction steering allows fine-grained control:

```python
# Example: steer away from violence (α_1) but towards helpfulness (α_2)
alpha = torch.tensor([-1.0, 2.0, 0.0])  # k=3
```

### Baseline Fitting

The baseline `v⁻` is computed from data, not trained:

```python
# Automatically done in initialize_from_mean_diff()
layer.fit_baseline(harmless_activations)

# Or manually
layer.v_minus = harmless_activations.mean(dim=0)
layer._baseline_fitted = True
```

### Projection Matrix Extraction

Get the full projection matrix for analysis:

```python
P = layer.get_projection_matrix()  # [dim, dim]

# Verify properties
assert torch.allclose(P @ P, P)  # Idempotent: P² = P

eigenvalues = torch.linalg.eigvalsh(P)
rank = (eigenvalues > 1e-6).sum()  # Should equal rank_k
```

## Examples

See:
- `examples/example_multidim_ace.py` - Full walkthrough with explanations
- `tests/test_multidim_ace.py` - Unit tests for all functionality

## References

1. **ACE (Affine Concept Editing)**
   - [Refusal in LLMs is an Affine Function](https://arxiv.org/abs/2411.09003v3)
   - Marshall et al., 2024

2. **Mean Difference Method**
   - [Refusal in Language Models Is Mediated by a Single Direction](https://arxiv.org/abs/2406.11717)
   - Arditi et al., 2024

3. **Pseudoinverse Projection**
   - Moore-Penrose pseudoinverse: `V⁺ = (V^T V)^{-1}V^T`
   - Gives minimum-norm solution for underdetermined systems

## FAQ

### Q: Why use pseudoinverse instead of just averaging?

**A:** Averaging doesn't preserve the geometric projection. The pseudoinverse gives the unique orthogonal projection onto the subspace.

### Q: What if vectors become nearly collinear during training?

**A:** The condition number will increase. Monitor it and either:
1. Add regularization to loss: `loss += λ * condition_number`
2. Switch to orthogonal mode
3. Reduce rank_k

### Q: Can I mix orthogonal and non-orthogonal layers?

**A:** Yes! Set `force_orthogonal` per layer in a custom model. For example:
- Early layers: orthogonal (efficiency)
- Middle layers: non-orthogonal (expressiveness)
- Late layers: orthogonal (stability)

### Q: Does this work with other PEFT methods (LoRA, etc.)?

**A:** Yes, the RDO adapters are complementary. You can combine:
```python
# Apply LoRA
model = peft.get_peft_model(model, lora_config)

# Then add RDO on top
model = get_unified_rdo_model(model, rdo_config)
```

---

**Implementation Status:** ✅ Complete (January 2025)

**Tested on:**
- Qwen/Qwen3-0.6B
- gemma-2-2b (expected to work)
- Llama-3.x (expected to work)
