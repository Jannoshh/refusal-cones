# Multi-Dimensional ACE Implementation Summary

## What Was Implemented

Extended the ACE (Affine Concept Editing) framework to support k-dimensional refusal subspaces with **non-orthogonal vectors**.

### Key Formula

```
h' = h - P_V(h - v⁻) + V·α
```

Where:
- `V ∈ R^{k×d}` - k arbitrary vectors (not necessarily orthogonal)
- `P_V = V(V^T V)^{-1}V^T` - Pseudoinverse projection
- `v⁻ ∈ R^d` - Harmless baseline (unique!)
- `α ∈ R^k` - Per-direction steering coefficients

## Files Modified

1. **`src/training/adapters/unified_rdo_adapter.py`**
   - Added `force_orthogonal` parameter to `UnifiedRDOConfig` (default: True)
   - Added `per_direction_steering` parameter (default: False)
   - Enhanced `RankKUnifiedLayer`:
     - Supports both orthogonal and non-orthogonal modes
     - ACE baseline restoration `v⁻`
     - Pseudoinverse projection `V(V^T V)^{-1}V^T`
     - Per-direction steering `α ∈ R^k`
   - Added methods:
     - `initialize_from_mean_diff()` - Initialize from mean difference + noise
     - `fit_baseline()` - Compute `v⁻` from data
     - `get_condition_number()` - Monitor numerical stability
   - Added model-level methods:
     - `initialize_rank_k_from_mean_diff()` - Initialize all layers
     - `get_all_condition_numbers()` - Check stability across layers

## Files Created

1. **`examples/example_multidim_ace.py`**
   - Complete walkthrough of non-orthogonal multi-dimensional ACE
   - Shows initialization, stability monitoring, training workflow

2. **`tests/test_multidim_ace.py`**
   - Unit tests for all new functionality
   - Tests orthogonal vs non-orthogonal projections
   - Tests baseline restoration, initialization, condition numbers

3. **`docs/MULTIDIMENSIONAL_ACE.md`**
   - Comprehensive documentation
   - Mathematical derivations
   - Usage examples and best practices
   - FAQ section

## Key Features

### 1. Non-Orthogonal Support

```python
config = UnifiedRDOConfig(
    enable_rank_k=True,
    rank_k=3,
    force_orthogonal=False,  # NEW: Allow non-orthogonal vectors
    use_baseline=True
)
```

### 2. Baseline Restoration (ACE)

The baseline `v⁻` is **unique** even for non-orthogonal vectors:

```python
proj_V(v⁻) = V(V^T V)^{-1}V^T v⁻  # Unique orthogonal projection
```

### 3. Initialization from Mean Difference

```python
model.initialize_rank_k_from_mean_diff(
    tokenizer,
    harmless_prompts,
    harmful_prompts,
    noise_std=0.01  # Small perturbations
)
```

This creates k vectors by adding noise to the mean difference direction - a good warm start for training.

### 4. Numerical Stability Monitoring

```python
# Check condition numbers
cond_numbers = model.get_all_condition_numbers()

if max(cond_numbers) > 100:
    print("Warning: numerical instability")
```

## Usage Example

```python
from transformers import AutoModelForCausalLM
from src.training.adapters import UnifiedRDOConfig, get_unified_rdo_model

# Configure
config = UnifiedRDOConfig(
    enable_rank_k=True,
    rank_k=3,
    force_orthogonal=False,      # Non-orthogonal
    use_baseline=True,            # ACE baseline
    per_direction_steering=True   # α ∈ R^3
)

# Load and wrap model
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B")
model = get_unified_rdo_model(model, config)

# Initialize from data
model.initialize_rank_k_from_mean_diff(
    tokenizer, harmless_prompts, harmful_prompts
)

# Monitor stability
print(f"Condition numbers: {model.get_all_condition_numbers()}")

# Train
optimizer = torch.optim.Adam(model.get_affine_parameters(), lr=1e-3)
# ... training loop
```

## Comparison: Orthogonal vs Non-Orthogonal

| Aspect | Orthogonal | Non-Orthogonal |
|--------|-----------|----------------|
| **Formula** | `P = VV^T` | `P = V(V^T V)^{-1}V^T` |
| **Complexity** | O(kd) | O(k²d + k³) |
| **Stability** | Always stable | Needs monitoring |
| **Expressiveness** | Less | More |
| **Use case** | Production | Research/Discovery |

## Answer to Your Question

> "I'm not sure about the restoration to baseline, is the solution unique?"

**Yes, it is unique!**

For non-orthogonal vectors `V = [v_1, ..., v_k]`, the baseline restoration:

```
proj_V(v⁻) = V(V^T V)^{-1}V^T v⁻
```

Is **unique** because:

1. **Geometric uniqueness**: Orthogonal projection onto a subspace always has a unique solution (closest point in the subspace)

2. **Algebraic uniqueness**: Given fixed `V` and `v⁻`, the formula is deterministic

3. The **coefficient representation** `c = (V^T V)^{-1}V^T v⁻` might not be unique (many ways to write the same point using non-orthogonal basis), but the **projection itself** `Vc` is unique

The pseudoinverse gives the minimum-norm coefficient representation, which is a well-defined unique choice.

## Testing

Run the tests with:

```bash
uv run pytest tests/test_multidim_ace.py -v
```

Tests cover:
- Orthogonal projection correctness
- Pseudoinverse projection properties (idempotency, span preservation)
- Baseline restoration
- Initialization from mean difference
- Condition number computation
- Forward pass with both modes

## Next Steps

1. **Try it out**: Run `examples/example_multidim_ace.py`
2. **Train**: Use non-orthogonal mode for discovery, orthogonal for production
3. **Monitor**: Check condition numbers during training
4. **Experiment**: Compare orthogonal vs non-orthogonal performance

## References

- See `docs/MULTIDIMENSIONAL_ACE.md` for full mathematical details
- See `examples/example_multidim_ace.py` for usage patterns
