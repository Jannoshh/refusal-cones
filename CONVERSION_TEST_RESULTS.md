# Conversion Equivalence Tests

## Overview

Tests to verify that **hooks** and **weight modifications** produce equivalent results.

## What the Tests Verify

### Test 1: Output Equivalence

**Objective**: Verify both methods produce identical outputs.

**Setup**:
- Simple linear layer: `y = Wx + b`
- Random input `x` and ablation vector `v`

**Method 1 (Hooks)**:
```python
# Register hook that projects out v
handle = layer.register_forward_hook(ablation_hook(v))
out1 = layer(x)
handle.remove()
```

**Method 2 (Weight Modification)**:
```python
# Modify weights in forward pass (differentiable!)
P = I - v @ v^T  # Projection matrix
W_modified = P @ W  # Differentiable w.r.t. v!
out2 = F.linear(x, W_modified, b)
```

**Expected**: `||out1 - out2|| < 1e-5`

**Mathematical Proof**:
```
Method 1: out1 = Wx + b - ((Wx + b) · v)v
Method 2: out2 = (P @ W)x + b = P(Wx + b) = (Wx + b) - ((Wx + b) · v)v
Therefore: out1 = out2  ✓
```

---

### Test 2: Gradient Flow

**Objective**: Verify gradients flow correctly through both methods.

**Setup**:
- Vector `v` is trainable parameter
- Model weights are frozen
- Compute loss and backprop

**Method 1 (Hooks)**:
```python
v = torch.randn(dim, requires_grad=True)
out = forward_with_hooks(model, x, v)
loss = out.pow(2).mean()
loss.backward()
grad1 = v.grad  # Should have gradient!
```

**Method 2 (Weight Modification)**:
```python
v = torch.randn(dim, requires_grad=True)
P = I - torch.outer(v, v)  # Differentiable!
out = (P @ W) @ x
loss = out.pow(2).mean()
loss.backward()
grad2 = v.grad  # Should have gradient!
```

**Expected**:
- Both `v.grad` should exist
- `||grad1 - grad2|| < 1e-4`

**Why This Works**:

PyTorch automatically computes:
```python
∂L/∂v = ∂L/∂P · ∂P/∂v
where ∂P/∂v = -∂(v ⊗ v)/∂v  # Differentiable!
```

This is why **LoRA works** - weight modifications can be differentiable!

---

### Test 3: Training Equivalence

**Objective**: Verify both methods converge to same solution during training.

**Setup**:
- Two identical models with frozen weights
- Two identical vector parameters
- Same optimizer (SGD with lr=0.01)
- Train for 10 steps

**Training Loop**:
```python
for step in range(10):
    # Method 1
    out1 = forward_with_hooks(model1, x, v1)
    loss1 = MSE(out1, target)
    loss1.backward()
    optimizer1.step()

    # Method 2
    out2 = forward_with_weight_mod(model2, x, v2)
    loss2 = MSE(out2, target)
    loss2.backward()
    optimizer2.step()
```

**Expected**:
- Loss trajectories identical: `|loss1[i] - loss2[i]| < 1e-4` for all i
- Final vectors identical: `||v1 - v2|| < 1e-4`

**Result**: Both methods can be used for training!

---

### Test 4: Real Transformer Layer

**Objective**: Verify equivalence on actual transformer architecture.

**Setup**:
- Load GPT-2 or similar
- Extract one transformer layer
- Apply vector ablation via both methods

**Why This Matters**:
- Real layers have complex structure (attention, MLP, residuals)
- Outputs are tuples (activations, cache, etc.)
- Tests practical applicability

**Expected**: Same output equivalence as simple linear layer.

---

### Test 5: Backward Compatibility

**Objective**: Verify conversion utilities work correctly.

**Tests**:
- `VectorModifiedLayer` wrapper works
- Can extract trainable parameters
- Compatible with `PerLayerRefusalVectors`
- Can convert between approaches

---

## Running the Tests

```bash
# Run all tests
python test_conversion_equivalence.py

# Expected output:
# ✓ PASS: Output Equivalence
# ✓ PASS: Gradient Flow
# ✓ PASS: Training Equivalence
# ✓ PASS: Real Transformer Layer
# ✓ PASS: Backward Compatibility
#
# Total: 5/5 tests passed
# ✓ All tests passed! Both approaches are equivalent.
```

## Key Insights

### 1. Weight Modifications ARE Differentiable

**You were right!** Like LoRA, weight modifications can be part of the computation graph:

```python
# This is differentiable w.r.t. v!
P = I - torch.outer(v, v)
W_modified = W @ P
output = W_modified @ x
```

Gradients flow through `torch.outer()` automatically.

### 2. Both Approaches Work for Training

- **Hooks**: Simpler, non-destructive
- **Weight mods**: Also work, more LoRA-like

Both are valid for RL training!

### 3. Choice Depends on Use Case

| Use Case | Recommended Approach | Reason |
|----------|---------------------|--------|
| **RL Training** | Either (hooks are simpler) | Both have correct gradients |
| **Testing** | Hooks (easier swap) | Non-destructive |
| **Deployment** | Weight mods (permanent) | No runtime hooks |
| **Analysis** | Hooks (reversible) | Easy to enable/disable |

### 4. Hybrid Strategy

**Best of both worlds**:
1. **Train with hooks** (simpler code, non-destructive)
2. **Deploy with weight mods** (bake vectors into model)

```python
# After training
trained_vectors = trainer.get_vectors()

# Convert for deployment
from conversion_utils import apply_vectors_to_weights_permanent
apply_vectors_to_weights_permanent(model, trained_vectors)

# Save standalone model
model.save_pretrained('steered_model')
```

## Conclusion

**Both approaches are mathematically equivalent and support gradient flow.**

The original analysis had an error - weight modifications DO work for training when done as part of the forward pass (like LoRA).

**Recommendation**:
- Keep current hook-based training (simpler)
- Add weight conversion for deployment (optional)
- Both approaches are valid - choose based on preference

The tests verify this equivalence across outputs, gradients, and training convergence.
