# Weight Orthogonalization for Refusal Ablation

## Paper Reference

From the original paper (§2.4):

> We can equivalently implement [directional ablation] by directly modifying component weights to never write to the r̂ direction in the first place. Specifically, we can take each matrix W_out ∈ R^(d_model×d_input) that writes to the residual stream, and orthogonalize its column vectors with respect to r̂:
>
> **W'_out ← W_out − r̂r̂ᵀW_out** (Equation 5)
>
> In a transformer architecture, the matrices that write to the residual stream are: the embedding matrix, the positional embedding matrix, attention out matrices, and MLP out matrices. Orthogonalizing all of these matrices, as well as any output biases, with respect to the direction r̂ effectively prevents the model from ever writing r̂ to its residual stream.

## Mathematical Derivation

### Hook-Based Ablation (Runtime)

```python
def hook(module, input, output):
    # h' = h - (h · v̂)v̂ = (I - v̂v̂ᵀ)h
    v_norm = v / (v.norm() + 1e-8)
    projection = torch.einsum('...d,d->...', output, v_norm)
    return output - torch.einsum('...,d->...d', projection, v_norm)
```

For output `h = W @ x`:
```
h' = (I - v̂v̂ᵀ)h = (I - v̂v̂ᵀ)(W @ x) = ((I - v̂v̂ᵀ)W) @ x
```

### Weight Orthogonalization (Equivalent)

The equivalent weight modification is **LEFT multiplication**:

```
W' = (I - v̂v̂ᵀ)W = P @ W

where P = I - v̂v̂ᵀ is the projection matrix
```

**This matches the paper:** `W'_out ← W_out − r̂r̂ᵀW_out = (I - r̂r̂ᵀ)W_out`

### Our Implementation (conversion_utils.py)

```python
def apply_ablation_to_weights(weight, v, in_place=False):
    P = get_projection_matrix(v)  # P = I - vvᵀ

    # LEFT multiplication: W' = P @ W
    W_modified = P @ weight  # ✅ Matches paper!

    return W_modified
```

## Matrices to Orthogonalize

Per the paper, ALL matrices that write to the residual stream must be orthogonalized:

| Component | Matrix | Shape | Notes |
|-----------|--------|-------|-------|
| **Token Embedding** | `embed_tokens.weight` | [vocab, d_model] | Initial embeddings |
| **Positional Embedding** | Varies by model | [max_pos, d_model] | If absolute positions |
| **Attention Output** | `self_attn.o_proj.weight` | [d_model, d_model] | Per layer |
| **MLP Output** | `mlp.down_proj.weight` | [d_model, d_intermediate] | Per layer |
| **Output Biases** | Various `.bias` | [d_model] | If present |

### Current Implementation Status

Our `apply_vectors_to_weights_permanent()` currently only modifies:
- ✅ `self_attn.o_proj.weight` (attention out)

To fully match the paper, we should also modify:
- ⚠️ `mlp.down_proj.weight` (MLP out)
- ⚠️ `embed_tokens.weight` (embedding)
- ⚠️ Output biases

## Three Approaches

### 1. Runtime Hooks (Training & Development)

```python
# Apply at runtime - non-destructive
handles = []
for layer in model.model.layers:
    handle = layer.register_forward_hook(ablation_hook(v))
    handles.append(handle)

output = model.generate(prompts)

for handle in handles:
    handle.remove()
```

**Pros:** Simple, non-destructive, easy to swap vectors
**Cons:** Runtime overhead, requires custom forward pass
**Use for:** Training, testing, development

### 2. LoRA-Style Wrappers (Training)

```python
class VectorModifiedLayer(nn.Module):
    def __init__(self, base_layer, vector):
        self.base_layer = base_layer
        self.vector = nn.Parameter(vector)  # Trainable!

    def forward(self, x):
        out = self.base_layer(x)
        # Differentiable projection
        v = self.vector / (self.vector.norm() + 1e-8)
        projection = torch.einsum('...d,d->...', out, v)
        return out - torch.einsum('...,d->...d', projection, v)
```

**Pros:** Differentiable, vectors as nn.Parameters, PEFT-compatible
**Cons:** More complex layer wrapping
**Use for:** Training with PEFT integration

### 3. Permanent Weight Modification (Deployment)

```python
def apply_vectors_to_weights_permanent(model, vectors):
    for idx, layer in enumerate(model.model.layers):
        v = vectors[idx]
        P = torch.eye(len(v)) - torch.outer(v, v)

        # Orthogonalize attention out: W' = P @ W
        layer.self_attn.o_proj.weight.data = (
            P @ layer.self_attn.o_proj.weight.data
        )

        # Orthogonalize MLP out: W' = P @ W
        layer.mlp.down_proj.weight.data = (
            P @ layer.mlp.down_proj.weight.data
        )
```

**Pros:** No runtime overhead, standard inference, baked into model
**Cons:** Destructive, can't remove vectors
**Use for:** Deployment after training

## Comparison Table

| Aspect | Hooks | LoRA-Style | Permanent |
|--------|-------|------------|-----------|
| **Training** | ✅ Yes | ✅ Yes | ❌ No |
| **Gradient flow** | ✅ Correct | ✅ Correct | N/A |
| **Model modification** | None | None | Permanent |
| **Runtime overhead** | Hook calls | Forward compute | None |
| **Reversibility** | ✅ Easy | ✅ Easy | ❌ No |
| **Paper equivalent** | Partial | Partial | ✅ Full |

## Recommended Workflow

1. **Training:** Use hooks (simpler) or LoRA-style (if PEFT integration needed)
2. **Deployment:** Convert to permanent weight modifications

```python
# After training
trained_vectors = trainer.get_vectors()

# Bake into model (permanent)
apply_vectors_to_weights_permanent(model, trained_vectors)
model.save_pretrained('steered_model')

# Now standard inference works
model = AutoModelForCausalLM.from_pretrained('steered_model')
outputs = model.generate(prompts)  # Steering baked in!
```

## Verifying Equivalence

Both approaches produce identical outputs (within numerical precision):

```python
# Method 1: Hooks
outputs_hooks = generate_with_hooks(model, prompts, vectors)

# Method 2: Weight modifications
modified_model = apply_permanent_modifications(model, vectors)
outputs_weights = modified_model.generate(prompts)

# Should match
assert torch.allclose(outputs_hooks, outputs_weights, atol=1e-5)
```

See `tests/test_conversion_equivalence.py` for comprehensive tests.

## Key Formula Summary

| Operation | Formula | Code |
|-----------|---------|------|
| **Projection matrix** | P = I - v̂v̂ᵀ | `P = I - torch.outer(v, v)` |
| **Hook ablation** | h' = (I - v̂v̂ᵀ)h | `h - (h·v)v` |
| **Weight orthogonalization** | W' = (I - v̂v̂ᵀ)W | `P @ W` (LEFT multiply) |
| **Bias orthogonalization** | b' = (I - v̂v̂ᵀ)b | `P @ b` |

**Critical:** Weight modification is LEFT multiplication (`P @ W`), not right (`W @ P`).
