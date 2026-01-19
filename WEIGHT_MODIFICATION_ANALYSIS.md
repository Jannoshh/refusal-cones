# Weight/Bias Modifications vs. Runtime Hooks

## The Question

Instead of applying steering vectors via runtime hooks, can we:
1. **Ablation** → Modify weights to permanently project out directions
2. **Addition** → Modify biases to add constant offsets

This could simplify implementation by avoiding custom forward passes.

## Mathematical Analysis

### Activation Addition via Bias Modification

**Current approach (hooks):**
```python
def hook(module, input, output):
    return output + alpha * v  # Add steering vector
```

**Equivalent bias modification:**
```python
# One-time modification
layer.bias = layer.bias + alpha * v

# Now standard forward pass achieves same result
output = layer(input)  # Bias already includes v
```

**Math:**
```
Standard:     h_next = W @ h + b
With hook:    h_next = W @ h + b + α*v
With bias:    h_next = W @ h + (b + α*v)
```

✅ **Perfect equivalence!** Just add `α*v` to bias vector.

### Ablation via Weight Modification

**Current approach (hooks):**
```python
def hook(module, input, output):
    return output - projection(output, v)  # Project out v
```

**Equivalent weight modification:**

This is trickier. The hook computes:
```
h' = h - (h · v)v  # Remove component in direction v
```

For a linear layer `h_next = W @ h`, if we want to remove direction `v` from the **output**, we can modify the output projection:

```python
# Modify weight matrix to project out v from outputs
W_modified = W @ (I - v @ v^T)
```

where `I - v @ v^T` is the projection matrix that removes direction `v`.

**Math:**
```
Standard:     h_next = W @ h
With hook:    h_next = (W @ h) - ((W @ h) · v)v = (I - vv^T) @ (W @ h)
With weights: h_next = (W @ (I - vv^T)) @ h = W' @ h
```

Where `W' = W @ (I - vv^T)` (assuming v is normalized).

✅ **Also equivalent!** But modifies weight matrix (rank-1 update).

## Implementation Comparison

### Current Approach: Runtime Hooks

```python
def generate_with_intervention(model, prompts, vectors):
    # Register hooks
    handles = []
    for idx, layer in enumerate(model.layers):
        def hook(module, input, output):
            # Ablation
            output = output - projection(output, vectors[idx])
            return output
        handles.append(layer.register_forward_hook(hook))

    # Generate
    outputs = model.generate(prompts)

    # Cleanup
    for h in handles:
        h.remove()

    return outputs
```

**Pros:**
- ✅ Non-destructive (model unchanged)
- ✅ Easy to swap vectors during training
- ✅ Gradients flow naturally through vectors
- ✅ Can test different vectors without model modifications

**Cons:**
- ❌ Requires custom forward pass logic
- ❌ Hook overhead during inference
- ❌ More complex code
- ❌ Can't use standard TRL/HF tools directly

### Proposed Approach: Weight/Bias Modifications

```python
def apply_vector_modifications(model, vectors, alpha=1.0):
    """Modify model weights/biases to include steering vectors."""
    for idx, layer in enumerate(model.layers):
        v = vectors[idx]

        # For ablation: modify output projection weights
        # W' = W @ (I - v @ v^T)
        I = torch.eye(layer.hidden_size, device=v.device)
        projection_matrix = I - torch.outer(v, v)

        # Modify weight (this is destructive!)
        layer.o_proj.weight.data = layer.o_proj.weight.data @ projection_matrix

        # For addition: modify bias (if needed)
        # layer.bias.data += alpha * v

def generate_with_intervention(model, prompts, vectors):
    # Apply modifications
    apply_vector_modifications(model, vectors)

    # Standard generation (no hooks!)
    outputs = model.generate(prompts)

    # Restore original weights
    restore_original_weights(model)

    return outputs
```

**Pros:**
- ✅ No runtime hooks needed
- ✅ Can use standard model.generate()
- ✅ Potentially faster inference
- ✅ Could work with TRL directly (?)

**Cons:**
- ❌ Destructive (changes model state)
- ❌ Need to track/restore original weights
- ❌ More complex gradient flow during training
- ❌ Applying/unapplying modifications has overhead
- ❌ Risk of numerical errors accumulating

## Impact on RL Training

### Current Approach (Hooks)

```python
def grpo_step(prompts, vectors):
    # 1. Sample vector perturbations
    sampled_vectors = [sample_noise(vectors) for _ in K]

    # 2. Generate with each (hooks apply vectors on-the-fly)
    for v_set in sampled_vectors:
        completions = generate_with_hooks(prompts, v_set)
        # ✓ Vectors are in computation graph
        # ✓ Gradients flow naturally

    # 3. Compute loss and backprop
    loss.backward()  # ✓ Gradients go to vectors directly
    optimizer.step()  # ✓ Update vectors
```

**Clean and straightforward** ✅

### Proposed Approach (Weight Modifications)

```python
def grpo_step(prompts, vectors):
    # 1. Sample vector perturbations
    sampled_vectors = [sample_noise(vectors) for _ in K]

    # 2. Generate with each (requires weight modifications)
    for v_set in sampled_vectors:
        # Apply modifications
        apply_modifications(model, v_set)  # Changes model state

        # Generate
        completions = model.generate(prompts)  # Not in computation graph!

        # Restore
        restore_weights(model)  # Changes model state again

        # ❌ How do gradients flow back to vectors?
        # ❌ Weight modifications broke the computation graph

    # 3. Compute loss and backprop
    loss.backward()  # ❌ No gradients to vectors!

    # Would need to manually compute gradients through weight modifications
    # Much more complex!
```

**Problem:** Weight modifications are outside the computation graph!

We'd need to:
1. Track how weight modifications depend on vectors
2. Manually compute gradients through the modification operation
3. Accumulate gradients correctly
4. Much more complex!

## Hybrid Approach: Best of Both Worlds?

### During Training: Use Hooks
```python
# Training: hooks keep vectors in computation graph
trainer = VectorGRPOTrainer(
    model=model,
    vectors=vectors,
    # Uses hooks internally for clean gradients
)
trainer.train(prompts, num_steps=1000)
```

### After Training: Convert to Weight Modifications
```python
# Deployment: convert to weight modifications for speed
final_vectors = trainer.get_vectors()

# Apply permanent modifications (one-time)
apply_vector_modifications(model, final_vectors)

# Save modified model
model.save_pretrained('model_with_steering')

# Now can use standard inference
outputs = model.generate(prompts)  # Fast, no hooks!
```

## Recommendation

**For your use case (RL training of steering vectors):**

### Keep Hooks for Training ✅

**Reasons:**
1. **Gradient flow:** Vectors stay in computation graph
2. **Flexibility:** Easy to try different vectors during exploration
3. **Safety:** Model stays frozen (non-destructive)
4. **Simplicity:** No weight apply/restore logic needed

### Use Weight Modifications for Deployment (Optional)

After training, if you want to deploy the final steering vectors:
```python
# One-time conversion
apply_vector_modifications(model, trained_vectors)
model.save_pretrained('steered_model')
```

Now you have a standalone model with steering baked in.

## Code Addition: Conversion Utilities

We could add a utility module:

```python
# conversion_utils.py

def convert_vectors_to_weights(model, vectors, inplace=False):
    """
    Convert trained steering vectors to permanent weight modifications.

    Use this AFTER training to create a deployable model.
    """
    if not inplace:
        model = copy.deepcopy(model)

    for idx, layer in enumerate(model.layers):
        v = vectors[idx]

        # Ablation: modify output projection
        I = torch.eye(v.shape[0], device=v.device)
        projection_matrix = I - torch.outer(v, v)
        layer.o_proj.weight.data = layer.o_proj.weight.data @ projection_matrix

    return model

def compare_hook_vs_weight_modification(model, vectors, prompts):
    """
    Verify that weight modifications produce same results as hooks.
    """
    # Method 1: Hooks
    outputs_hooks = generate_with_hooks(model, prompts, vectors)

    # Method 2: Weight modifications
    modified_model = convert_vectors_to_weights(model, vectors)
    outputs_weights = modified_model.generate(prompts)

    # Should be identical (within numerical precision)
    assert torch.allclose(outputs_hooks, outputs_weights, atol=1e-5)
```

## Summary

**Your intuition is correct:** Weight/bias modifications can achieve the same effect as runtime hooks.

**However, for RL training:**
- **Hooks are better:** Cleaner gradients, non-destructive, more flexible
- **Weight modifications complicate training:** Break computation graph, need manual gradient tracking

**Best approach:**
1. **Training:** Use hooks (current implementation) ✅
2. **Deployment:** Optionally convert to weight modifications for speed
3. **Add conversion utilities** for post-training deployment

**Does this simplify implementation?**
- For training: **No, makes it more complex**
- For deployment: **Yes, simpler inference**

**Recommendation:** Keep hook-based training, add optional weight conversion utilities for deployment.

---

## Implementation Plan

If you want to add weight conversion (for deployment, not training):

1. Create `conversion_utils.py` with:
   - `convert_vectors_to_weights()` - Apply permanent modifications
   - `verify_equivalence()` - Test that both methods match

2. Add to documentation:
   - When to use hooks (training, testing)
   - When to use weight modifications (deployment)
   - How to convert after training

3. Keep current RL implementation using hooks (don't change)

Would you like me to implement the conversion utilities as an optional deployment feature?
