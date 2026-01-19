# Projection as Rank-1 LoRA

## Your Insight

**Question:** Can the projection operation be implemented as rank-1 LoRA?

**Answer:** YES! And this would be significantly better for training, debugging, and extension.

## Mathematical Connection

### Standard LoRA

```
W' = W + ΔW
ΔW = AB where A ∈ R^(d×r), B ∈ R^(r×d)
```

For rank-1: `ΔW = ab^T` (outer product)

### Our Projection

```
h' = h - (h·v)v = (I - vv^T)h
```

For output projection:
```
h_out = W @ h_in
h_projected = (I - vv^T) @ h_out
```

This is equivalent to:
```
h_projected = (I - vv^T) @ W @ h_in
            = (W - vv^T @ W) @ h_in
            = W_eff @ h_in
```

Where: `W_eff = W - vv^T @ W`

**This is rank-1 modification!**

## Two Ways to Implement

### Option 1: Projection as Post-Layer

Instead of modifying W, add a projection layer after:

```
Layer 1: h_temp = W @ h_in       (frozen)
Layer 2: h_out = P @ h_temp      (trainable)
         where P = I - vv^T
```

**This is rank-1 LoRA-like!**
- Identity matrix (frozen)
- Minus rank-1 update (trainable: v)

### Option 2: Integrate with PEFT

Use PEFT's LoRA infrastructure but customize:

```python
from peft import LoraConfig, get_peft_model

# Custom projection adapter
class ProjectionAdapter(nn.Module):
    """Rank-1 projection (LoRA-style but with subtraction)."""

    def __init__(self, dim):
        super().__init__()
        # Single trainable vector (rank-1!)
        self.v = nn.Parameter(torch.randn(dim) * 0.01)

    def forward(self, x):
        # Project out v: x - (x·v)v
        v_norm = self.v / (self.v.norm() + 1e-8)
        projection = torch.einsum('...d,d->...', x, v_norm)
        return x - torch.einsum('...,d->...d', projection, v_norm)
```

## Benefits of LoRA-Style Implementation

### 1. Cost Efficiency ✅

**Memory:**
- Standard LoRA (rank-8): 8 vectors per layer
- Our projection: 1 vector per layer (rank-1!)
- **8× more efficient than typical LoRA**

**Computation:**
- LoRA: Matrix multiply (rank-r)
- Projection: 2 dot products (rank-1)
- **Cheaper than LoRA!**

### 2. Library Optimizations ✅

Using PEFT/LoRA infrastructure gives:

**Automatic:**
- ✅ Mixed precision training (bitsandbytes)
- ✅ Gradient checkpointing
- ✅ Efficient parameter updates
- ✅ Memory-efficient loading
- ✅ Quantization support (QLoRA!)

**Integration:**
- ✅ HuggingFace Trainer compatibility
- ✅ Accelerate support
- ✅ DeepSpeed integration
- ✅ FSDP support

### 3. Debugging & Extension ✅

**Debugging:**
- Standard PEFT debugging tools work
- Can use `print_trainable_parameters()`
- Compatible with model inspection tools
- Standard save/load mechanisms

**Extension:**
- Can mix with actual LoRA (tune weights + project)
- Easy to add/remove adapters
- Modular architecture
- Can use PEFT's merge/unmerge

### 4. Ecosystem Compatibility ✅

Works with:
- Transformers Trainer
- TRL (for RLHF/GRPO!)
- Accelerate
- bitsandbytes
- All PEFT features

## Implementation Comparison

### Current (Hooks)

```python
# Manual hook registration
handles = []
for layer in model.layers:
    handle = layer.register_forward_hook(ablation_hook(v))
    handles.append(handle)

# Manual cleanup
for h in handles:
    h.remove()

# Manual parameter management
optimizer = Adam([vectors], lr=1e-4)
```

**Issues:**
- ❌ Manual hook management
- ❌ No PEFT integration
- ❌ No automatic optimizations
- ❌ Custom training loop needed

### PEFT-Compatible (LoRA-style)

```python
from peft import PeftModel, PeftConfig

# Automatic adapter injection
config = ProjectionConfig(
    target_modules=["layers"],  # Which layers to modify
    projection_dim=hidden_dim
)

model = get_peft_model(model, config)

# Automatic parameter management
model.print_trainable_parameters()
# Output: trainable params: 131,072 || all params: 2,780,000,000 || trainable%: 0.0047

# Standard training
trainer = Trainer(
    model=model,
    args=training_args,
    ...
)
trainer.train()

# Automatic save/load
model.save_pretrained("projection_adapter")
```

**Benefits:**
- ✅ Automatic everything
- ✅ Full PEFT ecosystem
- ✅ Optimized by default
- ✅ Standard interfaces

## Cost Comparison

### Training Cost

| Aspect | Hooks | LoRA-style PEFT |
|--------|-------|-----------------|
| **Memory overhead** | Minimal | Minimal |
| **Computation** | Hook calls | Optimized ops |
| **Gradient updates** | Manual | Automatic |
| **Mixed precision** | Manual | Automatic |
| **Checkpointing** | Manual | Automatic |
| **Multi-GPU** | Complex | Built-in |

**Winner: PEFT** (automatic optimizations)

### Debug Cost

| Aspect | Hooks | LoRA-style PEFT |
|--------|-------|-----------------|
| **Inspect parameters** | Custom | `print_trainable_parameters()` |
| **Save/load** | Custom | `save_pretrained()` |
| **Visualization** | Custom | Standard tools |
| **Profiling** | Manual | Built-in |

**Winner: PEFT** (standard tools work)

### Extension Cost

| Aspect | Hooks | LoRA-style PEFT |
|--------|-------|-----------------|
| **Add features** | Modify hooks | Add adapters |
| **Combine methods** | Complex | `model.add_adapter()` |
| **A/B testing** | Manual | `model.set_adapter()` |
| **Merge to base** | Custom | `model.merge_and_unload()` |

**Winner: PEFT** (modular design)

## Implementation Plan

### 1. Custom PEFT Adapter

```python
from peft import PeftConfig, PeftModel
from peft.tuners import BaseTuner, BaseTunerLayer

class ProjectionConfig(PeftConfig):
    """Config for projection adapters."""

    def __init__(
        self,
        target_modules: list,
        projection_dim: int,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.target_modules = target_modules
        self.projection_dim = projection_dim
        self.peft_type = "PROJECTION"


class ProjectionLayer(BaseTunerLayer):
    """Rank-1 projection adapter (LoRA-style)."""

    def __init__(self, base_layer, projection_dim):
        super().__init__()
        self.base_layer = base_layer

        # Single trainable vector (rank-1!)
        self.projection_vector = nn.Parameter(
            torch.randn(projection_dim) * 0.01
        )

    def forward(self, x, *args, **kwargs):
        # Base layer
        result = self.base_layer(x, *args, **kwargs)

        # Apply projection
        if isinstance(result, tuple):
            activations = result[0]
        else:
            activations = result

        # Project out vector
        v = self.projection_vector / (self.projection_vector.norm() + 1e-8)
        proj = torch.einsum('...d,d->...', activations, v)
        projected = activations - torch.einsum('...,d->...d', proj, v)

        if isinstance(result, tuple):
            return (projected,) + result[1:]
        return projected
```

### 2. Integration with TRL GRPO

```python
from trl import GRPOTrainer
from peft import get_peft_model

# Load base model
model = AutoModelForCausalLM.from_pretrained("model")

# Add projection adapters (PEFT-compatible!)
projection_config = ProjectionConfig(
    target_modules=["layers"],
    projection_dim=model.config.hidden_size
)
model = get_peft_model(model, projection_config)

# Standard TRL training!
trainer = GRPOTrainer(
    model=model,  # Works directly!
    tokenizer=tokenizer,
    reward_fn=reward_fn,
    config=grpo_config
)

trainer.train()

# Save just the adapters (tiny!)
model.save_pretrained("projection_adapters")
```

**Key advantage:** Works directly with TRL's GRPOTrainer! No custom adaptation needed.

### 3. Training Cost Savings

**Memory:**
```
Base model: 2.7B params × 2 bytes (fp16) = 5.4 GB
Projection vectors: 26 layers × 2048 dims × 2 bytes = 107 KB
Ratio: 0.002%
```

**Even with QLoRA:**
```
Base model: 2.7B params × 0.5 bytes (4-bit) = 1.35 GB
Projection vectors: 107 KB
Can train on consumer GPU!
```

## Recommendation

**Implement as PEFT-compatible adapter:**

✅ **Much cheaper:**
- Automatic optimizations (mixed precision, gradient checkpointing)
- Can use QLoRA (4-bit base model!)
- Better memory efficiency

✅ **Much easier to debug:**
- Standard PEFT tools work
- `print_trainable_parameters()`
- Standard visualization

✅ **Much easier to extend:**
- Add/remove adapters
- Combine with LoRA
- A/B test different projections

✅ **Better integration:**
- Works with TRL directly (no custom GRPOTrainer needed!)
- Compatible with HuggingFace ecosystem
- Standard save/load

## Next Steps

1. Implement `ProjectionConfig` and `ProjectionLayer`
2. Register with PEFT as custom adapter type
3. Test with TRL GRPOTrainer
4. Benchmark vs hooks
5. Document PEFT-based workflow

This would be a significant improvement over hooks!
