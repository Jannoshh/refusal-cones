# Per-Layer Refusal Vector Training

This document describes the per-layer vector training implementation for studying how refusal cones evolve across transformer layers.

## Overview

Instead of training a single refusal vector and ablating it across all layers, this implementation trains **one vector per layer**. Each vector:
- Tries to minimize loss independently
- Works together with other layers' vectors
- Can be analyzed for layer-specific refusal properties

## Key Features

### 1. Per-Layer Vectors

```python
from per_layer_training import PerLayerRefusalVectors

# Initialize one vector per layer
layer_vectors = PerLayerRefusalVectors(
    n_layers=26,
    hidden_dim=2304,
    device='cuda'
)

# Access specific layer's vector
vector_layer_15 = layer_vectors.get_vector(15)

# Get all vectors
all_vectors = layer_vectors.get_all_vectors()  # Shape: [n_layers, hidden_dim]
```

### 2. Smooth Max Loss

The smooth max loss weights worst-performing vectors more heavily, allowing optimization to focus on improving the weakest layers.

#### Mathematical Formulation

```
smooth_max(losses) = max(losses) + τ * log(mean(exp((losses - max(losses)) / τ)))
```

Where:
- `τ` (temperature) controls the smoothness
- `τ → 0`: approaches hard max (focus on worst layer only)
- `τ → ∞`: approaches mean (treat all layers equally)

#### Properties

1. **Differentiable**: Unlike hard max, smooth max has gradients everywhere
2. **Bounded**: Always between max(losses) and max(losses) + τ*log(n)
3. **Weighted**: Automatically gives higher weight to higher losses

#### Example

```python
from per_layer_training import smooth_max_loss, weighted_smooth_max_loss

# Per-layer losses
losses = torch.tensor([1.5, 2.0, 6.0, 2.5, 1.8])

# Smooth max (focuses on the 6.0 layer)
sm = smooth_max_loss(losses, temperature=1.0)

# Weighted version (returns weights)
weighted_loss, weights = weighted_smooth_max_loss(losses, temperature=1.0)
# weights will be highest for the 6.0 loss
```

### 3. Training Procedure

The training alternates between two objectives:

1. **Per-layer optimization**: Each vector minimizes loss when applied to its layer alone
2. **Combined optimization**: All vectors work together when applied simultaneously

```python
# Simplified training loop
for layer_idx in range(n_layers):
    # Apply ablation only at this layer
    loss_single = compute_loss_with_single_ablation(layer_idx)
    per_layer_losses.append(loss_single)

# Aggregate with smooth max
total_loss = smooth_max_loss(per_layer_losses)

# Also compute combined loss (all layers)
combined_loss = compute_loss_with_all_ablations()

# Final loss
final_loss = 0.5 * total_loss + 0.5 * combined_loss
```

## Usage

### Basic Training

```python
from per_layer_training import train_per_layer_vectors

trained_vectors, history = train_per_layer_vectors(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    n_layers=26,
    hidden_dim=2304,
    batch_size=8,
    epochs=10,
    lr=1e-3,
    use_smooth_max=True,
    smooth_max_temperature=1.0,
    verbose=True
)
```

### Evaluation

```python
from per_layer_training import evaluate_per_layer_vectors

results = evaluate_per_layer_vectors(
    model=model,
    tokenizer=tokenizer,
    eval_dataset=eval_dataset,
    layer_vectors=trained_vectors,
    refusal_tokens=[235285],  # For Gemma
    device='cuda'
)

# Results include:
# - per_layer_scores: Refusal scores for each layer independently
# - combined_score: Refusal score when all layers work together
# - per_layer_refusal_rates: Fraction of refused prompts per layer
# - combined_refusal_rate: Overall refusal rate
```

## Research Questions

This implementation enables studying:

1. **Layer-wise evolution**: How does refusal change across layers?
2. **Critical layers**: Which layers are most important for refusal?
3. **Dimensionality**: What is the dimensionality of refusal subspace at each layer?
4. **Synergy**: Do layers work together or independently?

### Example Analysis

```python
# Analyze training history
for entry in history:
    per_layer_losses = entry['per_layer_losses']

    # Find worst-performing layer
    worst_layer = np.argmax(per_layer_losses)

    # Analyze loss progression
    print(f"Epoch {entry['epoch']}: Worst layer = {worst_layer}")
    print(f"  Loss range: {min(per_layer_losses):.3f} - {max(per_layer_losses):.3f}")
```

## Differences from Single-Vector Approach

| Aspect | Single Vector | Per-Layer Vectors |
|--------|--------------|-------------------|
| Parameters | 1 vector (e.g., 2304 dims) | n_layers vectors (e.g., 26 × 2304) |
| Ablation | Same direction at all layers | Different direction per layer |
| Training | One optimization target | n_layers + 1 targets (individual + combined) |
| Analysis | Global refusal direction | Layer-specific directions |
| Flexibility | Limited | Can adapt to layer-specific features |

## Smooth Max Temperature Tuning

Choose temperature based on your objective:

- **τ = 0.1**: Very focused on worst layer (aggressive)
- **τ = 1.0**: Balanced focus (default, recommended)
- **τ = 2.0**: More democratic, considers all layers
- **τ = 10.0**: Nearly uniform weighting

Visualize the effect:
```bash
python test_smooth_max.py
# Generates: results/plots/smooth_max_visualization.png
```

## Integration with Existing Code

To use with existing datasets from `directopt.py`:

```python
from directopt import CustomDataset, custom_collate

# Use your existing dataset
train_dataset = CustomDataset(...)

# Train per-layer vectors
trained_vectors, history = train_per_layer_vectors(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    ...
)
```

## Disabling Addition Loss

Per the implementation, addition loss is **not used** in per-layer training. The focus is purely on ablation:
- Each layer learns its refusal direction through ablation
- No activation addition is performed during training
- This allows cleaner analysis of refusal mechanisms

## Implementation Details

### Memory Efficiency

The implementation processes layers sequentially during the per-layer phase to minimize memory:

```python
for layer_idx in range(n_layers):
    # Only this layer's hook is active
    handle = register_hook_for_layer(layer_idx)
    loss = forward_and_compute_loss()
    handle.remove()  # Clean up immediately
```

### Gradient Flow

Gradients flow through:
1. Individual layer losses → smooth max → combined loss → vectors
2. Combined loss → all vectors simultaneously

This ensures both individual and collective optimization.

### Normalization

Vectors are normalized after each update:
```python
optimizer.step()
layer_vectors.normalize()  # Project back to unit sphere
```

## Future Extensions

Potential enhancements:

1. **Subspace learning**: Train a subspace (multiple vectors) per layer
2. **Layer clustering**: Group similar layers, share vectors within groups
3. **Adaptive temperature**: Learn temperature as a parameter
4. **Hierarchical training**: First train early layers, then late layers
5. **Transfer learning**: Initialize from pre-trained refusal directions

## References

- **Smooth maximum**: Common in robust optimization (e.g., robust neural network training)
- **Log-sum-exp trick**: Numerical stability technique
- **Per-layer interventions**: Activation steering literature (Zou et al., 2023)

## Testing

Run tests to verify implementation:

```bash
# Test smooth max properties
python test_smooth_max.py

# Run example training (requires model access)
python example_per_layer_training.py
```

## Saving and Loading

```python
# Save
torch.save({
    'vectors': layer_vectors.vectors.cpu(),
    'n_layers': n_layers,
    'hidden_dim': hidden_dim,
    'history': history
}, 'results/per_layer_vectors.pt')

# Load
checkpoint = torch.load('results/per_layer_vectors.pt')
layer_vectors = PerLayerRefusalVectors(
    n_layers=checkpoint['n_layers'],
    hidden_dim=checkpoint['hidden_dim']
)
layer_vectors.vectors.data = checkpoint['vectors'].to(device)
```

## Troubleshooting

### High Memory Usage

If training uses too much memory:
- Reduce `batch_size`
- Process fewer samples per epoch
- Use gradient accumulation

### Slow Convergence

If vectors aren't improving:
- Increase learning rate (try 1e-2)
- Reduce smooth max temperature (try 0.5)
- Check that dataset has sufficient signal
- Verify ablation is actually affecting outputs

### Uneven Layer Performance

If some layers perform much worse:
- This is expected! That's what smooth max addresses
- Lower temperature to focus more on worst layers
- Check if certain layers have different feature norms
- Consider layer-specific learning rates
