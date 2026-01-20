# Layer Structure Investigation

## The Question

**Do we need to optimize vectors at all layers independently, or can we predict vectors at layer i+1 from layer i?**

This is a fundamental question about the geometry of refusal vectors that could dramatically reduce the optimization space.

## What the Analysis Does

The `investigate_layer_structure.py` script answers:

1. **Adjacent layer similarity**: How correlated are vectors at neighboring layers?
   - High correlation (>0.7) → layers are smooth, could interpolate
   - Low correlation (<0.3) → layers are independent, need per-layer optimization

2. **Shared subspace**: Do all layer vectors lie in a low-dimensional subspace?
   - If yes → optimize in low-dimensional space then project to each layer
   - If no → need high-dimensional optimization

3. **Linear predictability**: Can we predict v[i+1] from v[i]?
   - Fits linear model: `v[i+1] = W @ v[i]`
   - High R² or cosine sim → could use recurrence relation
   - Low R² → layers transform information unpredictably

4. **Subset feasibility**: What if we optimize every 2nd, 4th, or 8th layer?
   - Tests interpolation quality
   - Quantifies dimension reduction vs quality tradeoff

## Usage

### Option 1: From pre-computed vectors (fastest)

```bash
# If you have Pareto vectors from a discovery run
uv run python scripts/investigate_layer_structure.py \
    --vectors results/pareto_vectors_TIMESTAMP.pt
```

### Option 2: Compute mean-difference vectors (requires model)

```bash
# Compute mean-diff vectors and analyze
uv run python scripts/investigate_layer_structure.py \
    --model Qwen/Qwen2.5-0.5B-Instruct \
    --harmful-file refusal_direction/dataset/splits/harmful_train.json \
    --harmless-file refusal_direction/dataset/splits/harmless_train.json \
    --n-prompts 32
```

## Expected Output

```
=== 1. Adjacent Layer Similarity ===
Mean cosine similarity between adjacent layers:
  Overall: 0.8234 ± 0.1203
  Range: [0.4521, 0.9567]

  Low similarity transitions (< 0.3):
    Layer 0 -> 1: 0.2341
    Layer 26 -> 27: 0.2891

=== 2. Shared Subspace Analysis (PCA) ===
  PCs for 90% variance: 12 (dim reduction: 1024 -> 12)
  PCs for 95% variance: 23 (dim reduction: 1024 -> 23)
  PCs for 99% variance: 67 (dim reduction: 1024 -> 67)

=== 3. Linear Predictability ===
  Vector 1/1:
    R² score: 0.6734
    Mean cosine similarity: 0.8123 ± 0.0891
    ✓ Strong linear relationship! Could predict layers from neighbors.

=== 4. Subset Optimization Analysis ===
  Optimize every 2nd layer (14/28 layers):
    Interpolation quality: 0.9567
    Dimension reduction: 28672 -> 14336 (50.0%)
    ✓ Excellent! Could safely use 2× subsampling.

  Optimize every 4th layer (7/28 layers):
    Interpolation quality: 0.8234
    Dimension reduction: 28672 -> 7168 (25.0%)
    ~ Good. 4× subsampling may work with minor quality loss.

  Optimize every 8th layer (4/28 layers):
    Interpolation quality: 0.6891
    Dimension reduction: 28672 -> 4096 (14.3%)
    ✗ Poor interpolation. Don't use 8× subsampling.
```

## Interpretation

### Scenario 1: High Layer Smoothness
- Adjacent similarity > 0.7
- Good interpolation with 2-4× subsampling
- **Implication**: Could optimize every 2nd or 4th layer and interpolate the rest
- **Dimension reduction**: 50-75% fewer parameters

### Scenario 2: Shared Low-Dimensional Subspace
- 90% variance in <50 PCs
- **Implication**: Optimize shared subspace coefficients instead of raw vectors
- **Approach**: `v[i] = U @ c[i]` where U is shared basis, c[i] are layer-specific coefficients

### Scenario 3: Linear Recurrence
- High R² for v[i+1] ~ W @ v[i]
- **Implication**: Learn transformation matrices W[i] or single shared W
- **Approach**: Optimize v[0] and W, generate others via recurrence

### Scenario 4: Independent Layers
- Low similarity, poor prediction, high PCA dims
- **Implication**: Layers encode different aspects of refusal, need per-layer optimization
- **No reduction possible** - current approach is optimal

## Possible Optimizations Based on Results

If the analysis shows structure, you could implement:

```python
# Option A: Subset + interpolation
class SubsetLayerOptimization:
    def __init__(self, n_layers, stride=2):
        self.optimized_layers = list(range(0, n_layers, stride))
        self.v_subset = nn.Parameter(torch.randn(len(self.optimized_layers), hidden_dim))

    def get_all_vectors(self):
        # Interpolate missing layers
        return interpolate(self.v_subset, self.optimized_layers, n_layers)

# Option B: Shared subspace
class SubspaceOptimization:
    def __init__(self, n_layers, hidden_dim, subspace_dim=50):
        self.U = nn.Parameter(torch.randn(hidden_dim, subspace_dim))  # Shared basis
        self.coeffs = nn.Parameter(torch.randn(n_layers, subspace_dim))  # Layer coefficients

    def get_all_vectors(self):
        return self.coeffs @ self.U.T  # [n_layers, hidden_dim]

# Option C: Recurrent transformation
class RecurrentOptimization:
    def __init__(self, n_layers, hidden_dim):
        self.v0 = nn.Parameter(torch.randn(hidden_dim))  # Initial vector
        self.W = nn.Parameter(torch.randn(hidden_dim, hidden_dim))  # Transformation

    def get_all_vectors(self):
        vectors = [self.v0]
        for i in range(n_layers - 1):
            v_next = self.W @ vectors[-1]
            v_next = v_next / v_next.norm()  # Keep normalized
            vectors.append(v_next)
        return torch.stack(vectors)
```

## Next Steps

1. **Run the analysis** on your model's mean-difference vectors
2. **Check the numbers** - which scenario matches your data?
3. **If structure exists**:
   - Implement reduced parameterization
   - Compare Pareto frontiers (full vs reduced)
   - Measure quality vs dimension tradeoff
4. **If no structure**:
   - Current per-layer optimization is likely optimal
   - Focus on improving GP efficiency instead

## Theoretical Context

The user's question connects to several research areas:

1. **Neural network layer similarity**: Adjacent layers often learn related features ([Raghu et al. 2017](https://arxiv.org/abs/1705.05393))

2. **Recurrent transformations**: Some steering directions propagate through layers via learned transformations

3. **Low-rank structure**: Activation spaces often lie in low-dimensional manifolds ([Ansuini et al. 2019](https://arxiv.org/abs/1905.12081))

4. **Layer importance**: Not all layers contribute equally to refusal ([Arditi et al. 2024](https://arxiv.org/abs/2406.11717) found middle layers dominate)

This analysis helps determine which (if any) of these structures apply to refusal vectors specifically.
