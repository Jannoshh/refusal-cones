# Implementation Details: Gradients and GP Structure

## Question 1: Batch Size for Gradient Estimation

### The Setup

When computing gradients, we need to evaluate R(v) over multiple prompts:

```python
def measure_refusal_with_grad(v: torch.Tensor) -> tuple:
    """
    Measure refusal when ablating with vector v.

    Args:
        v: Direction to ablate [n_layers, hidden_dim]

    Returns:
        R: Refusal rate (scalar)
        grad: Gradient ∂R/∂v [n_layers, hidden_dim]
    """
    v.requires_grad = True

    # Apply ablation to model
    model_ablated = apply_projection(model, v)

    # Generate on BATCH of harmful prompts
    responses = model_ablated.generate(harmful_prompts)  # ← Batch here!

    # Score all responses
    scores = harmbench_classifier(responses)  # [batch_size]

    # Refusal rate (mean over batch)
    R = (scores < 0.5).float().mean()  # ← Aggregation here!

    # Backprop through mean
    R.backward()
    grad = v.grad.clone()

    return R.item(), grad
```

### Batch Size Trade-offs

**Gradient variance vs computation:**

```python
# Small batch (e.g., 4 prompts)
R = mean([0.2, 0.8, 0.1, 0.9]) = 0.5
# High variance in gradient!
# ∇R is noisy but fast to compute

# Large batch (e.g., 32 prompts)
R = mean([0.2, 0.8, 0.1, 0.9, 0.3, 0.7, ...]) = 0.51
# Lower variance in gradient
# ∇R is stable but slow to compute
```

### Recommended Batch Sizes

| Phase | Batch Size | Reasoning |
|-------|-----------|-----------|
| **Discovery** | 8-16 | Balance speed vs stability |
| **Gradient ascent** | 4-8 | Fast iterations, can tolerate noise |
| **RDO training** | 4-8 | Per GPU, accumulate if needed |
| **Final evaluation** | 32+ | Accurate measurement |

### Implementation

```python
def measure_refusal_with_grad(
    v: torch.Tensor,
    prompts: List[str],
    batch_size: int = 8,  # ← Key parameter
    num_batches: int = 2   # ← Average over multiple batches
) -> Tuple[float, torch.Tensor]:
    """
    Compute R(v) and ∇R(v) with controlled batch size.

    Args:
        v: Direction [n_layers, hidden_dim]
        prompts: Pool of harmful prompts
        batch_size: Prompts per forward pass
        num_batches: Number of batches to average over

    Returns:
        R: Mean refusal rate
        grad: Mean gradient ∂R/∂v
    """
    v.requires_grad = True

    R_samples = []
    grad_accumulator = torch.zeros_like(v)

    for _ in range(num_batches):
        # Sample batch of prompts
        batch_prompts = random.sample(prompts, batch_size)

        # Apply ablation
        model_ablated = apply_projection(model, v)

        # Generate
        responses = model_ablated.generate(batch_prompts)

        # Score
        scores = harmbench_classifier(batch_prompts, responses)
        R_batch = (scores < 0.5).float().mean()

        # Backprop
        R_batch.backward()

        # Accumulate
        R_samples.append(R_batch.item())
        grad_accumulator += v.grad / num_batches

        # Zero gradient for next batch
        v.grad.zero_()

    # Final estimates
    R = np.mean(R_samples)
    grad = grad_accumulator

    return R, grad
```

### Gradient Variance Analysis

**Expected gradient variance:**

```
Var(∇R) ≈ σ²/batch_size

where σ² is the variance of individual prompt scores

Typical values:
  batch_size=1:   Var(∇R) ≈ 0.25  (very noisy!)
  batch_size=4:   Var(∇R) ≈ 0.06  (moderate)
  batch_size=16:  Var(∇R) ≈ 0.015 (stable)
  batch_size=64:  Var(∇R) ≈ 0.004 (very stable)
```

**Convergence implications:**

```python
# With noisy gradients (small batch)
lr = 0.05  # Need smaller learning rate
n_steps = 30  # Need more steps

# With stable gradients (large batch)
lr = 0.2  # Can use larger learning rate
n_steps = 10  # Fewer steps needed
```

### Practical Recommendations

**For gradient ascent (discovery):**
```python
config = GradientDiscoveryConfig(
    gradient_lr=0.1,
    batch_size=8,          # Moderate batch
    num_batches=2,         # Average over 2 batches = 16 total prompts
    n_gradient_steps=20    # Enough for convergence
)
```

**For RDO training:**
```python
training_args = TrainingArguments(
    per_device_train_batch_size=4,  # Small per GPU
    gradient_accumulation_steps=4,   # Accumulate to effective batch=16
    # Effective: 4 × 4 = 16 prompts per update
)
```

## Question 2: GP Structure - Joint vs Per-Layer

### Current Implementation: Joint GP

**What we're doing now:**

```python
# In adaptive_geometry_discovery.py

# Generate candidates
candidates = torch.randn(n, n_layers, hidden_dim)  # [n, 26, 2048]

# Flatten for GP
V_flat = candidates.reshape(n, -1)  # [n, 26×2048] = [n, 53248]

# Single GP over full space
mean, std = gp.predict(V_flat)  # GP input: R^53248
```

**Structure:**
- Input: v ∈ R^(n_layers × hidden_dim) = R^53,248
- Output: R(v) ∈ R
- One GP models: R: R^53248 → R

**Kernel:**
```python
K[i,j] = k(v_i, v_j) = exp(-||v_i - v_j||² / 2σ²)

where v_i, v_j ∈ R^53248
```

### Alternative: Per-Layer GP

**Structure:**
```python
# Separate GP for each layer
for layer in range(n_layers):
    v_layer = v[layer]  # [hidden_dim] = R^2048

    # Layer-specific GP
    gp_layer = SimpleGP()
    R_layer = gp_layer.predict(v_layer)  # GP input: R^2048

# Combine predictions
R_total = combine(R_0, R_1, ..., R_25)
```

**Combination strategies:**

```python
# Strategy 1: Weighted sum
R(v) = Σᵢ wᵢ · Rᵢ(vᵢ)

# Strategy 2: Product (all layers must contribute)
R(v) = ∏ᵢ Rᵢ(vᵢ)

# Strategy 3: Max (worst layer dominates)
R(v) = maxᵢ Rᵢ(vᵢ)

# Strategy 4: Learned combination
R(v) = f_θ(R₀(v₀), R₁(v₁), ..., R₂₅(v₂₅))
```

### Comparison

| Aspect | Joint GP | Per-Layer GP |
|--------|----------|--------------|
| **Input dimension** | d = 53,248 | d = 2,048 per GP |
| **Number of GPs** | 1 | 26 (one per layer) |
| **Captures inter-layer correlations** | ✓ Yes | ✗ No (assumes independence) |
| **Computational complexity** | O(n³) in 53K-dim | 26 × O(n³) in 2K-dim |
| **Memory** | O(n²) for 53K kernel | 26 × O(n²) for 2K kernels |
| **Kernel evaluation** | Expensive (53K distances) | Cheaper (2K distances) |
| **Sample efficiency** | Better (models correlations) | Worse (ignores structure) |
| **Parallelizable** | No | Yes (26 independent GPs) |

### Which to Use?

**Use Joint GP (current) when:**
- ✓ Layers likely interact (refusal is compositional effect)
- ✓ Have enough memory (n < 1000 observations)
- ✓ Want best sample efficiency
- ✓ Using sparse GP or inducing points for scaling

**Use Per-Layer GP when:**
- ✓ Layers act mostly independently
- ✓ Need to scale to many observations (n > 5000)
- ✓ Want parallelization across layers
- ✓ Memory is very limited

### Reality: Layers are Correlated!

**Evidence that joint GP is better:**

```python
# Refusal requires coordination across layers
# Example: Llama-2-7B refusal

Layer 0:  Detects "harmful" tokens
Layer 5:  Builds semantic understanding
Layer 10: Recognizes harmful intent
Layer 15: Plans refusal response
Layer 20: Generates refusal text
Layer 25: Outputs refusal

# If you ablate only Layer 10, refusal might still work
# If you ablate Layers 10-15, refusal breaks
# → Layers interact! Need joint modeling
```

**Correlation structure:**

```python
# Measure correlation between layers
for i, j in layer_pairs:
    corr[i,j] = correlation(R_when_ablating_i, R_when_ablating_j)

# Typical result:
# Adjacent layers: corr ≈ 0.7 (high!)
# Distant layers: corr ≈ 0.3 (moderate)
# → Not independent!
```

### Hybrid Approach (Best of Both)

**Recommended for large-scale:**

```python
class HybridGP:
    """
    Hybrid approach:
    1. Start with per-layer GPs (fast exploration)
    2. Discover low-dimensional structure via PCA
    3. Switch to joint GP in PCA subspace (accurate)
    """

    def __init__(self):
        # Phase 1: Per-layer GPs
        self.layer_gps = [SimpleGP() for _ in range(n_layers)]

        # Phase 2: Joint GP (will be initialized later)
        self.joint_gp = None

        # PCA state
        self.use_pca = False
        self.pca_components = None

    def predict(self, v: torch.Tensor):
        if self.use_pca and self.joint_gp is not None:
            # Use joint GP in PCA subspace
            return self.joint_gp.predict(v)
        else:
            # Use per-layer GPs
            predictions = []
            for layer in range(n_layers):
                mu, sigma = self.layer_gps[layer].predict(v[layer])
                predictions.append((mu, sigma))

            # Combine (e.g., weighted average)
            mu_combined = torch.stack([p[0] for p in predictions]).mean()
            sigma_combined = torch.stack([p[1] for p in predictions]).mean()

            return mu_combined, sigma_combined

    def switch_to_joint(self, V_observed, R_observed):
        """Switch to joint GP after discovering structure."""

        # PCA to find low-dimensional structure
        pca = PCA()
        pca.fit(V_observed.reshape(len(V_observed), -1))

        intrinsic_dim = estimate_intrinsic_dim(pca)

        if intrinsic_dim <= 100:  # Low enough to use joint GP
            print(f"Switching to joint GP in {intrinsic_dim}D subspace")

            self.use_pca = True
            self.pca_components = pca.components_[:intrinsic_dim]

            # Initialize joint GP in PCA space
            V_pca = project_to_pca(V_observed, self.pca_components)
            self.joint_gp = SimpleGP()
            self.joint_gp.fit(V_pca, R_observed)
```

**Benefits:**
- Fast early exploration (per-layer GPs, parallelizable)
- Accurate late exploration (joint GP in low-dim subspace)
- Automatically adapts to discovered geometry

### Recommended Implementation

For this project, use **joint GP** because:

1. **Refusal is compositional** - layers work together
2. **We have gradients** - reduces needed observations (n ≈ 50-100)
3. **PCA reduction** - can project to low-dim (k ≈ 3-10)
4. **Sparse GP** - if needed for scaling

**Code:**

```python
from adaptive_geometry_discovery import SimpleGP

# Joint GP over all layers
gp = SimpleGP(kernel_type='rbf', lengthscale=0.3)

# Observations (flattened)
V_observed = []  # Each element: [n_layers, hidden_dim]
R_observed = []

for v, R in zip(vectors, refusal_rates):
    # Flatten to single vector
    v_flat = v.reshape(-1)  # [n_layers * hidden_dim]
    V_observed.append(v_flat)
    R_observed.append(R)

# Fit joint GP
V_tensor = torch.stack(V_observed)  # [n, n_layers * hidden_dim]
R_tensor = torch.tensor(R_observed)
gp.fit(V_tensor, R_tensor)

# Predict
candidates_flat = candidates.reshape(n_candidates, -1)
mean, std = gp.predict(candidates_flat)
```

### Memory and Computational Costs

**Joint GP:**
```
Memory:  O(n² × d) where d = 53,248
         For n=100: ~5 GB (kernel matrix)

Compute: O(n³) for inference
         For n=100: ~1M operations (fast!)
```

**With sparse GP (recommended for n > 500):**
```
Memory:  O(n × m × d) where m = 100 (inducing points)
         For n=1000: ~500 MB

Compute: O(n × m²)
         For n=1000: ~10M operations (still fast)
```

### Summary

**Batch Size for Gradients:**
- Use `batch_size=8-16` for discovery
- Average over `num_batches=2` for stability
- Trade-off: variance vs speed
- Smaller batches OK for gradient ascent (can tolerate noise)
- Larger batches better for final evaluation

**GP Structure:**
- Use **joint GP** (current implementation is correct!)
- Layers are correlated (refusal is compositional)
- Input is flattened v ∈ R^53,248
- Use PCA reduction to k ≈ 3-10 dimensions if discovered
- Use sparse GP (inducing points) if n > 500 observations

**The current implementation is the right choice!**
