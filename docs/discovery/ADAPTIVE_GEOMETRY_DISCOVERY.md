# Adaptive Geometry Discovery for Refusal Subspaces

## The Problem with Fixed Cones

**Current approach (cones):**
- Fix rank k (e.g., k=3 vectors)
- Optimize k orthogonal directions
- **Problem:** How do we know k is right?
  - Too small k → Miss important directions
  - Too large k → Include non-refusal directions
  - Geometry might not be a simple k-dimensional subspace!

**Better question:** What is the *actual* geometry of the refusal subspace, and how can we discover it adaptively?

## Theoretical Foundation

### 1. Refusal Landscape as a Smooth Function

**Definition:** Let h ∈ ℝ^d be an activation vector. Define the refusal strength function:

```
R(v) = Expected refusal when ablating direction v
```

Where:
- v ∈ S^(d-1) (unit vectors on hypersphere)
- R(v) ∈ [0, 1] measures "how much ablating v increases refusal"

**Key insight:** R(v) is a smooth function on the hypersphere!

Why?
- Similar directions should have similar effects
- Continuity from model smoothness
- Can be modeled with smooth functions (GP, neural fields, etc.)

### 2. The Refusal Subspace Geometry

**Definition:** The refusal subspace is the set of directions where R(v) is high:

```
S_refusal = {v ∈ S^(d-1) : R(v) > threshold}
```

**Question:** What shape is S_refusal?

**Possibilities:**
1. **Simple cone:** k-dimensional linear subspace
2. **Curved manifold:** Smooth submanifold of S^(d-1)
3. **Multiple modes:** Several disconnected regions
4. **Hierarchical:** Nested structures at different scales

**We don't know a priori!** Need to discover adaptively.

### 3. Gaussian Process Model

**Approach:** Model R(v) as a Gaussian Process over the hypersphere.

**GP Definition:**
```
R ~ GP(μ(v), k(v, v'))
```

Where:
- μ(v): Mean function (prior belief about R)
- k(v, v'): Kernel function measuring similarity of directions

**Kernel choices:**

1. **Dot product kernel (linear):**
   ```
   k(v, v') = (v · v')^p
   ```
   Good for: Linear subspaces (standard cones)

2. **Geodesic kernel (spherical):**
   ```
   k(v, v') = exp(-θ²(v, v') / 2σ²)
   where θ(v, v') = arccos(v · v')
   ```
   Good for: Smooth manifolds on sphere

3. **Von Mises-Fisher kernel:**
   ```
   k(v, v') = exp(κ · v · v')
   ```
   Good for: Concentrated distributions around modes

**GP Posterior:**

After observing R at directions V = {v₁, ..., vₙ}:

```
R(v) | observations ~ N(μ_post(v), σ²_post(v))

μ_post(v) = μ(v) + k(v, V)ᵀ [K(V,V) + σ²I]⁻¹ (R_obs - μ(V))
σ²_post(v) = k(v, v) - k(v, V)ᵀ [K(V,V) + σ²I]⁻¹ k(v, V)
```

**Key property:** Uncertainty σ²_post(v) tells us where we're uncertain!

### 4. Active Exploration Strategy

**Goal:** Efficiently discover S_refusal by choosing which directions to probe next.

**Acquisition Functions:**

Borrowing from Bayesian Optimization:

1. **Upper Confidence Bound (UCB):**
   ```
   a_UCB(v) = μ_post(v) + β · σ_post(v)
   ```
   Balances: High expected refusal + High uncertainty

   **Intuition:** Explore directions that might be high refusal

2. **Expected Improvement:**
   ```
   a_EI(v) = E[max(0, R(v) - R_max)]
   where R_max = max observed R so far
   ```
   **Intuition:** Find directions better than current best

3. **Boundary Exploration:**
   ```
   a_boundary(v) = |μ_post(v) - threshold| · σ_post(v)
   ```
   **Intuition:** Find the boundary of S_refusal precisely

**Algorithm:**

```
1. Initialize: Sample random directions, measure R(v)
2. Fit GP: R ~ GP(μ, k)
3. While not converged:
   a. Compute acquisition function a(v) for all v
   b. Select: v_next = argmax_v a(v)
   c. Measure: R(v_next) by training and testing
   d. Update GP with new observation
4. Return: Discovered geometry
```

### 5. Geometry Extraction

Once we have accurate GP model of R(v), extract the geometry:

**Method 1: Thresholding**
```
S_refusal = {v : μ_post(v) > threshold}
```

**Method 2: Level Sets**
```
For multiple thresholds τ₁, τ₂, ...:
  S_τ = {v : μ_post(v) > τ}
```
This gives hierarchical structure!

**Method 3: Principal Directions**
```
Find directions where R(v) has local maxima:
∇_v R(v) = 0  and  eigenvalues of Hessian < 0
```

These are the "core" refusal directions.

**Method 4: Manifold Learning**
```
Sample points from high-R regions
Apply manifold learning (UMAP, diffusion maps)
Extract intrinsic dimension and coordinates
```

## 6. Practical Implementation

### Step 1: Define Refusal Strength Metric

**For ablation:**
```python
def measure_refusal_strength(model, vectors, harmful_prompts, refusal_tokens):
    """
    Measure how much ablating vectors increases refusal.

    Returns:
        R(v): Refusal rate with ablation - baseline refusal rate
    """
    # Baseline (no ablation)
    baseline_refusal = test_refusal_rate(model, harmful_prompts)

    # With ablation
    ablated_refusal = test_refusal_rate_with_ablation(
        model, harmful_prompts, vectors
    )

    # Refusal strength: how much ablation increases refusal
    R = ablated_refusal - baseline_refusal
    return R
```

**For adversarial (jailbreak research):**
```python
def measure_jailbreak_strength(model, vectors, harmful_prompts, reward_model):
    """
    Measure how much ablating vectors increases harmfulness.

    Returns:
        R(v): Harmfulness with ablation - baseline harmfulness
    """
    baseline_harm = test_harmfulness(model, harmful_prompts, reward_model)
    ablated_harm = test_harmfulness_with_ablation(
        model, harmful_prompts, vectors, reward_model
    )

    R = ablated_harm - baseline_harm
    return R
```

### Step 2: GP Implementation

```python
import gpytorch
from gpytorch.kernels import RBFKernel, LinearKernel

class RefusalGP(gpytorch.models.ExactGP):
    """
    Gaussian Process model for refusal strength R(v).
    """

    def __init__(self, train_v, train_R, kernel_type='rbf'):
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        super().__init__(train_v, train_R, likelihood)

        self.mean = gpytorch.means.ConstantMean()

        if kernel_type == 'rbf':
            # Geodesic distance on sphere
            self.covar = RBFKernel()
        elif kernel_type == 'linear':
            # Dot product (for linear subspaces)
            self.covar = LinearKernel()
        else:
            # Composite: linear + nonlinear
            self.covar = LinearKernel() + RBFKernel()

    def forward(self, v):
        mean = self.mean(v)
        covar = self.covar(v)
        return gpytorch.distributions.MultivariateNormal(mean, covar)
```

### Step 3: Acquisition Functions

```python
def ucb_acquisition(gp, v, beta=2.0):
    """
    Upper Confidence Bound acquisition.

    Args:
        gp: Trained GP model
        v: Direction to evaluate
        beta: Exploration parameter

    Returns:
        UCB score (higher = more promising)
    """
    with torch.no_grad():
        posterior = gp(v)
        mean = posterior.mean
        std = posterior.variance.sqrt()

    return mean + beta * std


def boundary_acquisition(gp, v, threshold=0.5):
    """
    Boundary exploration acquisition.

    Focuses on finding the boundary of refusal subspace.
    """
    with torch.no_grad():
        posterior = gp(v)
        mean = posterior.mean
        std = posterior.variance.sqrt()

    # High score near threshold with high uncertainty
    distance_to_threshold = torch.abs(mean - threshold)
    return -distance_to_threshold * std  # Negative because we want to minimize distance
```

### Step 4: Adaptive Exploration Loop

```python
def discover_refusal_geometry(
    model,
    tokenizer,
    harmful_prompts,
    refusal_tokens,
    n_layers,
    hidden_dim,
    n_iterations=100,
    n_init=20
):
    """
    Adaptively discover the geometry of refusal subspace.

    Returns:
        gp: Trained GP model of R(v)
        discovered_directions: High-refusal directions found
        geometry_info: Extracted geometric properties
    """
    # Step 1: Initialize with random directions
    print("Initializing with random exploration...")

    init_directions = []
    init_refusal_strengths = []

    for i in range(n_init):
        # Random direction
        v = torch.randn(n_layers, hidden_dim)
        v = v / v.norm(dim=1, keepdim=True)

        # Measure refusal strength
        R = measure_refusal_strength(model, v, harmful_prompts, refusal_tokens)

        init_directions.append(v)
        init_refusal_strengths.append(R)

        print(f"  Init {i+1}/{n_init}: R = {R:.3f}")

    # Convert to tensors
    V_observed = torch.stack(init_directions)
    R_observed = torch.tensor(init_refusal_strengths)

    # Step 2: Fit initial GP
    print("\nFitting Gaussian Process...")
    gp = RefusalGP(V_observed, R_observed, kernel_type='rbf+linear')

    # Train GP hyperparameters
    gp.train()
    likelihood = gp.likelihood
    optimizer = torch.optim.Adam(gp.parameters(), lr=0.1)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, gp)

    for epoch in range(50):
        optimizer.zero_grad()
        output = gp(V_observed)
        loss = -mll(output, R_observed)
        loss.backward()
        optimizer.step()

    gp.eval()

    # Step 3: Active exploration
    print("\nActive exploration...")

    for iteration in range(n_iterations):
        # Sample candidate directions
        n_candidates = 1000
        candidates = torch.randn(n_candidates, n_layers, hidden_dim)
        candidates = candidates / candidates.norm(dim=1, keepdim=True)

        # Compute acquisition scores
        with torch.no_grad():
            scores = []
            for v in candidates:
                score = ucb_acquisition(gp, v.unsqueeze(0), beta=2.0)
                scores.append(score.item())
            scores = torch.tensor(scores)

        # Select best candidate
        best_idx = scores.argmax()
        v_next = candidates[best_idx]

        # Measure refusal strength
        R_next = measure_refusal_strength(model, v_next, harmful_prompts, refusal_tokens)

        print(f"  Iteration {iteration+1}: R = {R_next:.3f}, UCB = {scores[best_idx]:.3f}")

        # Update GP
        V_observed = torch.cat([V_observed, v_next.unsqueeze(0)])
        R_observed = torch.cat([R_observed, torch.tensor([R_next])])

        # Retrain GP (incremental update would be faster)
        gp = RefusalGP(V_observed, R_observed, kernel_type='rbf+linear')
        # ... (retrain as above)

        # Check convergence
        if iteration > 10 and scores.max() < threshold_converged:
            print("  Converged!")
            break

    # Step 4: Extract geometry
    print("\nExtracting geometry...")
    geometry_info = extract_geometry(gp, V_observed, R_observed)

    return gp, V_observed, R_observed, geometry_info
```

### Step 5: Geometry Extraction

```python
def extract_geometry(gp, V_observed, R_observed, threshold=0.5):
    """
    Extract geometric properties of the refusal subspace.
    """
    geometry = {}

    # 1. Find principal directions (local maxima)
    print("  Finding principal refusal directions...")
    principal_dirs = find_local_maxima(gp, V_observed, R_observed)
    geometry['principal_directions'] = principal_dirs
    geometry['n_modes'] = len(principal_dirs)

    # 2. Estimate intrinsic dimension
    print("  Estimating intrinsic dimension...")
    high_refusal_points = V_observed[R_observed > threshold]
    if len(high_refusal_points) > 10:
        intrinsic_dim = estimate_intrinsic_dimension(high_refusal_points)
        geometry['intrinsic_dimension'] = intrinsic_dim

    # 3. Extract boundary
    print("  Extracting subspace boundary...")
    boundary_points = find_boundary_points(gp, threshold)
    geometry['boundary'] = boundary_points

    # 4. Characterize curvature
    print("  Analyzing curvature...")
    curvature = analyze_curvature(gp, principal_dirs)
    geometry['curvature'] = curvature

    return geometry


def find_local_maxima(gp, V_observed, R_observed, n_restarts=10):
    """
    Find local maxima of R(v) - these are principal refusal directions.
    """
    principal_dirs = []

    # Start from observed high-R points
    top_k = R_observed.topk(min(n_restarts, len(R_observed)))
    init_points = V_observed[top_k.indices]

    for v_init in init_points:
        # Optimize to local maximum
        v_opt = optimize_to_local_max(gp, v_init)

        # Check if new maximum
        is_new = True
        for v_existing in principal_dirs:
            if (v_opt - v_existing).norm() < 0.1:  # Close to existing
                is_new = False
                break

        if is_new:
            principal_dirs.append(v_opt)

    return principal_dirs


def estimate_intrinsic_dimension(points):
    """
    Estimate intrinsic dimension using MLE or correlation dimension.
    """
    from sklearn.decomposition import PCA

    # PCA-based estimate
    pca = PCA()
    pca.fit(points.numpy())

    # Find number of components explaining 95% variance
    cumsum = np.cumsum(pca.explained_variance_ratio_)
    intrinsic_dim = np.argmax(cumsum > 0.95) + 1

    return intrinsic_dim
```

## 7. Advanced: Neural Implicit Representations

Instead of GP, use a **neural field** to model R(v):

```python
class RefusalField(nn.Module):
    """
    Neural implicit representation of refusal landscape.

    Models R(v) as a neural network that takes direction v as input.
    """

    def __init__(self, input_dim, hidden_dim=256):
        super().__init__()

        # Positional encoding for better high-frequency representation
        self.encoding = PositionalEncoding(input_dim, n_freqs=10)

        # MLP
        encoded_dim = input_dim * (2 * 10 + 1)  # From positional encoding
        self.net = nn.Sequential(
            nn.Linear(encoded_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()  # R(v) ∈ [0, 1]
        )

    def forward(self, v):
        # Ensure unit vectors
        v = v / (v.norm(dim=-1, keepdim=True) + 1e-8)

        # Encode
        encoded = self.encoding(v)

        # Predict refusal strength
        R = self.net(encoded)

        return R.squeeze(-1)


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for better high-freq representation."""

    def __init__(self, input_dim, n_freqs=10):
        super().__init__()
        self.n_freqs = n_freqs
        self.input_dim = input_dim

        # Frequency bands
        freq_bands = 2.0 ** torch.linspace(0, n_freqs-1, n_freqs)
        self.register_buffer('freq_bands', freq_bands)

    def forward(self, v):
        # v: [batch, input_dim]

        # Compute sin and cos for each frequency
        encoded = [v]  # Include original

        for freq in self.freq_bands:
            encoded.append(torch.sin(freq * v))
            encoded.append(torch.cos(freq * v))

        return torch.cat(encoded, dim=-1)
```

**Advantages of neural fields:**
- Can represent complex, non-smooth geometries
- Scales to very high dimensions
- Can model hierarchical structure
- Can incorporate physics-based priors

## 8. Stopping Criteria

**When to stop exploration?**

1. **Uncertainty threshold:**
   ```
   max_v σ_post(v) < ε
   ```
   Stop when uncertainty everywhere is low

2. **Acquisition saturation:**
   ```
   max_v a(v) < threshold
   ```
   Stop when no promising directions remain

3. **Coverage criterion:**
   ```
   Fraction of sphere with σ_post(v) < ε > 0.95
   ```
   Stop when 95% of sphere is well-explored

4. **Practical budget:**
   ```
   n_evaluations > budget
   ```
   Stop after fixed number of refusal measurements

## 9. Theoretical Guarantees

**Question:** Can we bound the error in our discovered geometry?

**Answer:** Yes, under GP framework!

**Theorem (GP Confidence Bounds):**

With probability ≥ 1 - δ:
```
|R(v) - μ_post(v)| ≤ β_t · σ_post(v)

where β_t = 2 log(|S^(d-1)| t² π² / 6δ)
```

**Implication:** If we explore until σ_post(v) < ε everywhere, then we know R(v) to within ±β_t·ε everywhere!

**Corollary:** The boundary of S_refusal is located to within ±β_t·ε.

## 10. Multi-Scale Geometry

**Observation:** Refusal might have hierarchical structure!

**Example:**
- Coarse scale: General "harmful intent" direction
- Fine scale: Specific types of harm (violence, illegal, etc.)

**Approach:** Multi-resolution GP

```python
class MultiScaleGP:
    """
    Model refusal at multiple scales.
    """

    def __init__(self, scales=[0.1, 0.5, 2.0]):
        self.gps = []
        for length_scale in scales:
            gp = RefusalGP(kernel_lengthscale=length_scale)
            self.gps.append(gp)

    def forward(self, v):
        # Combine predictions from all scales
        predictions = [gp(v) for gp in self.gps]

        # Weighted average (can learn weights)
        combined = sum(predictions) / len(predictions)

        return combined
```

## Summary

**New approach: Adaptive Geometry Discovery**

Instead of fixed cones:
1. **Model refusal landscape:** R(v) as smooth function (GP or neural field)
2. **Active exploration:** Use acquisition functions to choose which directions to probe
3. **Extract geometry:** Find principal directions, intrinsic dimension, boundaries
4. **Theoretical guarantees:** GP confidence bounds ensure accuracy

**Benefits:**
- ✅ Discovers true geometry (not assumed)
- ✅ Adapts to data
- ✅ Principled uncertainty quantification
- ✅ Efficient exploration (active learning)
- ✅ Works for complex geometries (manifolds, multiple modes, etc.)

**Trade-offs:**
- ⚠ More complex than fixed cones
- ⚠ Requires multiple refusal measurements (but efficient!)
- ⚠ Need to define refusal strength metric

**Recommendation:** Use for research/analysis to understand refusal geometry, then can simplify to fixed cone if geometry is simple.
