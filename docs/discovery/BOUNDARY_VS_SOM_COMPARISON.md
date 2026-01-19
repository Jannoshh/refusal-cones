# Boundary Discovery vs SOM: A Comparison

This document compares our boundary/level-set discovery approach with the SOM (Sum-of-Max) method for characterizing refusal geometry.

## Overview

| Aspect | SOM Paper | Boundary Discovery |
|--------|-----------|-------------------|
| **Goal** | Find directions inside refusal region | Find the boundary of refusal region |
| **Output** | k sample directions | Complete boundary characterization |
| **Information** | Point samples | Full geometric structure |
| **Model** | Fixed cone assumption | Adaptive, data-driven shape |

## SOM Approach (Sum-of-Max)

The SOM method from the original paper finds refusal directions by:

1. **Assumption**: Refusal is mediated by a low-dimensional cone (linear subspace)
2. **Method**: Find k orthogonal directions that maximize refusal when ablated
3. **Output**: k vectors spanning the refusal cone

```python
# SOM conceptual approach
def som_discovery(measure_fn, k):
    """Find k orthogonal refusal directions."""
    directions = []
    for i in range(k):
        # Find direction maximizing refusal, orthogonal to previous
        v = optimize(
            objective=lambda v: measure_fn(v),
            constraint=orthogonal_to(directions)
        )
        directions.append(v)
    return directions
```

**Key limitation**: Assumes refusal is a cone. If the true geometry is curved or multi-modal, SOM may miss important structure.

## Boundary Discovery Approach

Our approach finds the **boundary** where R(v) crosses a threshold:

1. **No shape assumption**: Discovers actual geometry
2. **Method**: GP-based active learning with straddle acquisition
3. **Output**: Set of boundary points + geometric characterization

```python
# Boundary discovery conceptual approach
def boundary_discovery(measure_fn, threshold=0.5):
    """Find the boundary where R(v) ≈ threshold."""
    gp = GaussianProcess()

    while not converged:
        # Straddle acquisition: explore where uncertain AND near threshold
        v_next = argmax(beta * sigma(v) - |mu(v) - threshold|)
        R = measure_fn(v_next)
        gp.update(v_next, R)

    return identify_boundary_points(gp, threshold)
```

## Straddle Acquisition Function

The key innovation is the **straddle heuristic**:

```
acquisition(v) = β · σ(v) - |μ(v) - threshold|
```

- **β · σ(v)**: Exploration term (high in uncertain regions)
- **|μ(v) - threshold|**: Exploitation term (low near the boundary)

This naturally focuses sampling on the boundary while maintaining exploration.

### Gradient Enhancement (Optional)

When gradients are available:

```
acquisition(v) = straddle(v) + α · perpendicular_bonus(v)
```

Where `perpendicular_bonus` prefers directions along the contour (perpendicular to ∇R), helping trace the boundary efficiently.

## Detailed Comparison

### Pros of Boundary Discovery

1. **Complete Characterization**
   - Knowing the boundary = knowing everything about the region
   - Can reconstruct the full refusal region from boundary
   - Answers: "What is the shape of refusal space?"

2. **No Shape Assumptions**
   - Works for non-convex regions
   - Handles multiple disconnected components
   - Captures curved boundaries accurately

3. **Natural Uncertainty Quantification**
   - GP provides uncertainty estimates everywhere
   - Know where boundary is well-characterized vs uncertain
   - Can continue sampling to reduce uncertainty

4. **Geometric Insights**
   - Estimates boundary dimension (d-1 for d-dimensional interior)
   - Detects number of connected components
   - Provides inside/outside classification

5. **Robust to Local Minima**
   - Doesn't require finding global maximum
   - Threshold-based: more stable than peak-finding
   - Multiple boundary points provide redundancy

### Cons of Boundary Discovery

1. **More Measurements Required**
   - Boundary is higher-dimensional than cone
   - Need to characterize entire boundary surface
   - SOM: O(k²) measurements for k-dim cone
   - Boundary: O(k × more) for k-dim boundary

2. **Threshold Selection**
   - Requires choosing meaningful threshold
   - Different thresholds = different boundaries
   - May need domain knowledge to set

3. **Computational Overhead**
   - GP inference at each step: O(n³)
   - More complex than linear algebra
   - Sparse GP helps but adds complexity

4. **Interpretation Challenge**
   - Many boundary points harder to interpret than k directions
   - Need post-processing for actionable insights
   - May require PCA/clustering for summary

### Pros of SOM

1. **Interpretable Output**
   - k orthogonal directions are easy to understand
   - Direct analogy to PCA
   - Clear "refusal basis"

2. **Efficient for Simple Geometry**
   - If refusal IS a cone, SOM is optimal
   - Fewer measurements needed
   - Fast linear algebra operations

3. **Direct Training Integration**
   - k vectors → k-dim cone for ablation
   - Natural PEFT/LoRA parameterization
   - Established training pipelines

### Cons of SOM

1. **Strong Assumptions**
   - Assumes linear subspace (cone)
   - May miss curved or multi-modal structure
   - False confidence if assumption violated

2. **Local Optima**
   - Sequential greedy selection
   - Later directions constrained by earlier
   - May miss better configurations

3. **Fixed Dimensionality**
   - Must choose k in advance
   - Too small: miss structure
   - Too large: overfitting to noise

## When to Use Each

### Use Boundary Discovery When:

- **Unknown geometry**: Don't know if refusal is a cone
- **Research/exploration**: Want to understand the true structure
- **Multi-modal suspected**: May have disconnected refusal regions
- **Robust characterization**: Need uncertainty-aware estimates

### Use SOM When:

- **Known cone structure**: Prior evidence that refusal is linear
- **Fast iteration**: Need quick approximation
- **Training pipeline**: Direct integration with PEFT
- **Simple models**: Smaller models likely have simpler geometry

## Hybrid Approach

The best of both worlds:

```python
# 1. Discover boundary first
boundary_results = BoundaryGeometryDiscovery(...).discover()

# 2. Analyze boundary geometry
if boundary_results.boundary_dimension <= k:
    # Simple geometry: fit cone to boundary
    cone_vectors = fit_cone_to_boundary(boundary_results.boundary_points, k)
    # Use cone for efficient training
else:
    # Complex geometry: use boundary directly
    # Consider neural field or multi-component approach
```

## Empirical Considerations

### Measurement Budget Trade-offs

| Budget | SOM | Boundary |
|--------|-----|----------|
| 20-50 | Good for k≤5 | Rough boundary |
| 50-100 | Good for k≤10 | Moderate accuracy |
| 100-200 | Diminishing returns | Good characterization |
| 200+ | Unnecessary | Detailed boundary |

### Geometry Complexity

| Geometry | SOM Performance | Boundary Performance |
|----------|-----------------|---------------------|
| True cone | Excellent | Good (recovers cone) |
| Curved manifold | Poor | Good |
| Multiple modes | Very poor | Excellent |
| High-dimensional | Diminishing returns | Scalable with sparse GP |

## Implementation Notes

### Threshold Selection

For refusal tasks, common thresholds:

- **0.5**: Natural decision boundary (refuse vs comply)
- **0.1-0.2**: Conservative boundary (low refusal region)
- **0.8-0.9**: Aggressive boundary (high refusal region)

Recommendation: Start with 0.5, adjust based on use case.

### Post-Discovery Analysis

After boundary discovery, you can:

1. **Fit cone**: If boundary is approximately linear, fit k vectors
2. **Find peaks**: Search inside the boundary for refusal modes
3. **Sample interior**: Use boundary to constrain interior sampling
4. **Visualize**: Project to 2D/3D for inspection

### Integration with Training

```python
# Option 1: Use boundary for initialization
boundary = BoundaryDiscoveryResults(...)
v_init = boundary.boundary_points.mean(dim=0)  # Center of boundary

# Option 2: Use boundary for regularization
loss = task_loss + lambda * boundary_violation_penalty(v, boundary)

# Option 3: Sample within boundary
def sample_inside_boundary(boundary, n):
    """Sample points inside the discovered boundary."""
    # Use GP to predict R(v), accept if R > threshold
    ...
```

## Conclusion

Boundary discovery and SOM are complementary approaches:

- **SOM**: Fast, interpretable, efficient when assumptions hold
- **Boundary**: Complete, flexible, robust to complex geometry

For most research applications, we recommend:

1. Start with boundary discovery to understand true geometry
2. If geometry is simple (low boundary dimension), use SOM/cone
3. If geometry is complex, use boundary-aware methods

The straddle acquisition function makes boundary discovery efficient, requiring only ~50-100 measurements for useful characterization.
