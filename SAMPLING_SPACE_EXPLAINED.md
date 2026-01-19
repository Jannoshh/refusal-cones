# Sampling Space: Cones vs Adaptive Discovery

## Your Questions

> **Q1:** "I don't think we need the cones anymore?"
>
> **Q2:** "In 3D, what kind of surface are we getting?"
>
> **Q3:** "For 3D cones we were sampling from the hyper unit circle because of the normalization, what are we sampling from now?"

## Direct Answers

### Q1: Do we still need cones?

**Answer: Maybe! It depends on what we discover.**

```
IF discovered geometry is:
  • Low intrinsic dimension (≤10)
  • Linear/flat subspace
  • Single-mode
THEN → YES, use cones (more efficient!)

ELSE:
  • High intrinsic dimension (>10)
  • Curved manifold
  • Multi-modal
THEN → NO, use GP or neural field instead
```

**Adaptive discovery TELLS YOU which to use!**

The pipeline:
1. Run adaptive discovery (learns true geometry)
2. Get recommendation (cone vs GP/field)
3. Initialize appropriate model
4. Train with RDO

See `adaptive_to_training.py` for complete pipeline.

### Q2: What kind of surface in 3D?

**For cones (old approach):**
```
In 3D with rank-2 cone:
- Refusal directions restricted to 2D plane through origin
- After normalization: CIRCLE on the sphere
- This is S¹ (1-dimensional manifold) embedded in S² (2-sphere)

Visualization:
       z
       |
       |   •••
       | •••••  ← circle (intersection of plane & sphere)
       |•••••
      /|•••
     / | •
    /  |
   x   y
```

**For adaptive discovery (new approach):**
```
In 3D:
- R(v) is a SCALAR FIELD defined over entire sphere S²
- Like a "heat map" painted on the sphere
- High-refusal regions form islands/bands/peaks on the surface

Visualization:
       N
      /|\
     / | \     Red zone (R ≈ 1, high refusal)
    /  |  \
   |   •   |   Yellow zone (R ≈ 0.5, medium)
    \  |  /
     \ | /     Blue zone (R ≈ 0, low refusal)
      \|/
       S

The "surface" is R(v) visualized as color/height over S²
```

**Key difference:**
- Cones: Refusal constrained to LOW-DIMENSIONAL subset (circle in 3D)
- Adaptive: Refusal can be ANYWHERE on sphere (full 2D surface in 3D)

### Q3: What are we sampling from now?

**Cones:** Sample from S^(k-1) WITHIN k-dimensional subspace

```python
# For rank-2 cone in 3D
v1 = [1, 0, 0]  # Basis vector 1
v2 = [0, 1, 0]  # Basis vector 2

# Sample from circle (S¹)
theta = random.uniform(0, 2*pi)
v = cos(theta) * v1 + sin(theta) * v2

# This is S¹ ⊂ R³ (circle in 3D space)
```

**Adaptive Discovery:** Sample from FULL hypersphere S^(d-1)

```python
# For 3D case (d=3)
v = torch.randn(3)  # Random vector in R³
v = v / v.norm()    # Project to S²

# This is S² (entire sphere surface, not just circle!)

# For real case (d = 26 × 2048 = 53,248)
v = torch.randn(26, 2048)  # Random in R^53248
v = v / v.norm()           # Project to S^53247

# This is S^53247 (entire hypersphere!)
```

**Sampling space comparison:**

| Method | Space | Dimension | Size | Example (3D) |
|--------|-------|-----------|------|--------------|
| Cone (k=1) | S^0 in R^d | 0D | 2 points | Two poles on sphere |
| Cone (k=2) | S^1 in R^d | 1D | Circle | Great circle |
| Cone (k=3) | S^2 in R^d | 2D | Sphere | 2-sphere (if d≥3) |
| Adaptive | S^(d-1) | (d-1)D | Hypersphere | Entire sphere surface |

**Visual comparison in 3D:**

```
Cone (k=2) sampling:        Adaptive sampling:
       •                           • • •
     •   •                       • • • • •
    •     •                    • • • • • • •
     •   •                       • • • • •
       •                           • • •
   (circle)                    (full sphere)

   1D manifold                 2D manifold
   S¹ ⊂ S²                     S²
```

## Why the Change?

### Problem with Cones

Cones ASSUME refusal geometry is a low-dimensional LINEAR subspace:

```
Assumption: Refusal ⊆ span{v₁, v₂, ..., vₖ}

This means:
  ✗ Must be flat (no curvature)
  ✗ Must be low-rank (k ≪ d)
  ✗ Single connected region
```

**But what if reality is different?**

```
Reality could be:
  • Curved manifold (not flat)
  • High intrinsic dimension
  • Multiple disconnected modes
  • Non-convex boundaries
```

### Solution: Adaptive Discovery

Instead of ASSUMING geometry, DISCOVER it:

```
1. Sample from FULL hypersphere (no constraints)
2. Build GP model: R(v) ~ GP(μ, k)
3. Active exploration (UCB/EI to find interesting regions)
4. Extract geometry (intrinsic dim, curvature, modes)
5. THEN decide if cone approximation works
```

**Key insight:** We're now geometry-agnostic!

## Mathematical Details

### Cone Sampling Space

For rank-k cone with orthonormal basis {v₁, ..., vₖ}:

```
Sampling space: S^(k-1) ⊂ R^d

Parameterization:
  v = Σᵢ αᵢvᵢ  where Σᵢ αᵢ² = 1

Size: (k-1)-dimensional sphere
```

**Constraint:** All samples lie in k-dimensional subspace!

### Adaptive Sampling Space

No subspace constraint:

```
Sampling space: S^(d-1) = {v ∈ R^d : ||v|| = 1}

Parameterization:
  v ~ N(0, I)  then normalize

Size: (d-1)-dimensional hypersphere
```

**No constraint:** Any unit vector is valid!

### Dimensionality Comparison

For d = 26 × 2048 = 53,248:

```
Cone (k=3):
  Sampling space: S² (2-dimensional sphere)
  Volume: ~4π ≈ 12.6
  Degrees of freedom: 2

Adaptive:
  Sampling space: S^53247 (53,247-dimensional hypersphere)
  Volume: Unimaginably large
  Degrees of freedom: 53,247

Ratio: 53,247 / 2 = 26,623× more dimensions!
```

**Adaptive explores EXPONENTIALLY more directions!**

## Geometry Types and Recommendations

### Type 1: Low-Dimensional Linear (Use Cone)

```
Discovered:
  • Intrinsic dim: 2-3
  • PCA explains 95%+ variance
  • Single mode

Decision: Use cone (k=3)

Why: Geometry is simple, cone is efficient
  Params: 3 vectors vs 53,248-dimensional field
```

### Type 2: Curved Manifold (Use GP)

```
Discovered:
  • Intrinsic dim: 2-3 (low!)
  • BUT: curved, not flat
  • PCA poor fit

Decision: Use GP or neural field

Why: Low-dim but curved - cone won't fit
  Linear subspace can't capture curvature
```

### Type 3: High-Dimensional Diffuse (Use Neural Field)

```
Discovered:
  • Intrinsic dim: 50+
  • Spread across many directions
  • No clear structure

Decision: Use neural implicit field

Why: Too complex for simple representation
  Need capacity to model high-dim manifold
```

### Type 4: Multi-Modal (Use GP with Multi-Modal Kernel)

```
Discovered:
  • Multiple disconnected peaks
  • Different "types" of refusal

Decision: Use GP with sum kernel or mixture

Why: Single cone can't span disconnected regions
  Need multi-modal representation
```

## Code Examples

### Cone Sampling (Old)

```python
# Define k-dimensional subspace
basis = torch.randn(k, d)  # k basis vectors in R^d
basis = torch.qr(basis.T)[0].T  # Orthonormalize

# Sample from S^(k-1)
alpha = torch.randn(k)
alpha = alpha / alpha.norm()

# Construct direction in subspace
v = (alpha[:, None] * basis).sum(dim=0)  # Linear combination

# v is constrained to span{basis}!
```

### Adaptive Sampling (New)

```python
# Sample from S^(d-1) (no constraint!)
v = torch.randn(d)  # Random in R^d
v = v / v.norm()    # Project to sphere

# v can be ANY unit vector!
# No subspace constraint!
```

### Comparison

```python
# For d=53,248 and k=3:

# Cone sampling
cone_samples = sample_from_cone(k=3, d=53248, n=1000)
print(cone_samples.shape)  # [1000, 53248]
# All samples lie in 3D subspace!

# Check dimensionality
from sklearn.decomposition import PCA
pca = PCA().fit(cone_samples)
print(pca.explained_variance_ratio_[:5])
# [0.4, 0.35, 0.25, 0.0, 0.0]  ← Only 3 components!

# Adaptive sampling
adaptive_samples = sample_from_sphere(d=53248, n=1000)
print(adaptive_samples.shape)  # [1000, 53248]
# Samples spread across ALL directions!

# Check dimensionality
pca = PCA().fit(adaptive_samples)
print(pca.explained_variance_ratio_[:5])
# [0.002, 0.002, 0.002, 0.002, 0.002]  ← Spread evenly!
```

## Visualization Script

Run `visualize_geometry.py` to see:

1. **Cone vs Adaptive sampling** in 3D
   - Cone: Samples on circle (S¹)
   - Adaptive: Samples on sphere (S²)

2. **Refusal landscape** R(v) as heat map on sphere

3. **Different geometry types** and when to use cones

```bash
python visualize_geometry.py
```

This generates:
- `cone_vs_adaptive_3d.png` - Sampling comparison
- `geometry_scenarios.png` - Different geometry types
- `sampling_density.png` - Density comparison

## Complete Pipeline

See `adaptive_to_training.py`:

```python
# 1. Adaptive discovery
discovery = RefusalGeometryDiscovery(...)
results = discovery.discover()

# 2. Get recommendation
recommendation = results['recommendation']

if recommendation['use_cone']:
    # 3a. Initialize cone
    model = get_cone_model(
        base_model,
        cone_rank=recommendation['cone_rank'],
        init_vectors=results['geometry']['principal_directions']
    )
else:
    # 3b. Use standard RDO (or GP-guided)
    model = get_rdo_model(base_model, ...)

# 4. Train with RDO
trainer.train(model)
```

## Summary

| Question | Answer |
|----------|--------|
| **Need cones?** | Only if discovered geometry is low-dim & linear |
| **3D surface?** | R(v) as scalar field (heat map) over sphere S² |
| **Sampling from?** | S^(d-1) (full hypersphere), NOT S^(k-1) (circle) |

**Key insight:** Adaptive discovery explores the FULL space, then we decide if we can compress to a cone.

**The geometry determines the representation, not the other way around!**
