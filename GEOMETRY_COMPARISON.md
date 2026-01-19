# Geometry Comparison: Cones vs Adaptive Discovery

## The Question

> "I don't think we need the cones anymore? Also In 3D, what kind of surface are we getting? For 3D cones we were sampling from the hyper unit circle because of the normalization, what are we sampling from now?"

Great question! Let's clarify the fundamental difference in geometry and sampling.

## Cones: Assumed Linear Geometry

### What Cones Assume

Cones assume the refusal subspace is a **k-dimensional linear subspace** through the origin.

```
Cone assumption: Refusal directions lie in span{v₁, v₂, ..., vₖ}
```

### 3D Visualization (k=2 cone in 3D space)

```
     z
     |
     |    ╱
     |  ╱  refusal cone (2D plane)
     |╱_____ y
    ╱|
  ╱  |
x    |

The cone is a 2D plane through origin.
After normalization, refusal vectors lie on the CIRCLE
where this plane intersects the unit sphere.
```

**Sampling space for cones:**
- Sample from S^(k-1) (the (k-1)-dimensional sphere)
- In 3D with k=2: Sample from S^1 (a circle on the 2-sphere)
- Parameterization: v = α₁v₁ + α₂v₂ where α₁² + α₂² = 1

**Constraint:** Vectors are restricted to the k-dimensional subspace.

### What Cones Get Wrong

Real refusal geometry might NOT be a linear subspace:

```
❌ Assumption: Refusal subspace is flat (linear)

✓ Reality could be:
  - Curved manifold
  - Multiple disconnected regions
  - Non-convex boundaries
  - Variable dimensionality across the space
```

## Adaptive Discovery: Learn the True Geometry

### What We're Actually Modeling

Instead of assuming geometry, we model the refusal landscape as a **smooth function over the entire hypersphere**:

```
R(v): S^(d-1) → [0, 1]

Where:
- S^(d-1) is the unit hypersphere in R^d (d = n_layers × hidden_dim)
- R(v) measures refusal strength at direction v
- No geometric assumptions!
```

### 3D Visualization (full sphere, d=3)

For actual dimensions d = 26 × 2048 = 53,248, we can't visualize directly. But in 3D:

```
    R(v) is a scalar field on the sphere

         N (North Pole)
        /|\
       / | \
      /  |  \     R(v) = 0.9 (high refusal)
     /   |   \
    |    |    |   R(v) = 0.5 (medium)
    |    •    |
     \   |   /    R(v) = 0.1 (low refusal)
      \  |  /
       \ | /
        \|/
         S (South Pole)

The "surface" is like a heat map painted on the sphere.
High-refusal regions might be:
- Concentrated near "poles" (principal directions)
- Spread across curved bands
- Multiple disconnected peaks
```

**This is NOT a cone!** It's a scalar field over the entire sphere.

### Level Sets in 3D

For threshold τ = 0.5, the refusal boundary is:

```
Boundary = {v ∈ S² : R(v) = 0.5}

In 3D, this could be:
- A circle (if cone-like)
- An ellipse (if slightly curved)
- Multiple disconnected curves (if multi-modal)
- A fractal-like boundary (if complex)

        \  peak 1   /
         \  R>0.5  /
          \_______/    ← boundary curve

         /‾‾‾‾‾‾‾\
        /  peak 2  \   ← another region
```

### Sampling Space for Adaptive Discovery

**Key difference:** We sample from the **FULL hypersphere** S^(d-1), not a subspace.

```
Cone sampling:     S^(k-1) ⊂ R^d     (restricted to k-dim subspace)
Adaptive sampling: S^(d-1) = {v ∈ R^d : ||v|| = 1}  (all unit vectors)
```

**How we sample:**

```python
# Generate random unit vector on full sphere
v = torch.randn(n_layers, hidden_dim)  # Sample from R^d
v = v / v.norm()                        # Project to S^(d-1)

# This samples UNIFORMLY from the entire hypersphere
# Not restricted to any subspace!
```

**In 3D:**
- Cones (k=2): Sample from a circle (1D manifold on 2D sphere)
- Adaptive: Sample from entire sphere surface (2D manifold)

```
Cone (k=2) samples:          Adaptive samples:
      •                            •  •    •
        •                       •        •
          •                   •    •   •  •
        •                       •  •      •
      •                          •    •  •
    (circle)                   (full sphere)
```

## Do We Still Need Cones?

**Answer: It depends on what we discover!**

### Scenario 1: Discovered Geometry is Simple (Low-Dimensional Linear)

If adaptive discovery finds:
- Intrinsic dimension k ≪ d (e.g., k=3 in d=53,248)
- Principal directions form orthogonal basis
- Refusal strength is linear combination

**Then YES, use cones!**

```python
# After discovery
geometry = discovery.discover()

if geometry['intrinsic_dimension'] <= 10:
    # Geometry is simple - use cone
    cone_vectors = geometry['principal_directions']

    # Initialize cone with discovered vectors
    model = get_cone_model(
        base_model,
        cone_rank=len(cone_vectors),
        init_vectors=cone_vectors
    )
```

**Benefit:** More efficient (k parameters instead of d)

### Scenario 2: Discovered Geometry is Complex (Curved/Non-Linear)

If adaptive discovery finds:
- High intrinsic dimension (e.g., k=100)
- Curved manifold (not flat subspace)
- Multiple disconnected modes
- Non-convex boundaries

**Then NO, cones won't work well!**

```python
# After discovery
geometry = discovery.discover()

if geometry['intrinsic_dimension'] > 50 or geometry['is_curved']:
    # Geometry is complex - use GP or neural field

    # Option A: Use GP directly
    model = get_gp_guided_model(
        base_model,
        gp=discovery.gp
    )

    # Option B: Fit neural implicit field
    field = RefusalField(geometry)
    model = get_field_guided_model(base_model, field)
```

**Benefit:** Captures true geometry, not restricted to linear approximation

### The Pipeline

```
1. Run Adaptive Discovery
   ↓
   Discovers true geometry (GP model + statistics)
   ↓
2. Analyze Geometry
   ↓
   - Intrinsic dimension?
   - Curved or flat?
   - Single mode or multi-modal?
   ↓
3. Choose Representation
   ↓
   ┌─────────────────┬─────────────────┐
   │ Low-dim & flat  │ High-dim/curved │
   ├─────────────────┼─────────────────┤
   │ Use CONE        │ Use GP/Field    │
   │ (efficient)     │ (accurate)      │
   └─────────────────┴─────────────────┘
   ↓
4. Train with RDO
```

## Concrete Example: What We'd See in 3D

### If Refusal is Cone-Like (k=2 in d=3)

```
Discovery finds:
- Intrinsic dimension: 2
- Two principal directions: v₁, v₂
- High R(v) only in span{v₁, v₂}

Visualization:
         z
         |
         |   •  High R region
         | •   • (concentrated on circle)
         |• • • •
        /| • • •
       / |  • •
      /  | •
     x   y

Decision: Use 2D cone (efficient, captures geometry)
```

### If Refusal is Curved Manifold

```
Discovery finds:
- Intrinsic dimension: 2 (still low!)
- But: curved, not flat
- R(v) varies smoothly along curved 2D surface

Visualization:
         z
         |
         | •     •
         |  • • •
         |   •••    (curved band, not circle)
        /|  • • •
       / | •     •
      /  |
     x   y

Decision: Cone won't fit well - use GP or neural field
```

### If Refusal is Multi-Modal

```
Discovery finds:
- Multiple disconnected high-R regions
- Different "types" of refusal

Visualization:
         z
         |
       • | •
      •  |  •    Mode 1 (e.g., violent refusals)
         |
        /|    • •
       / |   •   •  Mode 2 (e.g., legal refusals)
      /  |
     x   y

Decision: Cones can't handle - use GP with multi-modal kernel
```

## Sampling Space Summary

| Method | Sampling Space | Dimension | Constraint |
|--------|---------------|-----------|------------|
| **Cone (k=2)** | Circle on sphere | 1D (S¹) | Must lie in span{v₁,v₂} |
| **Cone (k=3)** | Sphere in subspace | 2D (S²) | Must lie in span{v₁,v₂,v₃} |
| **Adaptive** | Full hypersphere | (d-1)D (S^(d-1)) | No constraint |

**Key insight:** Adaptive discovery explores the ENTIRE space, then we decide if we can compress to a cone.

## Mathematical Formulation

### Cone Assumption

```
Refusal subspace V = span{v₁, ..., vₖ}

Any refusal direction: v = Σᵢ αᵢvᵢ where Σᵢ αᵢ² = 1

Sampling: α ~ Uniform(S^(k-1))
```

**Constraint:** k ≪ d (much lower dimensional)

### Adaptive Discovery

```
Refusal landscape: R: S^(d-1) → [0, 1]

Model: R(v) ~ GP(0, k(v,v'))
  where k(v,v') = exp(-||v-v'||²/(2σ²))

Sampling: v ~ Uniform(S^(d-1))  [initially]
         v ~ Acquisition(GP)     [during exploration]
```

**No constraint:** Can explore full (d-1)-dimensional sphere!

## When to Use Each Approach

### Use Cones If:
✓ Discovery finds low intrinsic dimension (k ≤ 10)
✓ Geometry is approximately linear
✓ Want computational efficiency
✓ Interpretability is important

### Use Adaptive/GP If:
✓ Intrinsic dimension is high (k > 50)
✓ Geometry is curved or complex
✓ Multiple disconnected modes
✓ Accuracy more important than efficiency

### Hybrid Approach (Best of Both)

```python
# 1. Discover geometry
results = discovery.discover()

# 2. Extract low-rank approximation
if results['geometry']['intrinsic_dimension'] <= 10:
    # Use cone initialized from discovered directions
    cone_vectors = results['geometry']['principal_directions']
    model = get_cone_model(init_vectors=cone_vectors)
else:
    # Use GP guidance
    model = get_gp_guided_model(gp=results['gp'])

# 3. Refine with RDO
trainer.train(model)
```

## Visualization Code for 3D Case

Here's code to visualize the difference:

```python
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def visualize_cone_vs_adaptive():
    """Compare cone sampling vs adaptive discovery in 3D."""

    fig = plt.figure(figsize=(15, 5))

    # Cone sampling (k=2 in 3D)
    ax1 = fig.add_subplot(131, projection='3d')

    # Define 2D subspace (cone basis)
    v1 = np.array([1, 0, 0])
    v2 = np.array([0, 1, 0])

    # Sample from circle in this subspace
    theta = np.linspace(0, 2*np.pi, 100)
    cone_samples = np.array([
        np.cos(theta) * v1[:, None] + np.sin(theta) * v2[:, None]
    ]).squeeze().T

    ax1.plot(cone_samples[:, 0], cone_samples[:, 1], cone_samples[:, 2], 'b-', linewidth=2)
    ax1.scatter(cone_samples[::10, 0], cone_samples[::10, 1], cone_samples[::10, 2], c='blue')
    ax1.set_title('Cone Sampling (k=2)\nRestricted to Circle')

    # Adaptive sampling (full sphere)
    ax2 = fig.add_subplot(132, projection='3d')

    # Sample uniformly from sphere
    n_samples = 500
    adaptive_samples = np.random.randn(n_samples, 3)
    adaptive_samples /= np.linalg.norm(adaptive_samples, axis=1)[:, None]

    ax2.scatter(adaptive_samples[:, 0], adaptive_samples[:, 1], adaptive_samples[:, 2],
                c='red', alpha=0.3, s=10)
    ax2.set_title('Adaptive Sampling\nFull Sphere')

    # Refusal landscape R(v)
    ax3 = fig.add_subplot(133, projection='3d')

    # Create mock refusal function (high near v1, decays)
    def R(v):
        return np.exp(-5 * np.linalg.norm(v - v1))

    # Evaluate on sphere
    refusal_strength = np.array([R(v) for v in adaptive_samples])

    scatter = ax3.scatter(adaptive_samples[:, 0], adaptive_samples[:, 1], adaptive_samples[:, 2],
                         c=refusal_strength, cmap='hot', s=20)
    ax3.set_title('Refusal Landscape R(v)\nHeat Map on Sphere')
    plt.colorbar(scatter, ax=ax3, label='R(v)')

    plt.tight_layout()
    plt.savefig('cone_vs_adaptive_3d.png', dpi=150)
    print("Saved visualization to cone_vs_adaptive_3d.png")

if __name__ == '__main__':
    visualize_cone_vs_adaptive()
```

## Key Takeaways

1. **Cones assume linear subspace** - sample from S^(k-1) within that subspace
2. **Adaptive discovers true geometry** - sample from full S^(d-1)
3. **3D surface for adaptive** = heat map of R(v) over entire sphere
4. **Cones may still be useful** if discovered geometry is low-dimensional and linear
5. **Sampling space is much larger** for adaptive (exponentially more directions)

The adaptive approach is "geometry-agnostic" - it discovers what's actually there, then you can decide if a cone approximation is good enough.
