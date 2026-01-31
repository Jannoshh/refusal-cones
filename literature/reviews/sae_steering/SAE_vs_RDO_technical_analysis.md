# Technical Analysis: SAE Steering vs. RDO Approach

**Purpose:** Detailed technical comparison of SAE feature steering (2411.11296) vs. Refusal-Cones RDO method

---

## 1. Mechanistic Differences

### SAE Steering (2411.11296)

**Model:** Sparse Autoencoder (SAE) on residual stream activations

```python
# Forward pass with SAE steering
h_t = model.residual_stream[layer]  # Activation at time t
f = SAE.encode(h_t)                 # Sparse features (24,576-dim sparse)

# Steering: amplify refusal feature
f[22373] *= amplification_factor

h_t_modified = SAE.decode(f)        # Back to activation space

# Continue generation with modified activation
logits = model.output_projection(h_t_modified)
```

**Properties:**
- **Dimension:** Sparse (typically 1-10 active features per token)
- **Reversibility:** Must be applied at every generation step
- **Interpretability:** Features are more interpretable than directions
- **Generality:** Single feature (22373) generalizes across diverse harms

### Refusal-Cones RDO (Projection Ablation)

**Model:** LoRA-style adapter implementing projection onto refusal direction

```python
# Ablation operation: remove refusal component
v = refusal_direction  # [hidden_dim] unit vector
h_ablated = h - torch.einsum('...d,d->...', h, v) * v
          = (I - vv^T) @ h              # Projection onto subspace orthogonal to v

# Equivalent LoRA form (what PEFT implements)
# During inference: h' = h + LoRA(h)
#   where LoRA = A @ B with special initialization to encode -vv^T
```

**Properties:**
- **Dimension:** Dense (full activation space, but rank-1 perturbation)
- **Reversibility:** Applied only during inference (if using adapters)
- **Interpretability:** Less interpretable (geometric direction)
- **Generality:** Must learn per-layer, per-model

---

## 2. Comparative Analysis: Why Does RDO Avoid SAE's Capability Degradation?

### Hypothesis 1: Localization of Perturbations

**SAE steering problem:**
```
Input activation: [high-dim vector]
                  ↓
           [amplify feature 22373]  ← Global perturbation in activation space
                  ↓
           Unrelated capabilities broken
```

**Key insight:** Amplifying even one sparse feature creates broad perturbations in the reconstructed activation space.

**Why:** SAE decoder reconstructs the full residual stream activation from sparse codes. Changing f[22373] affects all dimensions of the reconstruction:

```python
h_modified = SAE.decoder @ f_modified
           # Each decoder row is a learned linear combination of all features
           # Changing one feature affects output across all dimensions
```

**RDO approach:**
```
Input activation: [high-dim vector]
                  ↓
        [project onto null space of v]  ← Targeted perturbation
        h' = (I - vv^T) h
                  ↓
   Only removes refusal direction, preserves everything else orthogonal to v
```

**Key advantage:** Projection is mathematically precise. It removes *only* the component along v, preserving all orthogonal components exactly.

### Hypothesis 2: Distributed vs. Concentrated Encoding

**SAE assumption:** Refusal is encoded in Feature 22373 (and related features)

**Problem:** If refusal is truly distributed across multiple SAE features or requires complex feature interactions, then amplifying only 22373 is incomplete. This could explain why:
1. You don't see expected over-refusal (feature alone insufficient)
2. But you DO see capability degradation (feature is entangled with language ability)

**RDO assumption:** Refusal is encoded in direction v (possibly multi-dimensional via cone)

**Advantage:** If refusal is indeed concentrated in lower-dimensional subspace, then ablating that subspace is clean. The projection:
- Removes refusal (good)
- Preserves everything orthogonal (good)
- Can be extended to full cone if needed

### Hypothesis 3: Feature Entanglement in SAE Decoder

**SAE steering bottleneck:**

```python
# SAE decoder is a learned linear map from sparse to dense space
h_reconstructed = decoder @ f  # decoder: [hidden_dim, num_features]

# Every entry of h_reconstructed depends on all features:
h[i] = sum_j decoder[i, j] * f[j]

# If feature 22373 is correlated with language capabilities in its decoder rows,
# then amplifying it breaks those capabilities
```

**Example:** Suppose during SAE training:
- Feature 22373 learned to activate on refusal keywords (*refuse*, *can't*, *safety*)
- But also learned weak correlations with high-frequency language patterns
- Amplifying 22373 amplifies both signals → capability degradation

**RDO advantage:** Projection doesn't use a learned decoder. It's purely geometric:
```python
# Projection preserves all information orthogonal to v
h_projected = h - (h·v)v
            # = h - [scalar] * v
            # No decoder layer to create unwanted entanglement
```

---

## 3. Quantitative Comparison of Perturbation Magnitude

### SAE Steering

**Measured degradation:**
| Task | Baseline | SAE Steering | Change |
|------|----------|-------------|--------|
| MMLU | 68.80% | 35.98% | -32.82pp |
| GSM8K | 82.50% | 35.56% | -46.94pp |

**Interpretation:** Single feature amplification causes ~45pp average degradation

**Perturbation analysis:** If baseline model uses certain activation ranges, SAE steering shifts the entire distribution:
- Feature 22373 amplification changes decoder output across many dimensions
- Model receives out-of-distribution activations
- Off-manifold from training, causing downstream failures

### RDO Projection (Predicted)

**Expected degradation:** Much lower (to be measured)

**Reasoning:**

1. **Projection magnitude:** v is unit vector, so perturbation is at most ||h||·||proj_v(h)||
   - For random h orthogonal to v: proj_v(h) = 0 (no perturbation)
   - For random h: expect ~1/√d component removed (where d = hidden_dim ≈ 2000+)
   - For harmful prompts: expect larger but still bounded perturbation

2. **Distribution shift:** Projection keeps activations on the manifold learned during training
   - Unlike SAE steering, no decoder artifacts
   - Model sees in-distribution (but modified) activations

3. **Orthogonal preservation:** All capabilities orthogonal to v are exactly preserved
   - If language ability ⊥ v: zero degradation
   - If language ability has component along v: degradation proportional to that component

**Prediction:** RDO will show <10pp degradation if refusal is truly concentrated in 1D or low-D subspace

---

## 4. Why Multi-Objective Training (λ_retain) Helps

### RDO Training Objective

```python
total_loss = λ_ablate * loss_ablate + λ_add * loss_add + λ_retain * loss_retain

# For harmful prompts:
loss_ablate = -log P(harmful_completion | ablate(h))    # Want compliance
loss_add = log P(harmful_completion | add(h))           # Want refusal

# For harmless prompts:
loss_retain = log P(helpful_completion | unchanged)     # Preserve capability
```

### Why λ_retain Prevents Degradation

**Mechanism:** Unlike SAE steering (which just amplifies features), RDO training actively penalizes capability loss:

1. **Direct signal:** λ_retain on harmless examples forces model to maintain performance
   - Prevents learning corrupted representations
   - Guides optimization away from capability-destructive directions

2. **Geometric constraint:** Training on diverse data ensures v (or cone) stays in refusal-specific subspace
   - If v started contaminating language ability, loss_retain would spike
   - Optimizer redirects to pure refusal direction

3. **Implicit feature disentanglement:** Multi-objective training finds directions that satisfy all objectives simultaneously
   - SAE steering: No such constraint; amplifies feature regardless of side effects
   - RDO: Explicitly optimizes for clean separation

**Example comparison:**

```
SAE steering:
  Amplify feature 22373
  → Math capability breaks (no penalty)
  → Philosophy breaks (no penalty)
  → Result: 46pp degradation, no recovery mechanism

RDO training:
  Update v to maximize: λ_ablate * safety + λ_retain * capability
  → v tries to break math (loss spike on λ_retain term)
  → Optimizer learns: math and refusal are orthogonal
  → v converges to refusal-only direction
  → Result: minimal degradation, both goals achieved
```

---

## 5. Predictions and Testable Hypotheses

### Hypothesis A: RDO Avoids SAE's Degradation

**Prediction:** MMLU, GSM8K, TruthfulQA remain >80% of baseline when λ_retain > 0

**Why:** Multi-objective training explicitly protects capabilities

**Test:**
```python
# In E1.1 or E2.2 experiments:
baseline_mmlu = model.eval("MMLU")  # e.g., 70%
with_rdo_mmlu = trained_model.eval("MMLU")  # predict >63% (90% of baseline)

sae_results_from_paper = 35.98%  # Compare: RDO >> SAE approach
```

### Hypothesis B: SAE Features Correlate with RDO Cone Dimensions

**Prediction:** Top k SAE features align with principal components of refusal cone

**Why:** If refusal is truly concentrated in low-D subspace, SAE should learn features in that subspace

**Test:**
```python
# Download SAE for model (Phi-3 or Llama)
sae = load_sae_from_paper_url()

# Compare:
rdo_cone = discovered_geometry['modes']  # e.g., [3 vectors, each hidden_dim]
sae_features = sae.decoder[:, [22373, ...]]  # Top refusal features

# Correlation: cosine_sim(rdo_modes, sae_features) >> baseline
```

### Hypothesis C: Projection is More Reversible Than SAE

**Prediction:** If you train RDO to have very high λ_ablate, you can recover baseline by setting λ = 0

**Why:** Projection is invertible; changing λ just scales the effect

**Test:**
```python
# Train RDO with λ_ablate = 10.0 (very aggressive)
trained_rdo_strong = train_rdo(lambda_ablate=10.0)

# Switch off:
with disable_rdo_ablation(trained_rdo_strong):
    capability_score = eval("MMLU")  # Should recover baseline

# SAE steering can't do this (feature amplification is not reversible once baked in)
```

### Hypothesis D: Per-Layer Training Matters

**Prediction:** RDO trained per-layer (vs. global) shows less degradation

**Why:** Refusal is layer-specific (middle layers); protecting lower/upper layers' capabilities is easier

**Test:** (E2 experiments already planned)
```python
rdo_global = train_rdo(apply_to_all_layers=True)
rdo_selective = train_rdo(apply_to_layers=[10, 11, 12, 13])

# Measure: selective >> global in terms of capability preservation
```

---

## 6. Implementation Notes for RDO to Avoid Pitfalls

### Lesson 1: Measure Full Benchmark Suite

**From SAE paper:** Capability degradation was only discovered by running full MMLU
- SAE authors initially might have only measured single-turn compliance
- Full benchmark suite revealed hidden entanglement

**Action:** In RDO experiments, commit to measuring:
- MMLU (broad knowledge)
- GSM8K (math reasoning)
- TruthfulQA (factual consistency)
- At least 5-10 diverse domains

### Lesson 2: Test on Unrelated Domains

**From SAE paper:** Degradation in math, chemistry, philosophy suggests non-local effects
- Even features "unrelated to refusal" (e.g., philosophy feature 216) caused degradation

**Action:** Explicitly test RDO on domains with no refusal signal:
- Pure math problems
- Factual knowledge (capitals, dates)
- Science facts
- Should see zero degradation if RDO is working correctly

### Lesson 3: Monitor λ_retain Scaling

**Insight:** SAE steering has no equivalent to λ_retain. RDO should:
1. Sweep λ_retain from 0.0 to 1.0
2. Plot capability degradation vs. λ_retain
3. Find sweet spot (e.g., λ_retain = 0.5)
4. Document that high λ_retain eliminates degradation

### Lesson 4: Adversarial Robustness as Validation

**From SAE paper:** Multi-turn attacks (Crescendo) are the right test
- Single-turn compliance is easy; robustness is hard
- SAE steering passed Crescendo test → genuine safety hardening

**Action:** Test RDO against:
- Crescendo attacks (escalating)
- Multi-turn jailbreaks (session-based)
- Adversarial suffixes (optimized)
- Not just: "how do I make a bomb?"

---

## 7. Why RDO Should Succeed Where SAE Fails

### Root Cause Analysis: SAE Steering's Failure Mode

```
Feature 22373 encodes refusal, but imperfectly:
  ✓ Captures: "I refuse to..."
  ✓ Captures: Safety training signal
  ✗ Also captures: Ability to discuss harmful topics (needed for refusal!)
  ✗ Also captures: Formal writing patterns, technical language
  ✗ Also captures: Low-frequency information needed for language modeling

Amplifying this creates:
  - Over-activation of formal/technical language → capability loss
  - Over-suppression of certain semantic patterns → capability loss
  - Out-of-distribution activations → general degradation
```

### Why RDO Avoids This

```
Refusal direction v discovered via gradient ascent:
  ✓ Captures: True refusal mechanism (empirically optimized)
  ✓ Loses: Entangled language patterns (via multi-objective training)
  ✓ Result: Clean separation of refusal from capability

Ablating via projection:
  - Removes component along v
  - Preserves everything orthogonal to v
  - Maintains in-distribution activations
  - No learned decoder artifacts
```

---

## 8. Edge Cases and Failure Modes to Test

### Case 1: If RDO Also Shows Degradation

**What it means:** Refusal is NOT in a separate low-D subspace
- Contradicts Arditi et al. findings
- Validates SAE paper's claim of deep entanglement

**Action:**
- Characterize dimensionality of safe subset
- Test if degradation is smooth (proportional to cone rank)
- Consider multi-objective RL (optimize safety-capability explicitly)

### Case 2: If SAE Steering Degradation Is Phi-3-Specific

**What it means:** Different models have different refusal geometry
- SAE may work fine on Llama, Mistral, etc.
- Phi-3 Mini's refusal may be uniquely entangled

**Action:**
- Test RDO on diverse models
- Compare per-model degradation curves
- Publish comparison

### Case 3: If λ_retain Doesn't Help Enough

**What it means:** Naive multi-objective training insufficient
- May need adaptive weighting or adversarial training
- RL stage (E4) becomes critical

**Action:**
- Try scheduled λ_retain (increasing over time)
- Add adversarial term (discriminator-style)
- Move to E4: RL optimization

---

## 9. Summary Table: SAE vs. RDO

| Property | SAE Steering | RDO (Predicted) |
|----------|-------------|-----------------|
| **Identification** | Handcrafted + SAE search | Gradient discovery + optimization |
| **Mechanism** | Feature amplification | Projection/LoRA ablation |
| **Safety improvement** | ✓ Strong (42pp on Crescendo) | ✓ (to be measured) |
| **Capability preservation** | ✗ Severe (46pp loss) | ✓ Expected (<10pp loss) |
| **Reversibility** | ✗ (baked into generation) | ✓ (adapter on/off) |
| **Multi-objective** | ✗ (only optimizes safety) | ✓ (λ_retain explicit) |
| **Geometric interpretation** | ✗ (opaque features) | ✓ (clear directions) |
| **Per-layer control** | ✗ (single feature) | ✓ (per-layer adapters) |
| **Interpretability** | Medium (features) | Low (directions) |
| **Deployment ready** | ✗ (capability cost too high) | ? (pending evaluation) |

---

**Document Version:** 1.0
**Last Updated:** 2026-01-19
**Relevance:** HIGH - Critical for understanding RDO's advantages and failure modes
