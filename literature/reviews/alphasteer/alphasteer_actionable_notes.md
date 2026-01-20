# AlphaSteer: Actionable Notes for Your Research

## Quick Reference: What This Means for Refusal-Cones

### 1. Immediate Next Steps (This Month)

#### A. Implement AlphaSteer as Baseline
**Why:** Enable direct comparison on same benchmarks; validate your multi-modal geometry advantage

**Tasks:**
- [ ] Extract null-space projection implementation from AlphaSteer paper
- [ ] Compute P̂ = I - H_b^⊤(H_b H_b^⊤)^(-1)H_b for test models
- [ ] Test on same 7 jailbreaks (AIM, AutoDAN, Cipher, GCG, Jailbroken, PAIR, ReNeLLM)
- [ ] Measure utility on AlpacaEval, XSTest, GSM8K, MATH500
- [ ] Create benchmark comparison table: AlphaSteer vs your RDO

**Expected result:** Establish baseline; show if your approach beats AlphaSteer

#### B. Validate Single-Direction Assumption
**Why:** AlphaSteer assumes refusal is single linear direction; your geometry paper suggests otherwise

**Questions to answer:**
1. How well does Arditi's mean-difference R explain refusal variance?
2. What fraction of refusal can be attributed to single dominant mode vs multiple modes?
3. On attacks AlphaSteer fails on, does multi-mode help?

**Experiment:**
```python
# Compute explained variance
r_arditi = mean(H_compliant) - mean(H_refused)  # single direction
r_arditi_normalized = r_arditi / ||r_arditi||

# On each attack, measure:
# 1. Single-direction effectiveness: score using just r_arditi
# 2. Residual directions: H_attack - (H_attack · r_arditi)r_arditi
# 3. Correlation: can residuals explain remaining variance?

# If residuals are small → single-direction is fine (AlphaSteer wins)
# If residuals are large → multi-mode geometry is needed (RDO wins)
```

**Output:** Quantitative evidence showing where single-direction breaks down

---

### 2. Short-term (Next 3 Months): Build Competitive Advantage

#### A. Null-Space Constraints + Multi-Modal Geometry
**Idea:** Keep AlphaSteer's principled null-space constraint, but for MULTIPLE modes

**Implementation:**
```python
class MultiModalNullSpaceRDO:
    """RDO with null-space constraints for each discovered mode"""

    def __init__(self, benign_activations, discovered_modes):
        # Compute single null-space (shared for all modes)
        H_b = benign_activations
        self.P_null = compute_null_space_projection(H_b)

        # Store discovered modes
        self.modes = discovered_modes  # List of mode vectors

    def constrained_steering(self, mode_weights):
        """
        Combine modes while respecting null-space constraint

        Δ = Σ_k α_k v_k
        where each (α_k * P_null) ensures zero effect on benign
        """
        steering = torch.zeros(self.hidden_dim)
        for k, (alpha, v_k) in enumerate(zip(mode_weights, self.modes)):
            # Project mode into null space
            v_k_constrained = self.P_null @ v_k
            steering += alpha * v_k_constrained

        return steering  # Guaranteed: steering @ H_b = 0
```

**Benefits:**
- Combines AlphaSteer's utility guarantee with multi-modal structure
- Mathematical rigor: null-space property proven for mode combinations
- Novel contribution: "multi-modal steering with null-space constraints"

**Expected result:** Better safety than AlphaSteer (more flexible), guaranteed utility preservation

#### B. Discover Geometry via Gradient Optimization
**Idea:** Use AlphaSteer as initialization, then discover modes with RDO

**Workflow:**
```
Step 1: Run AlphaSteer
  → Get baseline steering Δ_alpha with 91.93% DSR
  → Use as warm-start for RDO

Step 2: Run gradient discovery from Δ_alpha
  → Search for additional modes/directions
  → Should converge faster due to better initialization

Step 3: Train multi-modal RDO
  → Optimize for DSR + utility + multiple modes
  → Use null-space constraints for each mode

Result: "AlphaSteer + Geometry Discovery"
```

**Metric:** Convergence speed comparison
- From random init: 50-70 measurements (RDO paper)
- From AlphaSteer init: target <30 measurements
- Speedup factor: 2-3×

#### C. Analytical Validation
**Idea:** Prove that multi-modal steering with null-space preserves utility

**Theorem to prove:**
```
If P̂ is null-space projection for benign activations H_b,
and v_1, v_2, ..., v_k are discovered modes,
then for any α_1, α_2, ..., α_k ≥ 0:
  (Σ_i α_i v_i) · h_b = 0  for all h_b ∈ H_b
implies: steering = Σ_i α_i P̂ v_i has zero effect on benign
```

**Why this matters:**
- Extends AlphaSteer's theoretical guarantee to multiple modes
- Makes RDO's multi-modal approach as principled as AlphaSteer
- Publishable theoretical contribution

---

### 3. Medium-term (3-6 Months): Differentiation

#### A. Show Multi-Modal Advantage
**Experiment:** Cases where AlphaSteer fails but RDO succeeds

**Design:**
```
For each of 7 jailbreak attacks:
  1. Run AlphaSteer → get DSR_alpha
  2. Run RDO → get DSR_rdo
  3. Where RDO > AlphaSteer:
     a. Extract residual direction (what AlphaSteer missed)
     b. Show it aligns with discovered secondary mode
     c. Interpret: "Attack exploits direction orthogonal to primary mode"

Expected: Find 1-3 attacks where multi-modal helps significantly
Narrative: "Single-direction assumption too restrictive; discovered geometry explains gap"
```

**Table to create:**
```
Attack     | AlphaSteer | RDO | Improvement | Explanation
-----------|------------|-----|-------------|-------------
AIM        | 100%       | 100%| 0%          | Single mode sufficient
AutoDAN    | 99%        | 99% | 0%          | Single mode sufficient
Cipher     | 63%        | 88% | +25%        | Discovered mode 2 critical
GCG        | 97%        | 98% | +1%         | Marginal improvement
...
```

**Key insight:** Where do secondary modes matter?

#### B. RL Optimization Stage
**Idea:** After discovering geometry + training RDO, apply RL to push further

**Approach:**
```
Phase 1: Gradient discovery + RDO training
  → Converge to local optimum with discovered geometry

Phase 2: RL stage (following your existing approach)
  → Policy gradient to maximize: ASR - β * utility_loss
  → Can jointly optimize all discovered modes

Expected: Additional 5-10% DSR improvement
Narrative: "RDO + RL combines geometric understanding with agent learning"
```

#### C. Cross-Model Validation
**Claim:** Discovered geometry should generalize

**Experiment:**
```
Train on: Llama-3.1-8B
Evaluate on: Qwen2.5-7B, Gemma-2-9b

For each model:
  1. Discover modes on MODEL A
  2. Transfer modes to MODEL B (no retraining)
  3. Measure DSR/utility

Hypothesis: Modes should partially transfer (refusal geometry is universal)
Expected: >70% DSR on MODEL B using modes discovered from MODEL A
```

**If successful:** "Discovered refusal geometry transfers across models"

---

### 4. Long-term (6+ Months): Research Contributions

#### A. Theoretical Paper: Multi-Modal Null-Space Steering
**Title:** "Principled Multi-Modal Steering: Null-Space Constraints for Adaptive Refusal"

**Contributions:**
1. Extend AlphaSteer's single-direction theory to multi-modal case
2. Prove utility preservation for mode combinations
3. Characterize when multi-modal is necessary (geometric complexity measure)

**Structure:**
- Section 1: AlphaSteer's null-space principle (review)
- Section 2: Extension to multi-modal (new theory)
- Section 3: Geometric complexity measures (new)
- Section 4: When does multi-modal help? (empirical)

#### B. Empirical Paper: From Geometry to Steering
**Title:** "RDO: Geometry-Aware Refusal Steering via Adaptive Discovery and Multi-Objective Training"

**Contributions:**
1. Show discovered geometry improves over single-direction baselines
2. Demonstrate null-space constraints extend to multi-modal
3. RL stage finds better Pareto fronts
4. Cross-model evaluation of discovered geometry

**Comparison section:**
- AlphaSteer: single-direction, closed-form, efficient → strong baseline
- RDO: multi-modal, learned, powerful → your approach
- Trade-offs: efficiency vs. flexibility, theory vs. empirics

#### C. Position as Extension, Not Replacement
**Frame:** "AlphaSteer is the strong single-direction baseline; RDO is the geometry-aware extension"

**Key quotes for paper:**
```
"AlphaSteer demonstrates that principled null-space constraints achieve
excellent safety-utility balance under the single-direction assumption
[Sheng et al. 2026]. However, our geometry discovery reveals this
assumption is overly restrictive: refusal exhibits multi-modal structure
across layers and prompt types. We extend AlphaSteer's theoretical
foundation to multi-modal case via null-space constraints on discovered
modes, achieving better safety without compromising utility."
```

---

## 5. Specific Technical Recommendations

### A. Null-Space Implementation
**What to implement:**
```python
def compute_null_space_projection(H_b: torch.Tensor) -> torch.Tensor:
    """
    Compute null-space projection matrix P̂

    Args:
        H_b: [d, N_b] matrix of benign activations

    Returns:
        P̂: [d, d] projection matrix where P̂ @ v is in null space
    """
    # Following AlphaSteer Eq. 5-6
    d = H_b.shape[0]

    # Compute H_b H_b^T
    HHT = H_b @ H_b.T  # [d, d]

    # Add regularization for numerical stability
    HHT += 1e-6 * torch.eye(d, device=H_b.device)

    # Compute inverse
    HHT_inv = torch.linalg.inv(HHT)

    # Projection matrix
    P = torch.eye(d, device=H_b.device) - H_b.T @ HHT_inv @ H_b

    return P
```

**Testing:**
```python
# Verify property: P @ h_b = 0 for all h_b in H_b
residual = P @ H_b  # Should be ~0
assert torch.allclose(residual, torch.zeros_like(residual), atol=1e-5)

# Verify idempotence: P @ P = P
assert torch.allclose(P @ P, P, atol=1e-5)
```

### B. Comparison Script
**Create benchmark script:**
```python
def compare_alphasteer_vs_rdo():
    """
    Compare AlphaSteer baseline vs RDO on standard benchmarks
    """

    # 1. Extract refusal vector
    r = extract_refusal_direction(model, harmful_prompts, benign_prompts)

    # 2. Compute null-space projection
    P = compute_null_space_projection(benign_activations)

    # 3. Run AlphaSteer
    alphasteer_steering = train_alphasteer(
        H_m=malicious_activations,
        H_b=benign_activations,
        r=r,
        P=P,
        alpha=0.01  # Regularization
    )

    # 4. Run RDO discovery
    discovered_modes = discover_geometry(
        v_init=r,
        measure_fn=measure_refusal_with_grad,
        config=GradientDiscoveryConfig(...)
    )

    # 5. Run RDO training with null-space constraint
    rdo_steering = train_rdo_with_null_space(
        modes=discovered_modes,
        P=P,
        objectives={'ablate': 1.0, 'add': 1.0, 'retain': 0.5}
    )

    # 6. Evaluate both
    results = {
        'alphasteer': evaluate(model, alphasteer_steering, jailbreaks, utility_tasks),
        'rdo': evaluate(model, rdo_steering, jailbreaks, utility_tasks)
    }

    return results
```

### C. Measurement Consistency
**Make sure to measure:**
1. **DSR** (Defense Success Rate): % of jailbreak attacks prevented
   - Same 7 attacks as AlphaSteer (AIM, AutoDAN, Cipher, GCG, Jailbroken, PAIR, ReNeLLM)
   - Same 100 harmful questions from AdvBench
2. **Utility**: Same benchmarks
   - AlpacaEval (instruction following)
   - XSTest (safe questions)
   - GSM8K (math)
   - MATH500 (symbolic math)
3. **Activation dynamics**: PCA visualizations
   - Show benign activations unchanged
   - Show malicious activations shift to refusal

---

## 6. Publication Strategy

### Phase 1: Establish Baseline (Month 1-2)
- [ ] Implement AlphaSteer reproduction
- [ ] Match their reported numbers (91.93% DSR on Llama)
- [ ] Validate on Qwen and Gemma models
- **Output:** Baseline paper with AlphaSteer comparison

### Phase 2: Add Geometry (Month 3-4)
- [ ] Show discovered geometry differs from single-direction
- [ ] Prove multi-modal steering preserves utility
- [ ] Demonstrate cases where multi-modal helps
- **Output:** "Geometry-Aware Refusal Steering" paper

### Phase 3: Full RDO + RL (Month 5-6)
- [ ] Add RL optimization stage
- [ ] Cross-model transfer evaluation
- [ ] Final comparison: AlphaSteer vs RDO vs All-Methods
- **Output:** Full RDO paper

### Timeline
```
Month 1: Baseline + geometry discovery validation
Month 2: AlphaSteer implementation + benchmark
Month 3: Multi-modal steering theory + proof
Month 4: RDO training + RL stage
Month 5: Cross-model evaluation + ablations
Month 6: Paper writing + submission prep
```

---

## 7. Key Talking Points for Writing

### AlphaSteer's Contributions (Acknowledge)
1. ✓ First to use null-space projection for utility preservation
2. ✓ Achieved best safety-utility tradeoff vs. baselines (91.93% DSR)
3. ✓ Mathematically principled (vs heuristic prior methods)
4. ✓ Closed-form solution (very efficient)
5. ✓ Comprehensive evaluation (3 models × 7 attacks × 4 utility metrics)

### Why RDO Is Different (Positions Your Work)
1. ✓ Discovered geometry suggests multi-modal structure (beyond single-direction assumption)
2. ✓ Adaptive refusal vectors via gradient-based discovery
3. ✓ Multi-objective optimization jointly trains safety + utility + compliance
4. ✓ Extends null-space principle to multi-modal case (theoretical novelty)
5. ✓ RL stage finds better Pareto fronts

### Claimed Advantages
- Better DSR on complex attacks (where single-direction fails)
- Utility preservation guaranteed by null-space (like AlphaSteer)
- Geometry-informed initialization (faster convergence)
- Transferable across models (geometric universality)

---

## 8. Potential Pitfalls & Mitigations

### Risk 1: "AlphaSteer is already good enough"
**Mitigation:**
- Find specific attacks/scenarios where it fails
- Show your multi-modal approach fixes those cases
- Quantify improvement: e.g., +12% DSR on complex attacks

### Risk 2: "RDO is just more complex, not better"
**Mitigation:**
- Prove complexity is necessary (geometric argument)
- Show null-space constraint holds for multi-modal
- Demonstrate RL stage adds meaningful improvement

### Risk 3: "Single-direction might actually be optimal"
**Mitigation:**
- Test on very large prompt sets (if single-direction, should work)
- Cross-model evaluation (if universal, should transfer)
- If RDO doesn't outperform AlphaSteer, reframe as "computational study"

### Risk 4: "AlphaSteer paper will be published first, scoop you"
**Mitigation:**
- This paper is preprint (arXiv 2506.07022, current date Jan 2026)
- Assumed not yet in proceedings
- Your RDO can cite it, build on it
- Frame as "response to AlphaSteer's limitations"

---

## 9. Collaborative Opportunities

### With AlphaSteer Authors
- **Idea 1:** Jointly extend null-space theory to multi-modal
- **Idea 2:** Benchmark RDO against AlphaSteer on same hardware
- **Idea 3:** Combine AlphaSteer init + RDO optimization (published collaboration)

### With Geometry Paper Authors (Your Co-authors?)
- Leverage your existing "Geometry of Refusal" paper
- This is cited by AlphaSteer ([14] in references)
- Show how your geometry insights improve upon AlphaSteer

---

## Summary: 30-Second Elevator Pitch

**AlphaSteer achieves great safety-utility tradeoff using principled null-space constraints, but assumes single-direction refusal. Your RDO work extends this by discovering multi-modal geometric structure and training adaptive steering vectors while preserving AlphaSteer's utility guarantee through null-space projection. This achieves better safety on complex attacks without losing the theoretical rigor of the baseline.**

---

## Checklist for Next Actions

- [ ] Read full AlphaSteer paper (you have 3 summaries now)
- [ ] Implement null-space projection module
- [ ] Run AlphaSteer baseline on your test models
- [ ] Measure single-direction assumption validity
- [ ] Plan multi-modal extension experiment
- [ ] Schedule implementation (target: 2-4 weeks)
- [ ] Document all comparison metrics
- [ ] Draft paper skeleton positioning RDO vs AlphaSteer
- [ ] Set up benchmark script for continuous evaluation
- [ ] Plan publication timeline

