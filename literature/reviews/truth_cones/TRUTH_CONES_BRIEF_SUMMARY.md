# Brief Summary: Yu et al. (2505.21800) - Truth Cones

## Paper Core

**Title:** From Directions to Cones: Exploring Multidimensional Representations of Propositional Facts in LLMs

**Contribution:** Extends Wollschläger et al. (2502.17420)'s concept cone framework from refusal to truth domain.

---

## Citation of Foundation Work

Yu et al. **explicitly build on** arXiv:2502.17420:

1. **Method inheritance:** Uses identical three-term loss optimization:
   ```
   L_total = λ₁·L_add + λ₂·L_ablate + λ₃·L_retain
   ```

2. **Framework adoption:** "We extend the concept cone framework, recently introduced for modeling refusal, to the domain of truth" (Abstract)

3. **Definition inheritance:** Adapts Wollschläger's "truth properties" (monotonic scaling, surgical ablation)

---

## Key Innovation: Multi-Dimensional Truth

**Central Finding:** Truth is NOT a 1D direction (contra Marks & Tegmark 2024), but a multi-dimensional cone.

**Evidence:**

| Model | 1D | 2D | 3D | 4D | 5D |
|-------|----|----|----|----|-----|
| Qwen 7B | 100% | 100% | 100% | 100% | 100% |
| Gemma 9B | 100% | 100% | 100% | 98.6% | 97.3% |

Table 1: Answer Switching Rate (ASR) across cone dimensions. All larger models maintain >95% ASR at 5D.

---

## Three Lines of Evidence

### 1. Causal Interventions Work
- **Activation addition** on false statements → forces "Yes" answer
- **Directional ablation** on true statements → forces "No" answer
- Achieves near-100% answer switching in larger models

### 2. Cones Generalize Across Models
- Tested 5 models (Qwen: 3B, 7B, 14B; Gemma: 2B, 9B)
- Multi-dimensional cones found in all
- Consistent layer localization (60-75% depth)

### 3. Interventions Are Surgical
Mean KL divergence on unrelated tasks (Alpaca):
- All models < 0.05 (threshold: 0.1)
- Proves truth dimensions don't affect general capabilities

---

## Validation of Cone Framework

### Layer Localization (Fig 2)
Truth effectiveness peaks at 60-75% normalized layer depth - **identical pattern to refusal**

### Orthogonality to Linear Direction (Table 3)
Cosine similarity between cone basis vectors and classical DIM direction:
- v₁: ~0.1 (weak alignment)
- v₂₊: < 10⁻⁹ (orthogonal)

**Conclusion:** Additional cone dimensions capture structure **missed by linear methods**, not refinements of DIM.

### Monotonic Scaling
ASR scales monotonically with intervention magnitude α - satisfies formal truth property.

---

## Causal Methodology

**Binary truth judgement setup:**
1. Restrict logits to {"Yes", "No"} tokens
2. Formulate as binary cross-entropy optimization
3. Sample 64 random directions from cone interior
4. Measure answer switching on all sampled vectors

**Key adaptation from refusal:** Wide-scope retention on 30-token continuations (not just single-token) ensures robustness.

---

## Where Concept Cones Fail

**Appendix B.1-B.2: Negative Results**
- **Sentiment (SST):** No valid concept cone found
- **Toxicity (ToxiGen):** Failed to identify baseline direction

**Implication:** Concept cones generalize to binary/boolean properties (refusal, truth) but NOT gradient properties (sentiment, toxicity).

---

## Limitations

1. **Model scale:** Only 1.5B-7B tested; larger models untested
2. **Narrow scope:** Limited to simple, unambiguous propositions
3. **No interpretability:** Basis vectors lack semantic labels
4. **Limited generalization:** Only Qwen & Gemma families

---

## How This Extends Your Work

| Aspect | Refusal (Your Research) | Truth (This Paper) |
|--------|-------------------------|-------------------|
| **Foundation** | Wollschläger (2502.17420) | Yu et al. (2505.21800) |
| **Initial direction** | Mean diff method | Linear probe / DIM |
| **Multi-dim structure** | Gradient-based discovery + GP | Gradient optimization |
| **Dimension range** | 2-5D | 2-5D |
| **Layer localization** | Network-dependent | 60-75% normalized depth |
| **Intervention precision** | KL div < 0.05 | KL div < 0.05 |

**Validation:** Yu et al. proves that multi-dimensional concept cones are a **general framework** applicable across safety properties, strengthening the case for your refusal cone research.

---

## Relevance Summary

✓ **Directly cites** Wollschläger 2502.17420
✓ **Extends** concept cone framework to new domain
✓ **Validates** that multi-dimensional structure is general principle
✓ **Identifies** constraints (works for binary properties, not gradients)
✓ **Shares** methodology (three-term loss, Monte Carlo evaluation)
✓ **Provides** evidence for framework generalization beyond refusal
