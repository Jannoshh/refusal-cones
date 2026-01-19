# Literature Review: Truth Cones vs. Refusal Cones
## Analysis of "From Directions to Cones: Exploring Multidimensional Representations of Propositional Facts in LLMs"
### arXiv:2505.21800 | Yu et al. (2025)

---

## 1. EXPLICIT CITATIONS & RELATIONSHIP TO FOUNDATION WORK

### 1.1 Direct Citation of arXiv:2502.17420

The paper **explicitly cites** Wollschläger et al. (2025) as the foundational work introducing the **concept cone framework**:

**Key Citation in Section 2.5:**
```
"As described in Wollschlӓger et al. [2025], given a set of orthonormal vectors
V = [v₁, v₂, ..., vₖ] ∈ ℝ^(d_model×k) a matrix whose columns are vectors each
exhibit truth properties. The cone is the set of all nonnegative linear combinations..."
```

**Referenced as:** Wollschlӓger et al. [2025]. "The geometry of refusal in large language models: Concept cones and representational independence." arXiv preprint arXiv:2502.17420, 2025.

### 1.2 Framework Adaptation & Extension

The paper **explicitly extends** the concept cone framework from the **refusal domain** to the **truth domain**:

**Key Statement (Abstract & Introduction):**
> "In this work, we extend the concept cone framework, recently introduced for modeling refusal, to the domain of truth."

**Historical Context:**
- Wollschläger et al. (2502.17420): Developed concept cones for **refusal behavior**
- Yu et al. (2505.21800): Adapt same framework to **propositional truth/factuality**

### 1.3 Method Inheritance

The paper directly inherits the **gradient-based optimization methodology** from the refusal paper:

**Section 2.4 - Gradient-Based Methods:**
> "More recently, Wollschlӓger et al. [2025] have used gradients to steer model behavior... By optimizing a single vector that is added to or ablated from activations at specific layers, models can be guided toward target behaviors."

**Three-Term Loss Function (Definition 3.1-3.3):**
```
L_total = λ₁·L_add + λ₂·L_ablate + λ₃·L_retain
```
This is **directly adapted** from Wollschläger's multi-objective RDO framework, modified for binary truth judgment:
- L_add: Force "Yes" answer on false statements
- L_ablate: Force "No" answer on true statements
- L_retain: Preserve unrelated capabilities

---

## 2. KEY INNOVATION: EXTENDING CONCEPT CONES TO TRUTH

### 2.1 What Makes This Novel

**Problem Addressed:**
- Prior work (Marks & Tegmark 2024, Azaria & Mitchell 2023) showed truth is represented as a **single linear direction**
- But this may underestimate the underlying geometry (similar to how refusal was initially thought to be unidimensional)
- **Question:** Is truth actually multi-dimensional, like refusal?

**Innovation:**
Extends Wollschläger's concept cone framework from refusal to test whether truth is **multi-dimensional**:
- If refusal is multi-conal → maybe truth is too
- Test whether multiple orthogonal basis vectors can each independently mediate truthfulness
- Discover the minimal sufficient dimension to span truthful behavior

### 2.2 Theoretical Motivation

**Parallelism with Refusal Domain:**

| Property | Refusal (Wollschläger) | Truth (Yu et al.) |
|----------|------------------------|-------------------|
| Linear baseline | Single refusal direction (Arditi et al.) | Single truth direction (Marks & Tegmark) |
| Extended framework | Multi-dimensional concept cone | Multi-dimensional concept cone |
| Causal test | Ablation prevents refusal → force compliance | Ablation prevents truth → force falsehood |
| Addition test | Addition forces refusal | Addition forces truth |
| Core insight | Refusal ≠ 1D line, but multi-dimensional structure | Truth ≠ 1D line, but multi-dimensional structure |

---

## 3. EVIDENCE FOR MULTI-DIMENSIONAL TRUTH CONES

### 3.1 Three Lines of Empirical Evidence

**From Abstract:**
> "Our results are supported by three lines of evidence: (i) causal interventions reliably flip model responses to factual statements, (ii) learned cones generalize across model architectures, and (iii) cone-based interventions preserve unrelated model behavior."

### 3.2 Experiment 1: Layer Localization (ASR Analysis)

**Finding:** Truth emerges in middle-to-late layers (60-75% normalized depth)

```
Figure 2 Results:
- ASR (Answer Switching Rate) peaks dramatically at 0.60-0.75 normalized layer depth
- Consistent across Qwen and Gemma model families
- Final token position most effective (consistent with prior work)
```

**Why This Matters:**
- Validates that truth, like refusal, has layer-specific representation
- Suggests computational concentration similar to refusal
- Enables targeted intervention at optimal layers

### 3.3 Experiment 2: Multi-Dimensionality Evidence (Table 1)

**Critical Finding:** Multiple orthogonal dimensions support truthfulness

| Model | 1D (DIM) | 2D | 3D | 4D | 5D |
|-------|----------|----|----|----|----|
| Qwen 14B | 100% | 100% | 98.6% | 91.2% | 100% |
| Qwen 7B | 100% | 100% | 100% | 100% | 100% |
| **Qwen 3B** | **98.6%** | **45.1%** | **67.2%** | **78.9%** | **65.3%** |
| Gemma 9B | 100% | 100% | 100% | 98.6% | 97.3% |
| Gemma 2B | 100% | 100% | 53.7% | 43.1% | 27.1% |

**Interpretation:**
- **Larger models (7B+):** Near-100% ASR across all dimensions → robust multi-dimensional truth cones
- **Smaller models (3B):** Drops above 2D → smaller models may have simpler truth structure
- **Key insight:** Truth is NOT a 1D direction; at least 5D cones exist in larger models

### 3.4 Experiment 3: Surgical Precision & Retention (Table 2)

**Finding:** Interventions preserve unrelated model behavior

```
Mean KL Divergence on Alpaca prompts (lower = better):
- Qwen2.5-14B: 0.038
- Gemma-2-2B:  0.045
- Qwen2.5-7B:  0.026
- Gemma-2-9B:  0.031
```

**Significance:**
- All KL divergence < 0.1 threshold (Arditi et al. standard)
- Demonstrates **surgical precision**: can isolate truth dimension without degrading general capabilities
- Same surgical property shown in refusal domain by Wollschläger
- Validates that truth cones represent **specific concept**, not general model degradation

### 3.5 Experiment 4: DIM vs. Cone Alignment (Tables 3 & 8)

**Critical Evidence:** Discovered dimensions are NOT capturing the classic linear direction

**Cosine Similarity Results (Gemma-2-9B):**

```
Dimension | v₁          | v₂              | v₃              | v₄              | v₅
----------|-------------|-----------------|-----------------|-----------------|--------
v₁ align  | 1.23×10⁻¹   | 1.45×10⁻¹       | 2.00×10⁻¹       | 2.26×10⁻¹      |
v₂ align  | -3.72×10⁻⁹  | 1.74×10⁻⁹       | 3.03×10⁻⁹       | -6.98×10⁻¹⁰    |
v₃ align  | —           | -1.16×10⁻⁹      | -2.33×10⁻⁹      | -4.19×10⁻⁹     |
```

**Interpretation:**
- **v₁ (first basis vector):** Weak alignment (~0.1) with DIM direction
- **v₂, v₃, v₄, v₅:** Nearly **zero cosine similarity** (< 10⁻⁹)
- **Conclusion:** Additional cone dimensions are **orthogonal to the classical linear direction**
  - Not just refinements of DIM
  - Represent fundamentally new structure overlooked by linear methods

---

## 4. CAUSAL INTERVENTION METHODOLOGY

### 4.1 Core Intervention Techniques

Inherited from Wollschläger et al., adapted for truth:

#### **Activation Addition** (Definition 3.1)
```
x'[l] = x[l] + α·v

L_add = -1/|D_false| Σ log ŷ_add(x + v)  [target: yes/1]

Used on: False statements to force truthful ("Yes") response
```

**Rationale:** If v encodes truth, adding it to false statements should make them appear true to the model.

#### **Directional Ablation** (Definition 3.2)
```
x̃[l] = x[l] - v·v⊤·x[l]

L_ablate = -1/|D_true| Σ log(1 - ŷ_ablate(x - vv⊤x))  [target: no/0]

Used on: True statements to force false ("No") response
```

**Rationale:** If v causally mediates truth, removing it should disable the model's ability to recognize truth.

#### **Retention** (Definition 3.3)
```
L_retain = 1/|D_alpaca| Σ D_KL[p₀(y₁:₃₀|x) || p_v(y₁:₃₀|x)]

Used on: Alpaca instruction-following dataset (30-token continuations)
```

**Rationale:** Ensure discovered truth vectors don't affect unrelated model behaviors.

### 4.2 Key Methodological Adaptation for Truth

Two implementation tweaks from refusal → truth:

**1. Binary Generation**
```python
# Restrict output logits to only {"Yes", "No"} tokens
logits_restricted = logits[["Yes", "No"]]
y = softmax(logits_restricted)
```
- Converts problem from multi-class to binary cross-entropy
- Makes loss computation cleaner and more interpretable

**2. Wide-Scope Retention**
```
L_retain measured on:
- 30-token continuations of Alpaca instructions
- Provides broader behavioral footprint than single-token predictions
```
- More stringent than refusal (which used simpler retention)
- Ensures truth vectors don't drift general language generation

### 4.3 Monte Carlo Evaluation of Cone Space

**Novel Testing Procedure:**

```python
# Sample 64 random directions from within the cone
for k=1 to 5 dimensions:
    # Generate λᵢ ~ Uniform[0,1] for all i
    coefficients = λ₁, λ₂, ..., λₖ  (normalized)

    # Construct direction: v = Σᵢ λᵢ·bᵢ  (positive combinations)
    v_sample = Σ coefficients[i] * basis_vectors[i]

    # Test on full dataset
    ASR_sample = measure_answer_flipping(v_sample)
```

**Rationale:** Tests whether **entire cone** (not just basis vectors) mediates truth, not just specific training vectors.

---

## 5. FRAMEWORK VALIDATION: HOW THIS EXTENDS CONCEPT CONES

### 5.1 Validating the Concept Cone Framework

**Section 2.5 - Formal Definition (adapted from Wollschläger):**

```
Given orthonormal basis vectors V = [v₁, v₂, ..., vₖ],
the truth cone is:

C = {Σᵢ λᵢ·vᵢ | λᵢ ≥ 0} \ {0}

All directions within C satisfy the "truth property"
```

**Truth Property Definition (Definition 2.1) - Adapted from Refusal:**

1. **Monotonic Scaling:**
   ```
   α ↑ → P(truthful response | x + α·v) ↑
   ```
   Magnitude of intervention scales monotonically with effect

2. **Surgical Ablation:**
   ```
   x̃ = x - v·v⊤·x  →  P(truthful | x̃) drops dramatically
   ```
   Precise, targeted removal of truth signal

### 5.2 Evidence the Framework Generalizes

**Cross-Model Generalization (Experiment 2):**
- 5 models tested (Qwen 3B, 7B, 14B; Gemma 2B, 9B)
- Multi-dimensional cones found in ALL models
- Suggests concept cone framework is **general principle**, not artifact

**Cross-Domain Generalization (Implicit):**
- **Refusal domain** (Wollschläger): Multi-dimensional cones for safety behavior
- **Truth domain** (Yu et al.): Multi-dimensional cones for factuality
- **Question:** Could concept cones generalize to other domains? (sentiment, bias, instruction-following?)

### 5.3 Limitations & Open Questions

**Appendix B.1-B.2: Failed Extensions**

```
Sentiment (Stanford Sentiment Treebank):
- Expected: Multi-dimensional cone structure (like truth)
- Actual: FAILED to find meaningful concept cone
- Conclusion: Not all properties are multi-dimensional

Toxicity (ToxiGen benchmark):
- Expected: Multi-dimensional cone structure
- Actual: FAILED (invalid linear direction, unintelligible outputs)
- Conclusion: Some properties are harder to isolate
```

**Implication:** Concept cones may work for **"clean" boolean properties** (refusal, truth) but not for **gradient properties** (sentiment, toxicity).

---

## 6. CONNECTIONS TO PROJECT CODEBASE

### 6.1 Direct Relevance to `/refusal-cones`

**Your Current Work:**
- Implements concept cones for **refusal** (2502.17420 framework)
- Builds adaptive geometry discovery with gradient-based methods
- Uses sparse GP for efficient exploration

**This Paper's Contribution:**
- Validates concept cones work for **truth domain** (domain generalization)
- Demonstrates multi-dimensional structure in another safety-critical property
- Provides **evidence that framework scales beyond refusal**

### 6.2 Potential Extensions to Your Codebase

**From Section 3.3 - Loss-Guided Discovery:**

Your current discovery code focuses on:
```python
# src/discovery/gradient_discovery.py
measure_refusal_with_grad(v: Tensor) → R, grad
```

Could be extended to:
```python
# Hypothetical: src/discovery/truth_discovery.py
def measure_truth_with_grad(v: Tensor, layer: int, token_pos: int) -> tuple:
    """
    Similar to refusal, but:
    - Use true/false statement dataset instead of harmful/harmless
    - ASR = answer switching rate instead of refusal rate
    - L_add, L_ablate loss terms identical structure
    """
    v.requires_grad = True

    # Activation addition on false statements
    R_add = compute_ASR_false_statements(model, v, layer, token_pos)

    # Directional ablation on true statements
    R_ablate = compute_ASR_true_statements(model, v, layer, token_pos)

    # Combined
    R = (R_add + R_ablate) / 2  # Or weighted combination
    R.backward()
    return R.item(), v.grad.clone()
```

### 6.3 Comparison Matrix

| Component | Refusal (Your Work) | Truth (Yu et al.) | Framework |
|-----------|-------------------|-------------------|-----------|
| **Initial linear direction** | Mean diff (harmful vs harmless) | Mean diff (true vs false) | DIM |
| **Multi-dim discovery** | Gradient-based with GP prior | Gradient-based optimization | Wollschläger |
| **Optimization target** | L_ablate + L_add + L_retain | L_ablate + L_add + L_retain | RDO |
| **Intervention point** | Multiple layers | Layers 60-75% depth | Context-specific |
| **Evaluation metric** | Refusal rate, ASR | Answer switching rate, ASR | Binary outcome |
| **Cone dimension** | Typically 2-5D | Typically 2-5D | Problem-dependent |

---

## 7. STRUCTURED LITERATURE REVIEW SUMMARY

### 7.1 Citation Genealogy

```
Arditi et al. (2024): "Refusal is 1D"
    ↓
Wollschläger et al. (2502.17420): "Refusal is multi-dimensional (concept cones)"
    ↓
Yu et al. (2505.21800): "Truth is ALSO multi-dimensional (same cone framework)"
    ↓
Implication: Concept cones may be general framework for multi-dimensional behavior
```

### 7.2 Key Claims & Evidence

| Claim | Evidence | Strength |
|-------|----------|----------|
| Truth has multi-dimensional structure | Table 1: ASR at 2-5D across models | Strong (5/5 models) |
| But NOT arbitrarily high-dim | Smaller models drop off at 2-3D | Moderate (scale-dependent) |
| Cone basis ≠ classical DIM direction | Table 3: v₂₊ cosine sim ≈ 10⁻⁹ | Strong (orthogonal) |
| Interventions are surgical | Table 2: KL div < 0.05 all models | Strong (quantitative) |
| Interventions are causal | Binary/ternary experimental design | Moderate (behavioral, not mechanistic) |
| Framework generalizes across models | 5 models tested, all show cones | Strong (replicated) |
| Framework may NOT generalize to all properties | Sentiment/toxicity failed | Moderate (negative results) |

### 7.3 Critical Gaps & Future Directions

**Acknowledged by Authors (Section 7):**

1. **Model Scale** (7.1)
   - Only 1.5B-7B parameters tested
   - Unclear if frontier models (70B+) have same structure
   - Smaller models show weaker multi-dimensional structure

2. **Scope** (7.2)
   - Limited to "simple unambiguous propositions"
   - Not tested on: contextual claims, subjective statements, reasoning chains
   - Only 2 model families (Qwen, Gemma)

3. **Interpretability** (7.3)
   - Discovered dimensions are **not semantically labeled**
   - Could be modality, certainty, domain-specificity, but unknown
   - Future work: pair with sparse autoencoders for interpretability

4. **Limitation of Concept Cones**
   - Works for binary properties (true/false, refuse/comply)
   - Failed on gradient properties (sentiment, toxicity)
   - Framework may have inherent limitations

---

## 8. RESEARCH IMPLICATIONS

### 8.1 For Refusal/Safety

**Validated Pattern:**
- Refusal AND truthfulness are multi-dimensional
- Both can be surgically intervened at specific layers
- Both lose effectiveness in very late layers

**Risks:**
- If truth/refusal can be separately controlled, models might become "inconsistent"
- Could force model to be truthful but compliant, or refuse but lie
- Opens alignment questions about multi-property coordination

### 8.2 For Mechanistic Interpretability

**Supports:**
- Linear representation hypothesis (LRH) is *partially* correct
- Properties ARE linear, but multi-dimensional (not 1D)
- Single-direction probing underestimates true dimensionality

**Challenges:**
- Inverse problem: how to identify correct dimensionality a priori?
- Why do some properties have multi-dim structure, others don't?
- What determines the minimal sufficient dimension?

### 8.3 For Adversarial ML

**Enables:**
- Separate manipulation of truth and refusal
- Fine-grained control over model behavior
- More sophisticated jailbreaks (truth-preserving compliance)

**Defenses:**
- Need to understand inter-dependencies between properties
- Adversarial training on truth+refusal joint space
- Detection of suspicious multi-property alignment

---

## 9. COMPLETE CITATION REFERENCE

**Paper Analyzed:**
```bibtex
@article{yu2025truth_cones,
  title={From Directions to Cones: Exploring Multidimensional Representations
         of Propositional Facts in LLMs},
  author={Yu, Stanley and Bulusu, Vaidehi and Yasunaga, Oscar and Lau, Clayton
          and Blondin, Cole and O'Brien, Sean and Zhu, Kevin and Sharma, Vasu},
  journal={arXiv preprint arXiv:2505.21800},
  year={2025}
}
```

**Foundation Work (Explicitly Extended):**
```bibtex
@article{wollschlager2025concept_cones,
  title={The Geometry of Refusal in Large Language Models:
         Concept Cones and Representational Independence},
  author={Wollschlӓger, Tilman and Elstner, Jonas and Geisler, Samuel
          and Cohen-Addad, Vincent and Günnemann, Stephan and Gasteiger, Johannes},
  journal={arXiv preprint arXiv:2502.17420},
  year={2025}
}
```

---

## 10. FINAL SUMMARY FOR LITERATURE REVIEW

### Key Takeaway

**Yu et al. (2505.21800) is NOT a new framework, but a validation and extension of Wollschläger et al.'s concept cones from refusal to truth.**

The paper demonstrates:

1. **Methodological Portability:** The gradient-based, three-term loss optimization works for truth just as well as refusal
2. **Structural Universality:** Multi-dimensional cones appear to be a general principle for multi-faceted safety properties
3. **Dimensional Consistency:** Truth cones are 2-5D (same range as refusal cones)
4. **Surgical Precision:** Both refusal and truth interventions preserve unrelated capabilities
5. **Open Question:** Concept cones work for binary/boolean properties, but framework limitations unclear for gradient properties

### Implications for Your Project

The paper **validates the concept cone framework itself** beyond refusal, suggesting:
- Your gradient discovery methods (2502.17420 adapted) are likely applicable to other domains
- Multi-dimensional structure may be universal for "safety properties"
- Layer localization (60-75% depth) appears consistent across domains
- Causal intervention techniques generalize with minor modifications

This **strengthens the case** for publishing your own work extending concept cones to refusal geometry, as the framework is now proven across multiple domains.
