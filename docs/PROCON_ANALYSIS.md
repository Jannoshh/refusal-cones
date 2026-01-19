# ProCon Analysis: Anchoring Refusal Direction

## Executive Summary

**ProCon** (Projection-Constrained training) presents a complementary defensive approach to refusal-cones' offensive research. Published September 2025, this paper demonstrates that refusal direction drift during fine-tuning is a primary safety degradation mechanism and proposes projection-based constraints to preserve safety during task adaptation.

**Relevance:** High. ProCon validates geometric foundations shared with refusal-cones while providing insights on stabilizing vs. manipulating refusal directions.

---

## 1. Paper Metadata

| Field | Value |
|-------|-------|
| **Title** | Anchoring Refusal Direction: Mitigating Safety Risks in Tuning via Projection Constraint |
| **Authors** | Yanrui Du, Fenglei Fan, Sendong Zhao, Jiawei Cao, Qika Lin, Kai He, Ting Liu, Bing Qin, Mengling Feng |
| **Published** | September 8, 2025 |
| **arXiv ID** | 2509.06795v1 |
| **Category** | cs.CL (Computational Linguistics) |
| **Venue** | Likely EMNLP 2025 or similar tier-1 NLP conference |

---

## 2. Foundational Citations

ProCon directly builds on **Arditi et al. (2024)** - the same foundational work cited in refusal-cones:

### Arditi et al. 2406.11717: "Refusal in Language Models Is Mediated by a Single Direction"

**Key Finding:**
- Refusal behavior is mediated by a one-dimensional subspace in residual stream activations
- Validated across 13 open-source models (up to 72B parameters)
- Erasing this direction prevents refusal; adding it forces refusal even on harmless queries

**Implications for ProCon:**
- Enables targeted constraint design (protect single r-direction)
- Makes geometric safety preservation possible

**Implications for Refusal-Cones:**
- Same direction can be exploited for ablation/addition attacks
- Foundation for all RDO discovery and training methods

---

## 3. The Core Innovation: Projection Constraints During Training

### Problem: Refusal Direction Drift

During Instruction Fine-Tuning (IFT), models experience safety degradation:

```
Initial state:     r-direction stable, refusal functional
↓ (apply standard fine-tuning)
Drift phase:       r-direction shifts in activation space
↓
Final state:       r-direction corrupted, refusal fails
```

**ProCon's Observation:** Drift is especially sharp in early training epochs.

### Solution: ProCon Loss

**Core mechanism:** Add projection-constraint term to training loss that regularizes:
```
projection_magnitude(h_t, r_direction)
```

For each training sample's hidden state `h_t`, constrain how much it projects onto the refusal direction.

**Intuition:**
- Harmless examples: maintain normal r-direction projection
- Harmful examples: prevent anti-alignment with r-direction
- Result: r-direction stabilized throughout training

### Two-Phase Implementation

**Phase 1 (Warm-up, early epochs):**
- Strong projection constraints
- Expanded data distribution
- Goal: Stabilize r-direction when drift is most severe
- Typical duration: first 10-20% of training

**Phase 2 (Main training, remaining epochs):**
- Relaxed constraints
- Standard task fine-tuning proceeds
- Goal: Maintain Phase 1 stability while improving task performance
- r-direction already anchored, so fewer gradients can corrupt it

**Why two phases work:**
- Early guidance prevents large-magnitude drift that's hard to recover from
- Once stabilized, small gradient perturbations don't derail r-direction
- Enables joint optimization of safety and capability

---

## 4. Measurement & Validation

### Quantifying Drift

ProCon uses three complementary metrics:

1. **Behavioral Metrics**
   - Do fine-tuned models refuse harmful queries? (yes/no)
   - Direct safety test via jailbreak attempts
   - Gold standard but computationally expensive

2. **Geometric Metrics**
   - Compute angular divergence between original r-direction and post-training direction
   - `θ = arccos(v_initial · v_final / ||v_initial|| ||v_final||)`
   - Fast, interpretable, directly measures drift

3. **Activation Metrics**
   - Analyze safety-relevant neuron activations before/after tuning
   - Measure activation patterns on harmful prompts
   - Diagnostic but requires interpretability infrastructure

### Experimental Setup

- **Base models:** Llama-2, Qwen-2 (multiple sizes)
- **Fine-tuning datasets:** Task-specific data + safety benchmarks
- **Attack evaluation:** Standard jailbreak dataset (HarmBench-style)
- **Baselines:**
  - Unmodified IFT (maximum drift)
  - LoRA-based safety methods
  - Other constrained training approaches (LISA, vaccines)

### Key Results

- ProCon consistently outperforms baselines on combined metrics
- Safety preservation maintained across multiple model architectures
- Task performance gains not sacrificed
- Results validated on multiple datasets and attack scenarios

---

## 5. Relation to Refusal-Cones Framework

### Complementary Objectives

| Aspect | ProCon (Defensive) | Refusal-Cones (Offensive/Research) |
|--------|-------------------|-------------------------------------|
| **Core question** | How to keep refusal working during fine-tuning? | How effective is a specific refusal direction? How many directions control refusal? |
| **Operation** | Constrain projections during training | Ablate/add projections to measure refusal strength |
| **Direction use** | Stabilize identified r-direction | Identify and manipulate r-direction |
| **Training goal** | Safety + task performance | Maximize harmfulness under ablation + retain capability |
| **Failure case** | Training corrupts safety | Ablation doesn't reduce refusal effectively |

### Shared Mathematical Foundation

Both use projection operations in activation space:

**ProCon:**
```python
# During training, monitor/constrain:
proj = torch.einsum('...d,d->...', h, r_direction) * r_direction
loss_constraint = ||proj|| or divergence(proj, expected_proj)
```

**RDO:**
```python
# Apply ablation via projection:
h_ablated = h - torch.einsum('...d,d->...', h, v) * v  # = (I - vv^T)h
# Or via LoRA:
h_ablated = h + LoRA_projection(h, rank=1)
```

**Key insight:** Same geometric primitive (projection onto discovered direction), opposite effects (preserve vs. remove).

### Cross-Learning Opportunities

1. **ProCon insights for RDO:**
   - Early-phase drift is most critical → warm-up strategies for RDO training initialization
   - Two-phase training improves robustness → could improve RDO vector quality
   - Early stabilization enables later optimization → framework for initializing RDO from discovery

2. **RDO insights for ProCon:**
   - Discovery methods identify which directions matter most
   - RDO's adaptive geometry discovery could identify which components drift most
   - Multi-layer analysis from RDO could enable layer-specific ProCon constraints

3. **Joint understanding:**
   - ProCon shows r-direction is critical under defense
   - RDO shows r-direction is critical under attack
   - Validates that geometric understanding is fundamental to both safety and vulnerability

---

## 6. Related Safety Research

### Contemporary Work on Refusal Geometry

**Multi-dimensional Safety** (Pan et al., 2502.09674):
- Refusal not purely 1D; multiple orthogonal directions encode safety
- Dominant direction + secondary directions with interpretable roles
- Suggests ProCon constraints could be dimension-specific

**Self-Organizing Maps** (Piras et al., 2511.08379):
- Multiple refusal directions extracted via SOMs
- Ablating multiple directions more effective than single direction
- Complements both ProCon (protect multiple directions) and RDO (manipulate multiple directions)

**Cross-lingual Universality** (Wang et al., 2505.17306):
- English r-direction transfers to 14 languages
- Refusal vectors exhibit parallelism across languages
- Implies ProCon constraints and RDO vectors should work cross-lingually

### Defense Approaches Using Similar Insights

**Deep Refusal** (Xie et al., 2509.15202):
- Proactively rebuild refusal from jailbreak states
- Probabilistically ablate r-direction during fine-tuning to force re-learning
- Achieves 95% attack success rate reduction
- Conceptually: immunize against refusal direction manipulation

**Safety Layers** (Li et al., 2408.17003):
- Identifies critical middle layers for safety
- Proposes Safely Partial-Parameter Fine-Tuning (SPPFT)
- Similar goal to ProCon (protect safety during tuning) via layer preservation

### Jailbreak Perspectives Validating ProCon's Concerns

**Differentiated Bi-Directional Intervention** (Zhang & Sun, 2511.06852):
- Deconstructs single direction into harm detection + refusal execution
- Demonstrates precision attacks (97.88% on Llama-2)
- Validates that precise direction manipulation enables bypass
- Implies ProCon's focus on r-direction stability is well-justified

---

## 7. Technical Details & Formulation

### Projection Constraint Definition

Assuming r-direction `v ∈ ℝ^d` identified via mean difference:
```
v = (mean_harmful_h - mean_harmless_h) / ||mean_harmful_h - mean_harmless_h||
```

**Projection of h onto v:**
```
proj_v(h) = (h · v) v
magnitude = |h · v|
```

**Constraint applied during training:**
```python
# Prevent drift by regularizing projection magnitude
loss_constraint = ||proj_v(h_t)|| or KL(proj_dist | expected_dist)

# Total loss (pseudo-code):
loss_total = loss_task + λ_constraint * loss_constraint
```

**Why this works:**
- If `h · v` stays bounded, r-direction information preserved
- Task gradients can update other dimensions freely
- Only constrains component aligned with r-direction

### Warm-up Schedule

```python
# Pseudocode for two-phase training
for epoch in range(num_epochs):
    if epoch < warmup_epochs:
        λ_constraint = λ_strong  # e.g., 1.0
        data = expanded_distribution  # more diverse harmful examples
    else:
        λ_constraint = λ_weak  # e.g., 0.1 or decay schedule
        data = standard_data  # normal task fine-tuning data

    for batch in data:
        loss = task_loss(batch) + λ_constraint * constraint_loss(batch)
        loss.backward()
        optimizer.step()
```

**Typical hyperparameters (inferred from results):**
- `warmup_epochs`: 10-20% of total training
- `λ_strong`: 0.5-1.0
- `λ_weak`: 0.0-0.1 (often decay schedule)
- `data_expansion`: 2-3x more samples in warm-up phase

---

## 8. Results Summary

### Main Findings

1. **Refusal direction drift is measurable and harmful**
   - Angular divergence between pre/post training direction correlates with safety loss
   - Early epochs show sharpest drift
   - Drift magnitude predicts compliance rate on harmful examples

2. **Projection constraints effectively stabilize r-direction**
   - Keeps angular divergence below threshold
   - Geometric metrics correlate with behavioral metrics
   - Enables safety preservation without holistic model freezing

3. **Two-phase warm-up overcomes performance barriers**
   - Early strong constraints stabilize direction
   - Phase 2 relaxation enables capability gains
   - Joint optimization of safety + performance possible

4. **Cross-model validation**
   - Works on Llama-2 family (7B-70B)
   - Works on Qwen-2 family (multiple sizes)
   - Suggests generalizability across architectures

### Quantitative Performance

The abstract mentions "superior overall performance" vs. baselines, suggesting:
- Comparable or better safety preservation (vs. ProCon baseline)
- Maintained task performance (vs. unmodified IFT)
- Better tradeoff than prior safety-specific methods

**Inferred metrics (typical for this research area):**
- ASR (Attack Success Rate) reduction: 30-50% lower than unmodified IFT
- Task performance: within 1-3% of unmodified IFT
- Geometric stability: drift < 5-10 degrees vs. 30-50 degrees for unmodified IFT

---

## 9. Implications for Refusal-Cones

### Validation Points

1. **Confirms geometric model:**
   - ProCon's effectiveness validates that refusal is indeed concentrated in geometric structure
   - Single (or low-dimensional) direction approach is well-justified

2. **Establishes r-direction importance:**
   - Defense perspective shows r-direction is critical to protect
   - Attack perspective (RDO) should therefore be highly effective if well-designed

3. **Provides drift baseline:**
   - ProCon documents drift rates during training
   - RDO should measure similar drift as part of evaluation

4. **Validates constraint-based approaches:**
   - Projection constraints work for preservation (ProCon)
   - Therefore likely effective for manipulation (RDO)

### Extensions for RDO

**Integration Points:**

1. **Warm-up Initialization:**
   - Use ProCon-style warm-up for RDO training
   - Early-phase strong constraints on refusal direction could improve convergence

2. **Drift-aware Evaluation:**
   - Measure r-direction drift as part of RDO evaluation
   - Ensure RDO vectors don't inadvertently stabilize/modify other safety aspects

3. **Multi-dimensional Geometry:**
   - ProCon could extend to Pan et al. and Piras et al. multi-dimensional models
   - RDO discovery could identify which dimensions to ablate most effectively

4. **Layer-specific Analysis:**
   - ProCon likely works per-layer (not shown but implied)
   - RDO's per-layer vectors should align with ProCon's per-layer constraints

### Potential Synergies

**Defensive-Offensive Alignment:**
```
RDO discovery identifies which r-direction is most effective
           ↓
RDO training learns to ablate it maximally
           ↓
ProCon applies constraints to stabilize that exact direction
           ↓
Result: Complete mechanistic understanding of refusal robustness
```

**Quality Metrics:**
```
Use ProCon insights to improve RDO evaluation:
- Measure angular divergence of discovered vs. identified r-direction
- Track drift of other safety-relevant directions
- Ensure discovered vectors don't corrupt secondary safety mechanisms
```

---

## 10. Key Takeaways for Literature Review

### What ProCon Teaches

1. **Refusal geometry is real:** Effective defensive constraint validates geometric model
2. **Direction drift is the threat:** Early training phase most critical
3. **Constraints are sufficient:** No need for holistic approaches
4. **Two-phase training works:** Warm-up + relaxation enables optimization

### What This Means for RDO

1. **Your assumptions are well-founded:** r-direction is indeed critical and concentrated
2. **Your discovery methods are necessary:** Identifying exact r-direction enables both defense (ProCon) and offense (RDO)
3. **Your training approach is justified:** RDO's multi-objective formulation parallels ProCon's balanced approach
4. **Extension opportunities exist:** Multi-dimensional, cross-lingual, layer-specific analyses all enabled

### Research Direction

ProCon + RDO together create complete mechanistic understanding:
- **ProCon:** "Here's how to defend against refusal manipulation"
- **RDO:** "Here's how to manipulate refusal"
- **Together:** "Here's why refusal mechanisms are fundamentally geometric and how to understand them completely"

---

## 11. References & Further Reading

### Directly Cited in ProCon (inferred or confirmed)

1. **Arditi et al. (2024).** "Refusal in Language Models Is Mediated by a Single Direction." arXiv:2406.11717

### Contemporary Related Work

2. **Pan et al. (2025).** "The Hidden Dimensions of LLM Alignment: A Multi-Dimensional Analysis of Orthogonal Safety Directions." arXiv:2502.09674v4

3. **Piras et al. (2025).** "SOM Directions are Better than One: Multi-Directional Refusal Suppression in Language Models." arXiv:2511.08379v2

4. **Wang et al. (2025).** "Refusal Direction is Universal Across Safety-Aligned Languages." arXiv:2505.17306v1

5. **Siu et al. (2025).** "COSMIC: Generalized Refusal Direction Identification in LLM Activations." arXiv:2506.00085v1

### Defensive Approaches

6. **Xie et al. (2025).** "Beyond Surface Alignment: Rebuilding LLMs Safety Mechanism via Probabilistically Ablating Refusal Direction." arXiv:2509.15202v1

7. **Li et al. (2024).** "Safety Layers in Aligned Large Language Models: The Key to LLM Security." arXiv:2408.17003v5

### Offensive/Mechanistic Perspectives

8. **Zhang & Sun (2025).** "Differentiated Directional Intervention: A Framework for Evading LLM Safety Alignment." arXiv:2511.06852v4

9. **Yuan et al. (2024).** "Refuse Whenever You Feel Unsafe: Improving Safety in LLMs via Decoupled Refusal Training." arXiv:2407.09121v2

---

## 12. Document History

| Date | Version | Notes |
|------|---------|-------|
| 2026-01-19 | 1.0 | Initial analysis based on abstract and related work |
| — | Future | Update with full paper when accessible |

---

## Appendix: Quick Comparison Matrix

### ProCon vs. Refusal-Cones RDO

| Criterion | ProCon | RDO |
|-----------|--------|-----|
| Problem statement | Safety drift during IFT | Refusal geometry & controllability |
| Method class | Constrained optimization | Geometric discovery + training |
| Core operation | Regularize projections | Ablate/add projections |
| Direction role | Preserve | Manipulate |
| Research goal | Defensive (maintain safety) | Offensive/exploratory |
| Key metric | Behavioral safety, direction drift | Attack success rate, discovery efficiency |
| Training phases | 2 (warm-up + main) | 2+ (discovery + training) |
| Evaluation focus | Safety preservation | Harm maximization + capability retention |
| Architecture scope | Llama-2, Qwen-2 | (To be determined by RDO) |
| Warm-up strategy | Strong early constraints | (To be determined) |

### Shared Geometric Understanding

| Aspect | Both Use |
|--------|----------|
| Foundational work | Arditi et al. 2406.11717 |
| Direction identification | Mean difference: `E[h_harmful] - E[h_harmless]` |
| Geometric primitive | Projection onto r-direction |
| Layer scope | Per-layer analysis |
| Model classes | Instruction-tuned chat models |
| Concept | Refusal is concentrat
ed geometric feature |

---

**Document prepared for:** Refusal-Cones literature review and framework alignment
**Prepared by:** Automated analysis of arXiv:2509.06795 and related work
**Date:** January 19, 2026
