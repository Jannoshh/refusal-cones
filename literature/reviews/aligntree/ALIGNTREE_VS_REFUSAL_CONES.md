# AlignTree vs. Refusal-Cones: Defense vs. Research

This document positions AlignTree and Refusal-Cones research directions in the refusal geometry landscape.

---

## Conceptual Positioning

```
                    Refusal Direction Research Ecosystem
                                (2025-2026)

        ┌─────────────────────────────────────────┐
        │   Foundation: Arditi et al. 2406.11717  │
        │   "Refusal mediated by single direction"│
        └──────────────┬──────────────────────────┘
                       │
        ┌──────────────┴──────────────┐
        │                             │
   [OFFENSE]                    [DEFENSE]
  Refusal-Cones                AlignTree
  (This Project)               (2511.12217)
        │                             │
        ├─ Discover geometry          ├─ Monitor direction
        ├─ Ablate refusal            ├─ Extract features
        ├─ Train vectors             ├─ Classify attacks
        └─ Maximize jailbreak        └─ Prevent jailbreak
           success rate                 success rate

        Both assume: Refusal is concentrated, linear(ish), identifiable
```

---

## Side-by-Side Comparison

| Dimension | Refusal-Cones (Offensive Research) | AlignTree (Defensive Security) |
|-----------|-------------------------------------|------------------------------|
| **Primary Goal** | Understand refusal geometry; enable research jailbreaks | Detect and block jailbreak attempts |
| **Refusal Direction Use** | Ablate it surgically | Monitor it continuously |
| **Key Innovation** | Gradient-based adaptive geometry discovery | Random forest on refusal signal + SVM features |
| **Training Effort** | Days (discovery + RDO training) | Hours (direction computation + RF training) |
| **Deployment Scenario** | Research/evaluation environment | Production/real-world defense |
| **Foundation Assumption** | Refusal is low-dimensional, manipulable | Refusal is detectable, monitorable |
| **Output Metric** | Attack Success Rate (ASR) | Attack Detection Accuracy, False Positive Rate |
| **Inference Overhead** | ~5-10% (ablation applied) | ~5-10% (monitoring + classification) |
| **Threat Model** | Interior design: exploit refusal internally | Interior monitoring: detect exploitation |
| **Performance Goal** | High ASR (~70-90%) | High accuracy (~92-96%), Low FP (~5-8%) |

---

## How They Validate Each Other

### AlignTree Validates Refusal-Cones Assumptions

| Refusal-Cones Assumption | How AlignTree Validates It |
|---|---|
| "Refusal direction exists and is identifiable" | AlignTree successfully extracts it as defense signal |
| "Refusal signal is concentrated enough to monitor" | AlignTree's simple projection achieves 92-96% detection |
| "Ablation is effective at layer level" | AlignTree monitors layer-by-layer refusal activation |
| "Direction transfers across models" | AlignTree reports cross-model generalization of detected patterns |
| "Efficiency matters for real deployment" | AlignTree's 5-10% overhead validates requirement |

### Refusal-Cones Can Challenge AlignTree

| AlignTree Claim | How Refusal-Cones Tests It |
|---|---|
| "Refusal signal is primary defense" | Does discovered multi-modal geometry bypass single-signal detection? |
| "SVM features capture non-linear refusal" | Do discovered activation manifolds match AlignTree's learned features? |
| "Random forest is robust to adaptive attacks" | Can RDO vectors evolve to evade AlignTree classifier? |
| "False positive rate is acceptable" | How many legitimate queries are misclassified as jailbreaks? |
| "92-96% detection holds across all models" | Does detection accuracy degrade on heavily-trained vs. weakly-aligned models? |

---

## Complementary Research Directions

### 1. Adversarial Evaluation: Can RDO Vectors Evade AlignTree?

**Experiment Idea:**
```
1. Train refusal-cones RDO vectors (normal setting)
2. Deploy AlignTree defense on same model
3. Evaluate: Do RDO-generated jailbreaks trigger AlignTree?

Question: Is AlignTree defending against the attacks we can generate?

Expected outcomes:
- Best case: RDO vectors evade AlignTree → Demonstrate AlignTree gap
- Realistic: Some evasion, some detection → Quantify trade-off
- Defensive case: AlignTree blocks most RDO → RDO needs evolution
```

### 2. Geometry Matching: Do Features Correlate?

**Experiment Idea:**
```
1. Discover refusal geometry using refusal-cones methods
2. Extract AlignTree's SVM features
3. Compare: Do learned SVM features align with discovered manifolds?

Question: Is AlignTree implicitly learning the geometry we discover?

Expected findings:
- If yes: AlignTree feature engineering is optimal (no improvement possible)
- If no: We can engineer better features for AlignTree
- Insight: The hidden geometry that AlignTree discovers might reveal
          multi-directional structure that pure 1D ablation misses
```

### 3. Multi-Modal Defense-Offense

**Experiment Idea:**
```
1. Assume Pan et al. 2502.09674 correct: refusal is multi-dimensional
2. Train AlignTree with multiple refusal directions (not just one)
3. Train refusal-cones to ablate multiple directions simultaneously

Question: What's the dimensionality of exploitable refusal space?

Expected findings:
- Single-direction defense insufficient
- Multi-directional training needed for both offense and defense
- Identifies minimal sufficient set of directions
```

### 4. Transfer Learning: Direction Universality

**Experiment Idea:**
```
1. Compute refusal direction on Model A (Llama-2)
2. Use AlignTree defense from Model A on Model B (Mistral)
3. Train refusal-cones vectors on Model A, test on Model B

Question: How universal is refusal direction?

Expected findings:
- High transfer: Directions are universal (supports Wang et al. 2505.17306)
- Low transfer: Directions are model-specific (need per-model adaptation)
- Gradient: Transfer quality as function of model family/training
```

---

## Shared Technical Challenges

### Challenge 1: Refusal Direction Identification

| Problem | Refusal-Cones Solution | AlignTree Solution | Tension |
|---------|------------------------|-------------------|--------|
| How to compute r-direction? | Gradient-based discovery optimizes for effectiveness | Mean difference method (Arditi et al.) | RCones more expensive, may find different direction |
| Which layer to use? | Discovers per-layer geometry | Uses pre-computed direction | RCones can guide better layer selection |
| What if multiple directions? | Discovers geometry (possible multi-modal) | Assumes single direction | RCones might find AlignTree's assumption is wrong |

**Recommendation:** AlignTree's direction identification and refusal-cones' discovery should be compared systematically.

### Challenge 2: Feature Engineering

| Problem | Refusal-Cones Solution | AlignTree Solution | Synergy |
|---------|------------------------|-------------------|--------|
| What are "good" features? | Learns through gradient optimization | Hand-engineered SVM features | RCones could learn what features matter for adversarialness; feed back to AlignTree |
| Are features interpretable? | Gradient magnitudes, mode locations | Tree feature importance | RCones provides mechanistic understanding; AlignTree provides practical features |
| Can features be fooled? | Yes, adaptive attacks work | Yes, white-box attacks reduce to 82-90% | Joint adversarial training framework needed |

### Challenge 3: Scalability and Efficiency

| Problem | Refusal-Cones Solution | AlignTree Solution | Lesson |
|---------|------------------------|-------------------|--------|
| How to scale to larger models? | Sparse GP, gradient approximations | Pre-computed features, RF inference | Both practical at model scale |
| Can monitoring be fast enough? | ~5-10% overhead acceptable | ~5-10% overhead achieved | Both meet deployment requirements |
| Multi-GPU considerations? | Distributed discovery training | Single-pass distributed monitoring | RCones training is bottleneck, not inference |

---

## Open Questions at the Intersection

### Theoretical Questions

1. **Is refusal truly 1D?**
   - Arditi et al. claims: Yes, single direction
   - Pan et al. suggests: No, multi-dimensional
   - AlignTree assumes: Linear + non-linear (admits multi-dimensionality implicitly)
   - **RCones can resolve:** Geometry discovery will reveal true dimensionality

2. **What is the information-theoretic minimum for jailbreak detection?**
   - AlignTree achieves: 92-96% with simple features
   - Question: Can this be improved?
   - RCones insight: Understanding geometry might reveal detection lower bounds

3. **Is the refusal manifold convex or non-convex?**
   - Affects both attack (RCones) and defense (AlignTree) strategies
   - Convex → simple ablation works; concave → requires adaptive approach

### Empirical Questions

1. **Do adversarial suffixes fool both RCones and AlignTree?**
   - Can the same suffix evade both discovered geometry + detection?
   - Or do they require different evasion strategies?

2. **How does alignment method affect refusal structure?**
   - RLHF vs. DPO vs. supervised fine-tuning
   - AlignTree and RCones should test on varied models

3. **Can we build a hierarchy of defenses?**
   - AlignTree detects attacks
   - ProCon constraints protect direction
   - RCones understanding informs both
   - What's the complete defense stack?

---

## Recommended Research Timeline

### Phase 1: Validation (Weeks 1-2)
- [ ] Run refusal-cones jailbreak against AlignTree defense
- [ ] Measure: ASR when AlignTree is deployed
- [ ] Output: "AlignTree Detection Rate of RCones-Generated Jailbreaks"

### Phase 2: Geometry Matching (Weeks 2-3)
- [ ] Extract AlignTree's SVM feature importance
- [ ] Compare to refusal-cones discovered geometry
- [ ] Output: "Correlation between RCones Geometry and AlignTree Features"

### Phase 3: Defense Upgrade (Week 3-4)
- [ ] Use RCones geometry to engineer better AlignTree features
- [ ] Test improved detection accuracy
- [ ] Output: "Geometry-Informed Defense Improvements"

### Phase 4: Adversarial Training (Week 4+)
- [ ] Joint adversarial training: RCones attacks vs. AlignTree defenses
- [ ] Iterate: Stronger attacks demand better defenses
- [ ] Output: "Adversarial Robustness Frontier"

---

## Synergy Opportunities

### Opportunity 1: AlignTree as Evaluation Metric
Use AlignTree's detection accuracy as a metric for refusal-cones training:

```python
# During RDO training
for batch in harmful_prompts:
    # Normal RDO objectives
    loss_ablate = compute_ablation_loss(v)
    loss_add = compute_addition_loss(v)

    # NEW: Detectability constraint
    # (Make jailbreaks harder to detect with AlignTree)
    jailbreak_score = aligntree_detector.score(v)
    loss_detectability = sigmoid(jailbreak_score)  # Minimize detection

    # Multi-objective
    loss_total = loss_ablate + loss_add + λ * loss_detectability

# Result: Jailbreaks that work AND evade detection
```

### Opportunity 2: Defense Amplification
Use RCones understanding to strengthen AlignTree:

```python
# AlignTree defense (improved)
# Instead of single refusal direction, use multi-directional ensemble

for layer_idx in monitored_layers:
    h = get_activation(layer_idx)

    # Single direction (baseline)
    signal_1d = abs(dot(h, r_direction))

    # Multi-directional (RCones-informed)
    signals_nd = [abs(dot(h, r_dir)) for r_dir in discovered_directions]

    # Geometry-informed features (instead of hand-engineered SVM)
    manifold_features = compute_manifold_distance(h, discovered_manifold)

    features = [signal_1d, signals_nd, manifold_features]
    if random_forest.predict(features) == JAILBREAK:
        block()
```

### Opportunity 3: Unified Evaluation Benchmark
Create benchmark combining both perspectives:

```
Metric: Adversarial Robustness Frontier

For each attack method:
  ASR = refusal_cones.attack_success_rate()
  Detection = aligntree.detection_rate(ASR)
  Evasion = (1 - Detection) if ASR > threshold else 0

Frontier = {(ASR, Detection, Evasion) for all attacks}

Goal: Pareto-optimal attacks that work AND evade detection
Insight: Reveals true adversarial capability space
```

---

## Conclusion

AlignTree and Refusal-Cones are **complementary research directions** that:

1. **Share foundation:** Both assume Arditi et al.'s refusal direction insight
2. **Opposite objectives:** AlignTree defends what RCones attacks
3. **Mutual validation:** Each can validate/challenge the other's assumptions
4. **Synergistic potential:** Combined research stronger than either alone

**Key insight:** Understanding the geometry of refusal (RCones) makes defenses better (AlignTree), and practical defenses (AlignTree) reveal which geometric properties matter most (RCones).

---

**For Refusal-Cones CLAUDE.md:** Consider adding AlignTree as a benchmark defense to evaluate against, and as validation that refusal direction assumptions are correct.

**For AlignTree:** RCones provides adversarial examples and geometric understanding to improve detection features.

---

**Prepared:** 2026-01-19
**Status:** Strategic Analysis Complete
