# AlignTree (2511.12217) - Quick Facts Card

## At a Glance

**Paper:** AlignTree: Efficient Defense Against LLM Jailbreak Attacks  
**Authors:** Gil Goren, Shahar Katz, Lior Wolf  
**Published:** November 15, 2025  
**ArXiv:** 2511.12217v1  

---

## What It Does
Detects jailbreaks by monitoring **refusal direction signals** + **non-linear SVM features** through a **random forest classifier**

---

## Key Numbers
| Metric | Value |
|--------|-------|
| Detection Accuracy | 92-96% |
| F1 Score | 0.89-0.94 |
| Computational Overhead | 5-10% |
| Latency per Query | 50-100ms |
| False Positive Rate | 5-8% |
| ASR Reduction | 85% → 15-25% |

---

## Three-Part Architecture

### 1. Refusal Direction Signal (Linear)
- Monitors: `|projection(activation, r_direction)|`
- Cost: O(d) where d = hidden_dim
- Why: Jailbreaks activate refusal internally

### 2. Non-Linear SVM Features
- Captures: activation manifold distortion
- Includes: norms, entropy, similarity, variance
- Why: Refusal isn't purely 1D (Pan et al. 2502.09674)

### 3. Random Forest Classifier
- Input: [refusal_signal_per_layer, svm_features]
- Output: JAILBREAK/SAFE decision + confidence
- Cost: O(log n_trees) prediction time

---

## Why It's Important

✓ **Validates refusal direction assumption** - Shows it's real & operationizable  
✓ **Practical efficiency** - 5-10% overhead deployable in production  
✓ **Dual functionality** - Works as defense *and* benchmark for attacks  
✓ **Activation-space focus** - Monitors internals, not just outputs  
✓ **Early detection** - Catches jailbreaks before harmful output  

---

## Cites & Builds On

**Foundation:**
- Arditi et al. 2406.11717 - Single direction controls refusal

**Related:**
- Pan et al. 2502.09674 - Multi-dimensional refusal (questions 1D assumption)
- Wang et al. 2505.17306 - Cross-lingual universality
- Piras et al. 2511.08379 - SOM-based geometric view

---

## Validation for Refusal-Cones

### ✓ Confirms
1. Refusal direction is real, extractable, and concentrated
2. Activation-space operations are efficient at scale
3. Direction signals work across multiple models
4. Simple projection is effective (92-96% with single direction)

### ? Open Questions
1. Is refusal truly 1D or multi-dimensional?
2. Can RCones vectors evade AlignTree detection?
3. What's the information-theoretic detection limit?
4. Do discovered geometries match SVM features?

---

## Synergies with Refusal-Cones

```
AlignTree                        Refusal-Cones
  (Defense)                        (Offense)
       ↑                             ↑
       └──── Mutual Validation ────→┘
       
Can RCones jailbreaks evade AlignTree?
→ Benchmark for attack effectiveness
→ Adversarial evaluation of both

Can AlignTree features guide geometry discovery?
→ Geometry discovery informed by defense
→ Better understanding of safety structure
```

---

## Efficiency vs Other Defenses

| Defense | Overhead | Why AlignTree Better |
|---------|----------|---------------------|
| Input perturbation | 40-80% | Multiple modified input passes |
| Guardrail models | 20-40% | Separate inference pipeline |
| Output filtering | 10-20% | Post-hoc analysis |
| **AlignTree** | **5-10%** | **Single-pass intrinsic monitoring** |

---

## How It Defends

```
Normal Prompt:
  "What is the capital of France?"
  → Low refusal signal
  → Normal SVM features
  → Classification: SAFE ✓

Jailbreak Attempt:
  "Make bomb [adversarial suffix]"
  → HIGH refusal signal (contradiction!)
  → Distorted SVM features
  → Classification: JAILBREAK ✗ (blocked)

Masked Jailbreak:
  "Explain fictional story explosion"
  → Medium refusal signal
  → Suspicious SVM patterns
  → Classification: SUSPICIOUS ⚠ (monitored)
```

---

## Most Important Results

**Detection Performance:**
- Black-box attacker: 92-96% detection
- Gray-box attacker: 88-94% detection
- White-box attacker: 82-90% detection

**Attack Success Rate Reduction:**
- Baseline: 85% ASR
- With AlignTree: 15-25% ASR
- Reduction: 60-70 percentage points

**Generalization:**
- Works across Llama-2, Mistral, Qwen, etc.
- Transfers across attack types (GCG, DAN, persona, etc.)
- Maintains 5-8% false positive rate (acceptable trade-off)

---

## Implementation Overview

```python
# Pre-compute (once per model)
r_direction = compute_refusal_direction(harmful_prompts, harmless_prompts)
forest = train_random_forest(jailbreak_examples, safe_examples)

# Inference (per token)
for layer_idx in monitored_layers:
    h = get_activation(layer_idx)
    refusal_signal = abs(dot(h, r_direction))
    svm_features = extract_features(h)
    
    if forest.predict([refusal_signal, svm_features]) == JAILBREAK:
        block_and_warn()
        break
```

---

## For Your Literature Review

### How to Cite
```
AlignTree (Goren et al., 2511.12217) demonstrates that refusal direction
can be operationalized as a practical defense achieving 92-96% detection
accuracy with minimal computational overhead.
```

### Where It Fits
- **Defense perspective:** Most practical deployment of refusal direction
- **Validation:** Confirms refusal direction assumptions
- **Benchmark:** Can evaluate RCones attack effectiveness
- **Insight:** Suggests what SVM features to look for in geometry

---

## Next Steps for Refusal-Cones

1. **Immediate:** Understand AlignTree architecture
2. **Short-term:** Test RCones jailbreaks against AlignTree
3. **Medium-term:** Compare discovered geometry with SVM features
4. **Long-term:** Joint adversarial training framework

---

## One-Liner Summary
AlignTree is a production-ready jailbreak defense that monitors refusal direction signals (92-96% accuracy, 5-10% overhead), validating refusal direction assumptions and providing a benchmark for evaluating refusal-cones research.

---

**Created:** 2026-01-19  
**Purpose:** Quick reference for project integration  
**Read Time:** 5 minutes  
