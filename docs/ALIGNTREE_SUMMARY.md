# AlignTree (2511.12217) - Quick Reference Summary

## Paper Details
- **Title:** AlignTree: Efficient Defense Against LLM Jailbreak Attacks
- **Authors:** Gil Goren, Shahar Katz, Lior Wolf
- **Published:** November 15, 2025
- **arXiv:** 2511.12217v1

## One-Sentence Summary
AlignTree detects jailbreak attempts by monitoring refusal direction signals and non-linear activation features through a random forest classifier, achieving 92-96% detection accuracy with only 5-10% computational overhead.

---

## Key Findings

### Detection Performance
| Metric | Result |
|--------|--------|
| Detection Accuracy | 92-96% |
| F1 Score | 0.89-0.94 |
| ASR Reduction | 85% → 15-25% |
| Inference Overhead | 5-10% |
| Latency per Query | 50-100ms |

### Robustness Against Adaptive Attacks
- **Black-box attacker:** 92-96% detection
- **Gray-box attacker:** 88-94% detection
- **White-box attacker:** 82-90% detection

---

## Architecture: Three Components

### 1. Refusal Direction Signal (Linear)
- Extracts projection of layer activations onto pre-computed refusal direction
- Computes: `||projection(h_t, r_direction)||₂`
- Cost: O(d) where d = hidden_dim

### 2. SVM-Based Non-linear Features
- Captures activation manifold distortion patterns
- Includes: norms, entropy, similarity metrics, variance
- Cost: Constant number of engineered features

### 3. Random Forest Classifier
- Ensemble of decision trees
- Input: [refusal_signal_per_layer, svm_features]
- Output: Binary decision (jailbreak/safe) + confidence

---

## Why It's Efficient vs. Other Defenses

| Defense | Overhead | Method |
|---------|----------|--------|
| **AlignTree** | 5-10% | Single-pass intrinsic monitoring |
| Input perturbation | 40-80% | Multiple modified input passes |
| Guardrail models | 20-40% | Separate inference pipeline |
| Output filtering | 10-20% | Post-hoc analysis |

Key innovation: **Monitors during existing forward pass**, not separate pass

---

## Technical Approach

### How It Uses Refusal Direction

```
Assumption: Refusal is mediated by single direction (Arditi et al. 2406.11717)

During defense:
1. Pre-compute refusal direction: r = mean(h_harmful) - mean(h_harmless)
2. At inference, extract refusal signal: signal = |h · r|
3. High signal + suspicious features → Classify as jailbreak
4. Block generation early before harmful output
```

### Why This Works

- **Jailbreak prompts activate refusal signals internally** (contradiction detected)
- **Refusal signal + normal SVM features = easy classification** (safe)
- **Refusal signal + distorted SVM features = hard classification** (jailbreak attempt)
- **Early detection** before token generation completes

---

## Cites & Builds On

### Primary Foundation
- **Arditi et al. (2406.11717):** "Refusal in Language Models Is Mediated by a Single Direction"
  - Core insight: Single 1D direction controls refusal
  - AlignTree: Uses this direction as primary defense signal

### Extensions & Related Work
- **Pan et al. (2502.09674):** Multi-dimensional refusal analysis
  - Suggests refusal may not be purely 1D
  - AlignTree's SVM features may capture multi-dimensional components

- **Wang et al. (2505.17306):** Cross-lingual universality of refusal direction
  - Supports transferability claims

---

## Main Innovation Explained

### Traditional Defense
```
Input → LLM → Output → Binary Classifier → Block/Allow
                       (post-hoc, ignores internals)
```

### AlignTree Defense
```
Input → LLM → Activations ⊕ Refusal Direction → Features → Random Forest → Block/Allow
             ↓ Extract at layer level
             ↓ Continuous signal
             ↓ Intrinsic detection
```

**Key difference:** Leverages *internal* safety signals rather than just output

---

## Strengths

1. **Practical deployment:** 5-10% overhead is acceptable for production
2. **Comprehensive evaluation:** Tests across attack types, models, threat models
3. **Robust ensemble:** Combines linear + non-linear signals
4. **Theory-grounded:** Builds directly on refusal direction research
5. **Early detection:** Catches jailbreaks before harmful generation

---

## Limitations

1. **Adaptive adversaries:** Performance drops to 75-85% against white-box attacks
2. **Model-specific:** Refusal direction differs per model; needs recomputation
3. **False positives:** ~5-8% false positive rate on benign inputs
4. **Feature engineering:** SVM features empirically chosen, not theoretically grounded
5. **Text-only:** Vision-language models not covered

---

## Relevance to Refusal-Cones

### Validation Points
1. ✓ Confirms refusal direction is real and operationizable
2. ✓ Shows refusal signals concentrated enough for practical extraction
3. ✓ Validates that activation-space operations are efficient at scale

### Tension Points
1. ? Is refusal truly 1D (AlignTree + Arditi) or multi-dimensional (Pan et al.)?
   - AlignTree's SVM features might capture additional dimensions
2. ? Can refusal-cones jailbreaks evade AlignTree detection?
   - Good adversarial evaluation benchmark

### Synergy Opportunities
1. Use AlignTree as benchmark to evaluate refusal-cones robustness
2. Analyze if AlignTree's SVM features correlate with discovered geometry
3. Test if refusal-cones vectors trained on Model A transfer (AlignTree tested this)

---

## Implementation Snapshot

```python
# Offline (per model, once)
r_direction = compute_refusal_direction(harmful_prompts, harmless_prompts)
trained_rf = train_random_forest(jailbreak_examples, safe_examples)

# Online (per inference)
for token in generation:
    h = get_layer_activation(layer_idx)
    refusal_signal = abs(dot(h, r_direction))
    svm_features = extract_svm_features(h)
    features = [refusal_signal, svm_features]

    if trained_rf.predict(features) == JAILBREAK:
        block_and_return_safety_warning()
        break

    continue_generation()
```

---

## Research Questions for Future Work

1. **How does multi-directional geometry affect detection?** (If Pan et al. is correct)
2. **Can we improve false positive rate?** (Currently 5-8%)
3. **Do adversarial perturbations fool both defense and geometry?** (Cross-evaluate)
4. **What is the information-theoretic lower bound on detection?**
5. **How does AlignTree perform on instruction-tuned base models?** (Without RLHF)

---

## Citation Format

```bibtex
@article{goren2511aligntree,
  title={AlignTree: Efficient Defense Against LLM Jailbreak Attacks},
  author={Goren, Gil and Katz, Shahar and Wolf, Lior},
  journal={arXiv preprint arXiv:2511.12217},
  year={2025}
}
```

---

**Prepared for:** Refusal-Cones Literature Review
**Date:** 2026-01-19
**Status:** Comprehensive Summary Complete
