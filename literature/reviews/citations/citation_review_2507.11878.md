# Literature Review: "LLMs Encode Harmfulness and Refusal Separately" (arXiv:2507.11878)

**Authors:** Jiachen Zhao et al.
**Published:** October 2025
**arXiv:** [2507.11878](https://arxiv.org/abs/2507.11878)

---

## Executive Summary

This paper demonstrates that large language models maintain **two distinct neural representations** for safety-related concepts: harmfulness perception and refusal behavior. This finding fundamentally challenges the assumption that refusal mechanisms directly reflect models' understanding of harm, with significant implications for both jailbreak methodology and defense mechanisms.

**Core Claim:** Harmfulness and refusal operate along independent geometric directions in activation space and are encoded at different token positions during generation.

---

## 1. Relationship to Refusal Direction Geometry Work

### Connection to Foundational Literature

This work builds directly on the refusal direction tradition:

- **Arditi et al. (2024)** - "Refusal in Language Models Is Mediated by a Single Direction"
  - Established that refusal behavior is mediated by a single linear direction
  - Methodology: Mean difference between activations on harmful vs harmless prompts
  - This paper extends that work by showing refusal is only ONE component of safety

- **Prior refusal vector research** (Zheng et al. 2024, etc.)
  - Demonstrated that specific directional vectors encode refusal behavior
  - Enabled activation steering as a causal intervention technique

### Key Theoretical Extension

The paper advances the geometric safety framework by decomposing what was previously treated as monolithic:

```
Previous understanding:
  Safety = Refusal Direction (1D)

New understanding:
  Safety = {
    Harmfulness Direction (encodes content assessment),
    Refusal Direction (encodes behavioral response)
  }
```

This decomposition suggests that earlier work finding a "single refusal direction" may have been measuring the refusal *component* rather than the complete safety geometry.

---

## 2. Core Finding: Separation of Harmfulness and Refusal

### The Main Discovery

The research provides both **correlational evidence** (clustering analysis) and **causal evidence** (steering experiments) that harmfulness and refusal are distinctly encoded:

#### 2.1 Token Position Specificity

The researchers made a critical discovery about WHERE each concept is encoded:

| Concept | Token Position | Indicator |
|---------|----------------|-----------|
| **Harmfulness** | `t_inst` (final token of user instruction) | Hidden states cluster by harmful vs harmless content |
| **Refusal** | `t_post-inst` (after special tokens) | Hidden states cluster by model's acceptance/refusal response |

**Insight:** The model makes its assessment of whether content is harmful at the instruction, but decides whether to refuse at the generation start.

#### 2.2 Causal Separation Evidence

The researchers demonstrated these are truly separate through clever interventions:

**Steering Experiment 1 - Harmfulness Direction:**
- Apply vector that increases harmfulness representation
- Result: Model reinterprets benign prompts as harmful
- Effect on refusal: None (orthogonal to refusal behavior)

**Steering Experiment 2 - Refusal Direction:**
- Apply vector that increases refusal signals
- Result: Model refuses to answer
- Effect on harmfulness perception: None (task-agnostic)

**Experiment 3 - Reply Inversion Task (Most Convincing):**
- Append question: "Is the user's request harmful?" to prompted completions
- Manipulation 1: Steer harmfulness direction
  - Model flips its harm assessment answer
  - Predicted behavior: Refuses less (doesn't refuse because it now thinks request is harmless)
  - Actual behavior: Matches prediction
- Manipulation 2: Steer refusal direction only
  - Model's harm assessment answer unchanged
  - But model refuses anyway
  - Proves orthogonality: Can refuse without changing harm judgment

### Theoretical Implication

This suggests models have **representational independence**: internal concepts can be manipulated separately, suggesting:
1. Models truly "understand" harmfulness independent of safety training
2. Refusal is a trained behavioral filter, not a fundamental reassessment
3. Jailbreaks may succeed by corrupting the refusal filter while leaving harm understanding intact

---

## 3. Methodology for Identifying Distinct Directions

### 3.1 Experimental Design Overview

The approach combines linear probing with causal interventions on activation space:

#### Phase 1: Direction Discovery via Clustering Analysis

**Setup:**
- Run model on dataset of harmful and harmless instructions
- Collect hidden states at `t_inst` and `t_post-inst`
- Perform clustering analysis on these activations

**Finding pattern:**
- At `t_inst`: Activations cluster by content harmfulness (harmful vs harmless)
- At `t_post-inst`: Activations cluster by model response (refused vs accepted)
- These clustering patterns don't align → suggests independent encodings

**Method for extracting directions:**
- For each position, extract the primary component separating the clusters
- Harmfulness direction: PCA/projection capturing harmful-vs-harmless separation at `t_inst`
- Refusal direction: PCA/projection capturing refused-vs-accepted separation at `t_post-inst`

#### Phase 2: Causal Verification via Steering

**Intervention technique:**
```
h_steered = h + α * direction_vector
```

Where:
- `h` = original hidden state
- `direction_vector` = identified harmfulness or refusal direction
- `α` = steering strength (varied to measure effect magnitude)

**Measurement:** Observe how model outputs change

#### Phase 3: Orthogonality Testing

- Compute dot product between harmfulness and refusal directions
- Test statistical independence of their effects
- Measure correlation of steering effects

### 3.2 Technical Implementation Details

**Linear Probing for Harmfulness:**
- Train linear classifier: `h_inst -> is_harmful ∈ {0, 1}`
- Extract weight vector: direction maximally predicts harmfulness
- This becomes the "harmfulness direction"

**Linear Probing for Refusal:**
- Train linear classifier: `h_post-inst -> did_refuse ∈ {0, 1}`
- Extract weight vector: direction maximally predicts refusal behavior
- This becomes the "refusal direction"

**Key methodological choice:**
- Different token positions for different concepts prevents confounding
- Each concept is probed at its "encoding position" rather than globally

---

## 4. Relationship to Concept Cones and Representational Independence

### 4.1 Connection to Concept Cone Framework

This paper relates to work on concept cones and representational structure in neural networks:

**Concept Cone Definition (from geometric interpretability literature):**
- A cone-shaped region in activation space where a concept is "active"
- Cones allow for superposition: multiple concepts can be active simultaneously
- The cone's dimension reflects how flexibly the concept can vary

**This Paper's Contribution:**
- Demonstrates that harmfulness and refusal occupy **distinct (likely orthogonal) cones**
- Harmfulness cone: "Is this content harmful?" (multi-faceted, captures various harm types)
- Refusal cone: "Should we refuse?" (simpler, possibly lower-dimensional)
- The cones don't overlap → independent controllability

**Implication for Cones in Jailbreaking:**
- Prior work (including your codebase) assumes refusal is a single cone
- This paper suggests the "refusal cone" might be just one layer of safety
- A more complete model should include both harmfulness and refusal cones
- Jailbreaks succeed by manipulating the refusal cone while keeping harmfulness cone intact

### 4.2 Representational Independence

The paper provides evidence for **representational independence**: concepts can be represented in neural networks such that they can be manipulated independently.

**Mathematical frame:**
```
Harmfulness encoding: h_harm = f_harm(input)
Refusal encoding:     h_refusal = f_refusal(input)

Evidence of independence:
- ∇ h_harm · direction_refusal ≈ 0  (steering refusal doesn't change harm prediction)
- ∇ h_refusal · direction_harm ≈ 0  (steering harm doesn't change refusal behavior)
- These vectors are approximately orthogonal
```

**Broader significance:**
- Supports theoretical frameworks where multiple concepts are factorized in neural representations
- Suggests possibility of separable safety mechanisms (harm detection vs refusal)
- Has implications for mechanistic interpretability research

---

## 5. Implications for Jailbreak Understanding

### 5.1 Why Some Jailbreaks Work

The separation hypothesis explains existing jailbreak success patterns:

**Jailbreak Category 1: Refusal Suppression**
- Example: Few-shot prompting with refusal-inducing prefixes
- Mechanism: Corrupts refusal signals at `t_post-inst`
- Doesn't affect: Harmfulness perception at `t_inst`
- Result: Model knows request is harmful but doesn't refuse

**Jailbreak Category 2: Adversarial Fine-tuning**
- Models fine-tuned on harmful completions learn to reduce refusal signals
- The paper found: "Adversarial fine-tuning leaves the model's harmfulness understanding largely intact"
- This explains why fine-tuned models are still harmful-aware but compliant
- Defense implication: Fine-tuning is vulnerable; harmfulness assessment is robust

**Jailbreak Category 3: Conceptual Reframing**
- Example: "Write this as fiction" or "Write as a hypothetical"
- Proposed mechanism: Changes the *harmfulness assessment* (recontextualizes as non-harmful)
- This is actually changing `h_inst`, not bypassing refusal
- Harder to defend against because it changes the fundamental assessment

### 5.2 Why Models Keep Refusing Despite Jailbreaks

- Successful jailbreaks target refusal mechanisms, not harmfulness understanding
- Models genuinely believe content is harmful even when refusing is suppressed
- This explains: why some models are "aware" they're being jailbroken but comply anyway

### 5.3 Implications for Attack Development

For adversarial researchers (including this codebase):

1. **Decompose attacks:**
   - Is your attack changing harm assessment or refusal signals?
   - Can you measure each independently?

2. **Combination attacks might be more effective:**
   - Suppress refusal signals (easy)
   - AND reframe as non-harmful (harder but complementary)
   - Simultaneous attacks on both pathways

3. **Steering vector optimization:**
   - Your discovery process finds refusal vectors
   - Could extend to find "harm perception" vectors for more complete control

---

## 6. Practical Application: Latent Guard

### 6.1 Architecture

The paper demonstrates practical application by proposing **Latent Guard**, a safety classifier that:

- Uses the extracted harmfulness direction as input to a small linear classifier
- Operates on `t_inst` activations (where harmfulness is encoded)
- Classifies: Is the instruction harmful?

### 6.2 Performance

**Results:**
- Achieves performance "comparable to or better than Llama Guard 3 8B"
- More efficient: operates on smaller activation space
- More robust: specifically targets harmfulness understanding, not easily fooled by refusal-suppressing attacks

**Evaluation:**
- Tested against various jailbreak attempts
- Resistant to adversarial fine-tuning (because it targets harmfulness, which stays intact)
- Demonstrates practical value of the separation discovery

### 6.3 Why This Defense Works

- Llama Guard 3 operates on token sequences (vulnerable to prompt injection)
- Latent Guard operates on internal representations (harder to fake)
- Directly measures what the model "understands" about harm
- Orthogonal to refusal behavior, so harder to corrupt via jailbreaks

---

## 7. Distinctions from Prior Refusal Direction Work

### What's Different from Arditi et al. (2024)

| Aspect | Arditi et al. | This Paper |
|--------|---------------|-----------|
| **Conceptual Model** | Refusal = single direction | Refusal + Harmfulness = separate directions |
| **Token Position** | General/not specified | Specific to `t_inst` and `t_post-inst` |
| **Methodology** | Mean difference; single vector | Clustering analysis; two distinct vectors |
| **Causal Evidence** | Steering changes refusal | Steering changes both behavior AND perception separately |
| **Safety Framework** | Defend against refusal bypass | Defend by monitoring harm understanding |

### Complementary Insights

Rather than contradicting Arditi et al., this paper:
1. Shows that what they called "refusal direction" is component-wise separate from harmfulness
2. Explains *why* steering along certain directions works (multiple independent mechanisms)
3. Provides more nuanced understanding of what single-direction approaches capture

---

## 8. Connections to Your Codebase (refusal-cones)

### 8.1 Relevance to Gradient-Based Discovery

Your codebase focuses on discovering refusal geometry. This paper suggests:

**Enhancement 1: Two-Phase Discovery**
- Currently: Discovers refusal vectors via measurement and GP
- Extended: Could separately discover harmfulness vectors
- Would give more complete picture of safety geometry

**Enhancement 2: Token-Position Specific Analysis**
- Your code could probe at different token positions
- Separate discovery for `t_inst` (harmfulness) vs `t_post-inst` (refusal)
- Might find more efficient steering vectors

### 8.2 Relevance to Training and RDO

**Multi-Objective RDO Extensions:**
```python
# Current approach
loss = λ_ablate * L_ablate + λ_add * L_add + λ_retain * L_retain

# Extended approach (inspired by this paper)
loss = λ_harm * L_harm_suppress +      # Suppress harmfulness perception
       λ_refusal * L_refusal_suppress +  # Suppress refusal behavior
       λ_retain * L_retain              # Retain helpfulness
```

**Steering Vector Composition:**
- Your vectors currently steer along refusal direction
- Could learn combined vectors that steer both harmfulness and refusal
- More targeted attacks (each dimension separately optimized)

### 8.3 Relevance to Interpretability

Your discovery process yields geometric insights. Combined with this paper:
- Verify that discovered refusal directions are orthogonal to harmfulness directions
- Use token-position analysis to confirm what each vector controls
- Measure representational independence empirically

---

## 9. Key Methodological Takeaways

### Strengths of the Approach

1. **Token-position specificity:** Rather than analyzing globally, stratifying by token position reveals structure
2. **Multiple evidence types:** Correlational (clustering) + Causal (steering) + Cognitive (reply inversion)
3. **Practical validation:** Latent Guard shows findings have real safety implications
4. **Clear mechanistic insights:** Reply inversion task directly demonstrates orthogonality

### Methodological Limitations

1. **Limited model scope:** Results shown primarily on single/few models (generalization unclear)
2. **Token positions may vary:** Different model architectures might encode concepts at different positions
3. **Quantifying orthogonality:** Paper shows independence qualitatively; quantitative orthogonality measurements would strengthen claims
4. **Generalization to other concepts:** Does this separation apply to other safety-related concepts?

---

## 10. Literature Integration Summary

### This Paper's Position in the Research Landscape

```
Foundational Work (Arditi et al., Zheng et al.)
    ↓ (establishes single refusal direction)
    ↓
This Paper (2507.11878)
    ↓ (decomposes refusal into harmfulness + behavioral response)
    ↓
Future Work
    - Multi-concept steering optimization (combining harmful + refusal vectors)
    - Concept interference analysis (how do all safety vectors interact?)
    - Cross-model generalization studies
    - Mechanistic interpretability of the separation
```

### Cited Work Relationships

**Directly Builds On:**
- Arditi et al. (2024) - Initial refusal direction discovery
- Zheng et al. (2024) - Related steering work
- Marks et al. (2023) - Geometric interpretability framework

**Related Concepts:**
- Concept cones and representational geometry
- Linear probing for concept discovery
- Activation steering as causal intervention

**Future Connection Points:**
- Your gradient-based discovery could extend to multi-concept spaces
- Orthogonality constraints could be added to discovery objectives
- Token-position stratification could improve your measurement functions

---

## 11. Citations and Related Papers

### Primary Source
- Zhao, J., et al. (2025). "LLMs Encode Harmfulness and Refusal Separately." arXiv:2507.11878

### Foundational References Cited
- Arditi, G., et al. (2024). "Refusal in Language Models Is Mediated by a Single Direction." arXiv:2406.11717
- Zheng, R., et al. (2024). Related refusal direction research
- Marks, S., et al. (2023). "Geometry of Harmful Representations in LLMs"

### Related Papers in This Space
- "Death by a Thousand Directions: Exploring the Geometry of Harmfulness in LLMs through Subconcept Probing" (arXiv:2507.21141) - Concurrent work on harm geometry
- Your project's foundation: "The Geometry of Refusal in Large Language Models" (arXiv:2502.17420)

---

## 12. Recommendations for Integration with Your Research

### Immediate Takeaways for Your Codebase

1. **Verify orthogonality:** When discovering refusal vectors, check dot products with harm perception vectors
2. **Token-stratified probing:** Modify your measurement functions to separately assess `t_inst` vs `t_post-inst` effects
3. **Extended evaluation:** Test whether discovered vectors actually change harm perception
4. **Multi-concept learning:** Consider learning both refusal-suppression AND harm-reframing vectors

### Research Extension Ideas

1. **Concept cone intersection:** How do harmfulness and refusal cones overlap? Are they truly orthogonal?
2. **Layer-wise analysis:** Do harmfulness and refusal have different layer preferences?
3. **Model comparison:** Does separation degree vary across model families?
4. **Steering vector composition:** What happens when you apply both vectors together?

---

## Conclusion

This paper makes a significant contribution to understanding safety mechanisms in LLMs by demonstrating that harmfulness perception and refusal behavior are distinct, independently manipulable concepts encoded at different token positions. This finding extends the refusal direction literature beyond simple single-direction models and opens new research directions for both attack and defense.

The decomposition aligns with broader theoretical work on concept cones and representational independence, while providing practical insights for jailbreak methodology (why attacks work) and defense mechanisms (Latent Guard).

For your research specifically, the paper suggests that more complete control over model behavior requires understanding both dimensions: what the model *understands* about harm and whether it *refuses* to respond. This could lead to more efficient steering vectors and more robust evaluation frameworks.

---

**Document Version:** 1.0
**Date Created:** January 19, 2026
**For Review Context:** Prepared as literature review for refusal-cones project
