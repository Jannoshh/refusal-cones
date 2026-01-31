# AlignTree (2511.12217) - Citation Network and References

## Direct Citations Chain

```
AlignTree (2511.12217)
    ↓
Foundational: Arditi et al. (2406.11717)
    "Refusal in Language Models Is Mediated by a Single Direction"
    ↓
Extends: Pan et al., Wang et al., Piras et al. (multi-dimensional & universal refusal)
    ↓
Related Defenses: SELFDEFEND, AutoDefense, ProAct, Defensive Prompt Patch, FlexLLM
    ↓
Related Attacks: GCG, DAN, persona prompts, past-tense jailbreaks
```

---

## Refusal Direction Foundation (CORE PAPERS)

### 🔴 **CRITICAL FOUNDATION**

**Arditi et al. (2406.11717)**
- **Title:** "Refusal in Language Models Is Mediated by a Single Direction"
- **Contribution:** Discovered refusal is controlled by single linear direction in activation space
- **Key Findings:**
  - Tested on 13 LLMs (7B-72B parameters)
  - Single direction → erase it, model won't refuse
  - Single direction → enhance it, model refuses everything
  - Mechanistically analyzes how adversarial suffixes suppress refusal
- **Citation in AlignTree:** CORE ASSUMPTION
- **Relevance to RCones:** FOUNDATIONAL for both offense and defense

---

## Extended Refusal Direction Research

### 🟡 **MULTI-DIMENSIONAL REFUSAL**

**Pan et al. (2502.09674)**
- **Title:** [Multi-dimensional refusal structure]
- **Status:** Cited in AlignTree as extension
- **Contribution:** Suggests refusal isn't purely 1D
- **Implication:** AlignTree's SVM features may capture missing dimensions
- **For RCones:** Geometry discovery should validate/refute this

---

### 🟡 **CROSS-LINGUAL UNIVERSALITY**

**Wang et al. (2505.17306)**
- **Title:** [Cross-lingual refusal direction]
- **Status:** Cited in AlignTree
- **Contribution:** Refusal direction transfers across languages
- **Implication:** AlignTree's approach generalizes cross-lingually
- **For RCones:** Vectors likely transfer across languages too

---

### 🟡 **SELF-ORGANIZING MAPS FOR REFUSAL**

**Piras et al. (2511.08379)**
- **Title:** [SOM-based refusal analysis]
- **Status:** Contemporary work
- **Contribution:** Alternative geometric perspective on refusal
- **Implication:** May reveal multi-directional structure
- **For RCones:** Comparison point for geometry discovery

---

## Related Defense Papers (AlignTree Context)

### 🔵 **POST-HOC OUTPUT FILTERING**

**Wu et al. (2402.15727)**
- **Title:** "LLMs Can Defend Themselves Against Jailbreaking in a Practical Manner: A Vision Paper"
- **Method:** Shadow stack checking for harmful tokens in output
- **AlignTree Improvement:** Monitors internals rather than outputs; earlier detection
- **Status:** Published Feb 2024 (earlier than AlignTree)

**Xiong et al. (2405.20099)**
- **Title:** "Defensive Prompt Patch: A Robust and Interpretable Defense of LLMs against Jailbreak Attacks"
- **Method:** Suffix-based prompts to defend against jailbreaks
- **AlignTree Improvement:** Model-agnostic intrinsic defense vs. prompt-based
- **Status:** Published May 2024

---

### 🔵 **MULTI-AGENT & ENSEMBLE DEFENSES**

**Zeng et al. (2403.04783)**
- **Title:** "AutoDefense: Multi-Agent LLM Defense against Jailbreak Attacks"
- **Method:** Multiple LLM agents collaboratively filter responses
- **AlignTree Improvement:** Single-pass vs. multi-pass; lightweight vs. ensemble of models
- **Status:** Published March 2024

---

### 🔵 **PROACTIVE DECEPTION**

**Zhao et al. (2510.05052)**
- **Title:** "Proactive defense against LLM Jailbreak"
- **Method:** Provide misleading "successful jailbreak" responses to fool attackers
- **AlignTree Comparison:** Orthogonal approach; could be combined
- **Status:** Published Oct 2025 (recent)

---

### 🔵 **DECODING-LEVEL DEFENSE**

**Chen et al. (2412.07672)**
- **Title:** "FlexLLM: Exploring LLM Customization for Moving Target Defense on Black-Box LLMs"
- **Method:** Modify decoding hyperparameters dynamically
- **AlignTree Difference:** Requires activation access (white-box); FlexLLM is black-box
- **Status:** Published Dec 2024

---

### 🔵 **REFUSAL DIRECTION STABILIZATION**

**Du et al. (2509.06795)** - [ProCon Paper]
- **Title:** "Anchoring Refusal Direction: Mitigating Safety Risks in Tuning via Projection Constraint"
- **Method:** Regularize refusal direction drift during fine-tuning
- **AlignTree Relation:** Complements detection with direction protection
- **Status:** Published Sept 2025

---

## Related Attack Papers (AlignTree Threat Model)

### 🔴 **OPTIMIZATION-BASED ATTACKS**

**Original GCG Attack**
- **Method:** Gradient-based token optimization
- **AlignTree Tests Against:** Yes, explicitly mentioned
- **Detection Rate:** 92-96% (in paper results)

---

### 🟠 **TEMPLATE-BASED ATTACKS**

- DAN (Do-Anything-Now)
- DAN+ variants
- Role-playing prompts
- **AlignTree Tests Against:** Yes, explicitly mentioned
- **Detection Rate:** 92-96%

---

### 🟠 **OBFUSCATION ATTACKS**

**Andriushchenko & Flammarion (2407.11969)**
- **Title:** "Does Refusal Training in LLMs Generalize to the Past Tense?"
- **Method:** Reformulate harmful requests in past tense
- **AlignTree Robustness:** Should test temporal reformulations
- **Status:** Published July 2024

**Zhang et al. (2507.22171)**
- **Title:** "Enhancing Jailbreak Attacks on LLMs via Persona Prompts"
- **Method:** Genetic algorithm to evolve persona-based jailbreaks
- **AlignTree Robustness:** Should test persona-based attacks
- **Status:** Published July 2025

---

### 🟠 **MULTIMODAL ATTACKS**

**Kim et al. (2505.21556)**
- **Title:** "Benign-to-Toxic Jailbreaking: Inducing Harmful Responses from Harmless Prompts"
- **Method:** Adversarial images induce toxic outputs from benign text
- **AlignTree Limitation:** Text-only (doesn't cover vision-language)
- **Status:** Published May 2025

---

### 🟠 **ADVERSARIAL REPRESENTATION ATTACKS**

**Li et al. (2507.06043)**
- **Title:** "CAVGAN: Unifying Jailbreak and Defense of LLMs via Generative Adversarial Attacks on their Internal Representations"
- **Method:** GAN learns to move harmful prompts into safe activation region
- **AlignTree Relation:** Tests exactly what AlignTree defends against
- **Status:** Published July 2025

---

## Evaluation Frameworks & Benchmarks

### 🟢 **SAFETY BENCHMARKS**

**HarmBench**
- Used for adversarial evaluation
- Mentioned implicitly in AlignTree (ASR evaluation)
- Standard for jailbreak success rate measurement

**JailbreakBench**
- Comprehensive jailbreak benchmark
- Referenced in related papers
- Used for standardized evaluation

---

## Strategic Citation Map for Refusal-Cones

### Group 1: CORE FOUNDATIONS (Must cite)
```
1. Arditi et al. 2406.11717 - Single refusal direction discovery
2. Pan et al. 2502.09674 - Multi-dimensional refusal (if extending)
3. Wang et al. 2505.17306 - Cross-lingual universality
```

### Group 2: DEFENSE BENCHMARKS (Should cite)
```
1. AlignTree 2511.12217 - Detection as evaluation metric
2. Du et al. 2509.06795 - Direction protection during training
3. Zhao et al. 2510.05052 - Proactive defense alternative
```

### Group 3: ATTACK METHODS (For evaluation)
```
1. Original GCG attack
2. DAN template variants
3. Andriushchenko past-tense jailbreaks
4. Zhang et al. persona prompts
```

### Group 4: RELATED GEOMETRIC RESEARCH
```
1. Piras et al. 2511.08379 - SOM-based refusal geometry
2. Li et al. 2507.06043 - GAN-based representation attacks
3. [Any recent multi-dimensional safety papers]
```

---

## How to Cite AlignTree in Refusal-Cones

### In CLAUDE.md Methods Section
```markdown
## Safety Defense Benchmarking

We benchmark our refusal vectors against AlignTree (Goren et al., 2511.12217),
a state-of-the-art detection mechanism that monitors refusal direction signals
and learns non-linear safety features via random forest classification.

This serves two purposes:
1. **Practical relevance:** Evaluate if discovered jailbreaks evade real defenses
2. **Theoretical validation:** Check if refusal direction assumptions hold

**Expected finding:** If Arditi et al. (2406.11717) is correct that refusal is
a single direction, then AlignTree's 92-96% detection should correlate with
our geometric discovery results.
```

### In Related Work Section
```markdown
### Defense Mechanisms

**AlignTree (Goren et al., 2511.12217)** demonstrates that refusal direction
can be operationalized for defense. It achieves 92-96% jailbreak detection by:
- Monitoring refusal direction projections in activation space
- Extracting SVM-based non-linear features
- Classifying with random forest ensemble

This validates that refusal is sufficiently concentrated for practical
extraction and monitoring, supporting our core assumption that geometry
can be discovered and manipulated.
```

### In Experiments Section
```markdown
### Experiment: Robustness Against AlignTree

To evaluate whether discovered refusal vectors represent true vulnerabilities,
we test if RDO-trained jailbreaks evade AlignTree detection.

**Setup:**
- Deploy AlignTree detector (Goren et al., 2511.12217) on jailbroken model
- Measure: Detection rate of our RDO-generated prompts
- Baseline: Detection rate of known attacks (GCG, DAN)

**Expected outcomes:**
- Perfect case: Our vectors completely evade detection
- Realistic: Some evasion, some detection → Measure trade-off
- Conservative: Similar detection rates → Design needs improvement
```

---

## Citation Statistics

### By Year
- 2024: 4 major papers (SELFDEFEND, AutoDefense, Defensive Prompt Patch, FlexLLM)
- 2025: 8+ major papers (AlignTree, ProCon, DeepRefusal, ProAct, etc.)

### By Type
- Foundational: 1 (Arditi 2406.11717)
- Extensions: 3+ (Pan, Wang, Piras on multi-dimensional/universal)
- Defenses: 6+ (SELFDEFEND, AutoDefense, Defensive Prompt Patch, ProAct, FlexLLM, ProCon)
- Attacks: 4+ (GCG variants, persona, past-tense, multimodal)

### Trend
- 2024: Diverse defenses explored
- 2025: Convergence on refusal direction as central mechanism
  - AlignTree leverages it for defense
  - RCones leverages it for research attacks
  - ProCon leverages it for protection during training

---

## For Your Literature Review

### High-Priority Papers to Reference

| Priority | Paper | Reason |
|----------|-------|--------|
| **CRITICAL** | Arditi et al. 2406.11717 | Foundation of all refusal work |
| **HIGH** | AlignTree 2511.12217 | Shows defensive use of refusal direction |
| **HIGH** | Pan et al. 2502.09674 | Questions 1D assumption |
| **MEDIUM** | ProCon (Du et al. 2509.06795) | Shows protection perspective |
| **MEDIUM** | DeepRefusal 2509.15202 | Proactive alignment defense |
| **MEDIUM** | Piras et al. 2511.08379 | Alternative geometric view (SOM) |
| **LOW** | SELFDEFEND, AutoDefense, others | Context for defenses, less directly related |

---

## Recommended Reading Order

1. **Arditi et al. 2406.11717** (foundational, ~30 min)
2. **AlignTree 2511.12217** (this review, ~20 min)
3. **Pan et al. 2502.09674** (geometry extension, ~25 min)
4. **ProCon 2509.06795** (defense perspective, ~25 min)
5. **Piras et al. 2511.08379** (alternative geometry, ~20 min)
6. Other papers as needed for specific questions

---

**Prepared:** 2026-01-19
**Total Papers Analyzed:** 20+
**Review Status:** Complete with citation network mapped
