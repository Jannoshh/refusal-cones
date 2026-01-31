# Citation Review

This doc is a living review of papers that cite the repo’s base paper (*“The Geometry of Refusal in Large Language Models: Concept Cones and Representational Independence”*).

## How to update

- Run: `python3 scripts/update_citation_review.py`
- Optional env vars:
  - `OPENALEX_MAILTO=you@domain.com` (recommended by OpenAlex)
  - `SEMANTIC_SCHOLAR_API_KEY=...` (optional; higher rate limits)

The script only rewrites the section between the markers below; everything else is for manual synthesis.

<!-- BEGIN AUTO-CITATION-REVIEW -->
_Last updated: 2026-01-19 16:49 UTC_

## Base paper (auto)
- Query: The Geometry of Refusal in Large Language Models: Concept Cones and Representational Independence

## Planned experiments (auto, from `docs/PAPER_PLAN.md`)
- E1.1: Discovery Efficiency
- E1.2: Geometry Quality
- E2.1: Dimensionality by Layer
- E2.2: Per-Layer ASR Analysis
- E2.3: Layer Ablation Study
- E3.1: Affine vs. Separate Operations
- E3.2: Training Efficiency
- E4.1: SFT → RL Pipeline
- E4.2: RL from Different Initializations
- E4.3: Characterizing the Ceiling
- E5.1: Component Ablation
- E5.2: Hyperparameter Sensitivity

## Fetch warnings (auto)
- URLError for https://api.openalex.org/works?search=The+Geometry+of+Refusal+in+Large+Language+Models%3A+Concept+Cones+and+Representational+Independence&per-page=5: <urlopen error [Errno 8] nodename nor servname provided, or not known>
- OpenAlex base paper not found; skipping OpenAlex citers.

## Citing works (auto, n=0)

| Year | Title | Venue | Links | Suggested experiments | Auto findings |
|---:|---|---|---|---|---|

### Details (auto)

<!-- END AUTO-CITATION-REVIEW -->

## Manual synthesis

**Last manual update:** 2026-01-19

### Executive Summary

We identified **25+ papers** building on refusal direction geometry work. Key findings:

| Category | Count | Key Result |
|----------|-------|------------|
| **Defense Methods** | 8 | Refusal geometry enables 35-95% ASR reduction |
| **Attack Methods** | 5 | Multi-dimensional geometry enables surgical jailbreaks |
| **Theoretical Extensions** | 4 | Concept cones generalize to truth, cross-lingual, multimodal |
| **Mechanistic Understanding** | 5 | Harmfulness ≠ refusal; safety in few attention heads |
| **New Modalities** | 3 | Extends to video diffusion, audio LLMs, VLMs |

### High-level takeaways

1. **Multi-dimensionality validated:** Multiple independent teams confirm refusal is NOT single-direction (Truth cones: 2-5D, MEUV: K orthogonal vectors, RepIt: concept-specific decomposition)

2. **Geometry enables both attack AND defense:** Same discovered structure powers surgical jailbreaks and robust defenses

3. **Cross-domain generalization:** Concept cones apply to refusal, truth, video generation, audio LLMs, VLMs

4. **Safety-capability tradeoff addressable:** SAE steering shows -46pp capability loss; our RDO with λ_retain designed to avoid this

5. **Layer and token position matter:** Refusal at generation start, harmfulness at instruction end, safety heads in layers 12-22

### Comparison to our planned experiments

| Experiment | Related Papers | Insights |
|------------|---------------|----------|
| E1 Discovery | COSMIC, SRA | COSMIC's cosine similarity complements gradient discovery; SRA's spectral cleaning for post-processing |
| E2 Per-Layer | Attention Heads (2508.19697), Harmfulness/Refusal (2507.11878) | Safety concentrates in <10% of heads; different token positions for different concepts |
| E3 Unified Affine | AlphaSteer, ProCon | Null-space constraints preserve utility; projection constraints prevent drift |
| E4 RL | SAE Steering (2411.11296) | SAE steering shows capability loss; validate our λ_retain prevents this |
| E5 Ablations | RepIt, MEUV | Concept-specific vs. monolithic; orthogonal decomposition |

### Notes by paper

---

## ProCon: Anchoring Refusal Direction (2509.06795)

**Paper:** Anchoring Refusal Direction: Mitigating Safety Risks in Tuning via Projection Constraint

**Authors:** Yanrui Du, Fenglei Fan, Sendong Zhao, Jiawei Cao, Qika Lin, Kai He, Ting Liu, Bing Qin, Mengling Feng

**Published:** September 8, 2025

### Key Contributions

1. **Problem Identification:** Refusal direction (r-direction) drifts during instruction fine-tuning, compromising safety
2. **Core Method:** Projection-constrained loss that regularizes projection magnitude of hidden states onto r-direction
3. **Solution to Performance Barriers:** Warm-up strategy with early-stage strong constraints + expanded data distribution
4. **Results:** Significantly mitigates safety risks while preserving task performance gains

### How It Relates to Refusal-Cones

| Dimension | ProCon | Refusal-Cones RDO |
|-----------|--------|-------------------|
| **Foundation** | Builds on Arditi et al. 2406.11717 (single refusal direction) | Same foundational work |
| **Goal** | Preserve/stabilize r-direction during fine-tuning | Discover and ablate r-direction for jailbreaks |
| **Constraint Type** | Projection regularization in loss | Adapter-based projection ablation |
| **Training Scenario** | Defense: protect safety during task IFT | Offense/Research: maximize harm under ablation |
| **Geometric Operation** | Constrain projections: `proj_magnitude(h, v) regulated` | Ablation projection: `h' = (I - vv^T)h` |
| **Direction Role** | Stabilize & protect | Identify & manipulate |

### Technical Synergies

1. **Shared Direction Identification:**
   - Both identify r-direction via mean difference: `E[h_harmful] - E[h_harmless]`
   - ProCon constrains it during training; RDO ablates it to measure refusal strength

2. **Projection Mathematics:**
   - ProCon regularizes projection magnitude onto r-direction
   - RDO's projection ablation mathematically equivalent to rank-1 LoRA
   - Both operate on same geometric primitives

3. **Multi-dimensional Safety (Future):**
   - ProCon could extend to Pan et al. (2502.09674) multi-dimensional directions
   - RDO discovery could identify which dimensions to constrain
   - Complementary understanding of safety geometry

4. **Warm-up Insights:**
   - ProCon's two-phase training (sharp early drift → stabilization) applicable to RDO training initialization
   - Early guidance might improve RDO vector quality

### Defensive Learning

ProCon demonstrates that:
- Early-stage r-direction drift is the primary threat vector during fine-tuning
- Geometric constraints more effective than holistic approaches
- This validates RDO's assumption that r-direction is concentrated/identifiable

### Key References for RDO

ProCon cites foundational work directly relevant to refusal-cones:

1. **Arditi et al. 2406.11717** - "Refusal in Language Models Is Mediated by a Single Direction"
   - Core finding: single direction controls refusal
   - Enables both ProCon (protection) and RDO (manipulation)

2. **Multi-dimensional Extensions** (contemporary):
   - Pan et al. 2502.09674 - Multiple dimensions control safety
   - Piras et al. 2511.08379 - Self-Organizing Maps for multi-directional refusal
   - Suggests RDO should consider multi-dimensional geometry

3. **Universal Properties**:
   - Wang et al. 2505.17306 - Cross-lingual universality of r-direction
   - Implies RDO vectors likely transfer across languages

### Potential RDO Extensions

1. **Prevent Unintended Drift:** Apply ProCon constraints to non-target layers during RDO training
2. **Quality Metrics:** Measure r-direction drift as part of RDO evaluation
3. **Multi-objective RDO:** Explicitly protect certain directions while ablating others
4. **Warm-up Strategies:** Use ProCon-style early-phase guidance for RDO initialization

### Notes for Future Work

- **Comparison point:** ProCon shows defensive use of geometric constraints; RDO is offensive equivalent
- **Validation:** ProCon's effectiveness validates that r-direction is indeed critical geometric feature
- **Generalization:** Tests across Llama-2, Qwen-2 show robustness; RDO should test similar models
- **Layer specificity:** ProCon appears to work per-layer; check if RDO layer-specificity matches

---

## Defense Methods

### DeepRefusal (2509.15202)
**"Beyond Surface Alignment: Rebuilding LLMs Safety Mechanism via Probabilistically Ablating Refusal Direction"**

- **Authors:** Xie et al.
- **Key Innovation:** Probabilistically ablates refusal direction across layers during fine-tuning
- **Results:** ~95% ASR reduction across 4 LLM families
- **Mechanism:** Forces models to rebuild safety from jailbreak states
- **Relevance:** Uses our multi-dimensional geometry as defense - distributed refusal is harder to attack

### AlphaSteer (2506.07022)
**"Learning Refusal Steering with Principled Null-Space Constraint"**

- **Authors:** Sheng et al. (USTC)
- **Key Innovation:** Null-space projection ensures zero effect on benign activations
- **Results:** 91.93% defense success rate, 80%+ utility preservation
- **Direct Citation:** Cites our paper as [14]
- **Gap:** Assumes single direction; our multi-modal discovery could extend

### AlignTree (2511.12217)
**"Efficient Defense Against LLM Jailbreak Attacks"**

- **Authors:** Goren et al.
- **Key Innovation:** Random forest using refusal direction + SVM features
- **Results:** 92-96% detection accuracy, 5-10% overhead
- **Relevance:** Practical deployment of refusal geometry for defense

### ALKALI/GRACE (2506.08885)
**"Safeguarding LLMs through Geometric Representation-Aware Contrastive Enhancement"**

- **Key Innovation:** Identifies "latent camouflage" - adversarial prompts embedding near safe ones
- **AVQI Metric:** Cluster separation and compactness for vulnerability detection
- **Results:** 35-39% ASR reduction
- **Insight:** Directional manipulation alone insufficient; manifold structure matters

### Safety Attention Heads (2508.19697)
**"Safety Alignment Should Be Made More Than Just A Few Attention Heads"**

- **Key Innovation:** RDSHA identifies safety-critical heads using refusal direction
- **Finding:** Safety in <10% of attention heads (middle-to-upper layers)
- **AHD Training:** Distributes safety, reducing ASR from 71-100% to 0-21%
- **Bridge:** Connects geometric (cones) and mechanistic (heads) perspectives

### RCS for VLMs (2512.12069)
**"Rethinking Jailbreak Detection with Representational Contrastive Scoring"**

- **Key Innovation:** Contrastive scoring using internal geometry for VLM jailbreak detection
- **Results:** 99.36% AUROC with Mahalanobis + FLAVA
- **Relevance:** Extends geometric safety to multimodal domain

### False Refusal Mitigation (2410.03415)
**"Surgical, Cheap, and Flexible: Mitigating False Refusal"** (ICLR 2025)

- **Key Innovation:** Separates false refusal vector from true refusal via orthogonalization
- **Results:** 50.8% false refusal reduction while preserving safety
- **Insight:** Refusal geometry has multiple distinct components

---

## Attack Methods & Understanding

### RepIt (2509.13281)
**"Steering Language Models with Concept-Specific Refusal Vectors"**

- **Authors:** Siu et al. (Berkeley)
- **Key Innovation:** Isolates concept-specific refusal vectors via partial orthogonalization
- **Results:** Target ASR 0.4-0.7 while preserving refusal on 21 non-target concepts
- **Finding:** Edits localize to 100-200 neurons (3.8-5.1% of hidden state)
- **Direct Citation:** Cites concept cones; operationalizes multi-dimensional structure
- **Dual-Use:** Enables evaluation evasion (pass benchmarks, harbor narrow jailbreaks)

### MEUV (2509.12221)
**"Mutually Exclusive Unlock Vectors for Fine-Grained Capability Activation"**

- **Key Innovation:** Factorizes monolithic refusal into K topic-aligned orthogonal vectors
- **Results:** ≥87% ASR with 90% cross-topic leakage reduction
- **Cross-Lingual:** Chinese→English transfer works (language-agnostic refusal subspace)
- **Relevance:** Validates multi-dimensional structure; adds topic-specific decomposition

### Surgical Refusal Ablation (2601.08489)
**"Disentangling Safety from Intelligence via Concept-Guided Spectral Cleaning"**

- **Key Innovation:** Spectral residualization to clean polysemantic refusal vectors
- **Results:** 47× distribution drift reduction (KL 2.088→0.044), 0-2% refusal
- **"Ghost Noise":** Much capability damage is spectral bleeding, not inherent tradeoff
- **Application:** Post-processing for discovered modes to minimize collateral damage

### Adversarial Suffix Transfer (2510.22014)
**"Toward Understanding the Transferability of Adversarial Suffixes"**

- **Three Properties Predict Transfer:** (1) Refusal connectivity, (2) Suffix push, (3) Orthogonal shift
- **Results:** Interventions improve GCG ASR by 37-83%
- **Finding:** Semantic similarity weakly correlates; geometry dominates transfer
- **Insight:** Both parallel AND orthogonal dimensions matter

### Agent Ablation (2410.10871)
**"Applying Refusal-Vector Ablation to Llama 3.1 70B Agents"**

- **Critical Finding:** Unmodified Llama 3.1 70B completes 64% of harmful agentic tasks
- **Post-Ablation:** 93% harmful task completion
- **Insight:** Safety fine-tuning doesn't generalize to agentic behavior
- **Relevance:** Refusal direction generalizes to agentic contexts

---

## Theoretical Extensions

### Truth Cones (2505.21800) - **DIRECT EXTENSION**
**"From Directions to Cones: Exploring Multidimensional Representations of Propositional Facts"**

- **Authors:** Yu et al.
- **Key Innovation:** **Explicitly extends concept cones from refusal to TRUTH**
- **Results:** Multi-dimensional truth cones (2-5D) with same properties
- **Direct Quote:** "extend the concept cone framework, recently introduced for modeling refusal"
- **Uses:** Our three-term loss, methodology, and terminology
- **Implication:** Concept cones are general framework for boolean behaviors

### Cross-Lingual Universality (2505.17306)
**"Refusal Direction is Universal Across Safety-Aligned Languages"**

- **Key Innovation:** English refusal vector bypasses safety in 13+ other languages
- **Bidirectionality:** Any language → any other (not just English-centric)
- **Mechanism:** Parallelism of refusal vectors in embedding space
- **Implication:** Single discovery pass (in English) may suffice for multilingual models

### Harmfulness vs. Refusal (2507.11878)
**"LLMs Encode Harmfulness and Refusal Separately"**

- **Key Innovation:** Two distinct directions at different token positions
  - Harmfulness: at instruction end (t_inst)
  - Refusal: at generation start (t_post-inst)
- **Causal Evidence:** Steering harmfulness changes judgment; steering refusal changes output
- **Jailbreak Insight:** Most jailbreaks suppress refusal, leave harmfulness intact
- **Latent Guard:** Uses harmfulness direction for robust detection

### Geometric Unification (2512.07355)
**"A Geometric Unification of Concept Learning with Concept Cones"**

- **Domain:** Vision models (NOT LLMs) - parallel discovery
- **Key Innovation:** Unifies CBMs and SAEs via cone geometry
- **Finding:** Both methods learn cones; differ in selection mechanism
- **Note:** Does NOT cite our paper; independent development

---

## New Modalities

### Video Unlearning (2506.07891)
**"Video Unlearning via Low-Rank Refusal Vector"**

- **Key Innovation:** First refusal vectors for VIDEO diffusion models
- **Method:** Low-rank cPCA factorization (only 0.8% of embedding space)
- **Results:** 69.6% gore reduction, 40% copyright reduction; quality preserved
- **Data Efficiency:** Only 5 safe/unsafe pairs required

### COSMIC (2506.00085)
**"Generalized Refusal Direction Identification in LLM Activations"**

- **Key Innovation:** Output-independent direction selection via cosine similarity
- **Advantage:** Works in adversarial settings where output provides no signal
- **Complementary:** We discover directions; COSMIC selects optimal layer/position

### SAE Steering (2411.11296)
**"Steering Language Model Refusal with Sparse Autoencoders"**

- **Critical Finding:** Safety +42pp on Crescendo BUT -46pp GSM8K, -33pp MMLU
- **Root Cause:** Refusal features deeply entangled with general capabilities
- **Implication:** Our multi-objective RDO (λ_retain) explicitly addresses this tradeoff
- **Validation Need:** Must measure MMLU/GSM8K on RDO models

---

## Summary Table

| Paper | Category | Key Result | Direct Citation |
|-------|----------|------------|-----------------|
| DeepRefusal | Defense | 95% ASR reduction | Indirect |
| ProCon | Defense | Drift prevention | Indirect |
| AlphaSteer | Defense | 91.93% defense, 80% utility | **Yes [14]** |
| AlignTree | Defense | 92-96% detection | Indirect |
| ALKALI | Defense | 35-39% ASR reduction | Indirect |
| Attention Heads | Defense | Safety in <10% heads | Indirect |
| RCS | Defense | 99.36% AUROC for VLMs | Indirect |
| False Refusal | Defense | 50.8% false refusal reduction | Indirect |
| RepIt | Attack | Concept-specific jailbreaks | **Yes** |
| MEUV | Attack | 87% ASR, 90% less leakage | Indirect |
| SRA | Attack | 47× less drift | Indirect |
| Suffix Transfer | Attack | 37-83% ASR improvement | Indirect |
| Agent Ablation | Attack | 93% agentic jailbreak | Indirect |
| Truth Cones | Theory | **Direct extension** | **Yes** |
| Cross-Lingual | Theory | Language universality | Indirect |
| Harmfulness/Refusal | Theory | Two separate directions | Indirect |
| Video Unlearning | Modality | Video diffusion | Indirect |
| COSMIC | Method | Output-independent selection | Indirect |
| SAE Steering | Mech. Interp | Safety-capability tradeoff | Indirect |

---

## Actionable Priorities

Based on review of all citing papers against our multi-layer geometry discovery with λ_retain:

### Do This

| Paper | Action | Priority |
|-------|--------|----------|
| **Harmfulness/Refusal (2507.11878)** | Extract activations at both instruction-end AND generation-start positions. May find two separate cones. | **High** |

### Not Useful For Us

| Paper | Why Not |
|-------|---------|
| **COSMIC** | Output-independent measurement unlikely to improve over our gradient-based approach |
| **SRA spectral cleaning** | λ_retain loss already handles capability preservation; no need for post-processing |
| **MEUV orthogonality** | Their K-orthogonal assumption already proven wrong by prior work |
| **Cross-lingual** | Out of scope; not pursuing multilingual evaluation |

### Key Insight: Token Position Matters

From Harmfulness/Refusal paper (2507.11878):
- **Refusal direction**: Located at generation start (t_post-inst)
- **Harmfulness direction**: Located at instruction end (t_inst)

Current discovery extracts at last token only. Should extract at multiple positions to find complete geometry.

```python
# TODO: Update discovery to extract at multiple positions
positions = {
    'instruction_end': find_instruction_end(prompt),
    'generation_start': find_instruction_end(prompt) + 1,
}
```

---

## Research Opportunities (Deprioritized)

These are interesting but not immediately actionable:

1. Extend to video/audio using our methodology (if multimodal becomes relevant)
2. Build on AlignTree for practical deployment (defense application)
3. Compare with RepIt's concept-specific decomposition (if pursuing targeted jailbreaks)
