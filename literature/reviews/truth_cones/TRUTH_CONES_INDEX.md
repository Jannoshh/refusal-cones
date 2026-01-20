# Truth Cones Research Index

## Paper Reference
**Title:** From Directions to Cones: Exploring Multidimensional Representations of Propositional Facts in LLMs

**Authors:** Stanley Yu, Vaidehi Bulusu, Oscar Yasunaga, Clayton Lau, Cole Blondin, Sean O'Brien, Kevin Zhu, Vasu Sharma

**arXiv ID:** 2505.21800

**Publication Date:** May 2025

**Foundation Work:** Wollschläger et al., "The Geometry of Refusal in Large Language Models: Concept Cones and Representational Independence" (arXiv:2502.17420, February 2025)

---

## Document Guide

### 1. TRUTH_CONES_ANALYSIS.md (COMPREHENSIVE)
**Best for:** Deep literature review, publication writing, complete understanding

**Contents:**
- 10 detailed sections with full evidence mapping
- Citation genealogy showing how this paper extends 2502.17420
- All four experimental findings with interpretation
- Framework validation with supporting evidence
- Research implications for refusal/safety, mechanistic interpretability, adversarial ML
- Complete limitations and future directions
- Structured reference sections

**Sections:**
1. Explicit Citations & Relationship to Foundation Work
2. Key Innovation: Extending Concept Cones to Truth
3. Evidence for Multi-Dimensional Truth Cones
4. Causal Intervention Methodology
5. Framework Validation: How This Extends Concept Cones
6. Connections to Project Codebase
7. Structured Literature Review Summary
8. Final Summary for Literature Review
9. Complete Citation Reference
10. Key Takeaway & Implications

**Word count:** ~2000 | **Reading time:** 15-20 minutes

---

### 2. TRUTH_CONES_BRIEF_SUMMARY.md (1-PAGE OVERVIEW)
**Best for:** Quick reference, literature review citations, presentations

**Contents:**
- Paper core in 1 sentence
- How it cites/builds on Wollschläger (2502.17420)
- Key innovation with table of evidence
- Three lines of evidence (causal, generalization, surgical)
- Framework validation summary
- Where cones fail (sentiment/toxicity)
- Comparison matrix with refusal cones
- Relevance checklist

**Key tables:**
- ASR across cone dimensions (Table 1 from paper)
- Layer localization patterns
- Model comparison matrix

**Word count:** ~800 | **Reading time:** 5-7 minutes

---

### 3. TRUTH_CONES_KEY_EXCERPTS.md (DIRECT QUOTATIONS)
**Best for:** Academic citations, verifying claims, direct evidence

**Contents:**
- All key passages from paper organized by section
- 10 explicit citations of Wollschläger (2502.17420)
- Framework inheritance shown through verbatim quotes
- Three-term loss definitions with original notation
- Multi-dimensionality evidence from experiments
- Orthogonality evidence (Tables 3 & 8)
- Surgical precision findings
- Layer localization data
- Limitations and boundary cases
- Complete bibliography reference

**Organized by:**
- Explicit citation & framework extension
- Three-term loss optimization
- Evidence for multi-dimensionality
- Critical distinction (orthogonality to DIM)
- Surgical precision
- Layer localization
- Framework limitations
- Future directions

**Use cases:** Copy-paste for papers, verify interpretations, find exact quotes

---

### 4. TRUTH_CONES_METHODOLOGY_COMPARISON.md (TECHNICAL DEEP-DIVE)
**Best for:** Implementation, methodology comparison, extending framework

**Contents:**
- Side-by-side comparison of refusal vs. truth methodologies
- Initial direction discovery (DIM adaptation)
- Multi-dimensional concept cones framework
- Three-term loss breakdown (with code-like pseudocode)
- Causal interventions (activation addition & ablation)
- Evaluation metrics mapping
- Monte Carlo evaluation procedure
- Retention/fidelity mechanisms
- Discovered dimensionality patterns
- Layer localization comparison
- Framework boundary analysis
- Implementation template in Python

**Sections:**
1. Overview
2. Initial Direction Discovery
3. Multi-Dimensional Extension: Concept Cones
4. Three-Term Loss Optimization
5. Causal Interventions
6. Evaluation Metrics
7. Monte Carlo Evaluation of Cone Space
8. Retention/Fidelity Mechanism
9. Search Algorithm
10. Discovered Dimensionality
11. Layer Localization
12. Framework Boundary & Limitations
13. Comparison Matrix
14. Implementation Template

**Python code:** Ready-to-implement truth cone discovery class

**Word count:** ~2500 | **Reading time:** 20-25 minutes

---

## Quick Navigation

### By Use Case

**I need to cite this paper in my writing:**
1. Start with **TRUTH_CONES_BRIEF_SUMMARY.md** (structure & key claims)
2. Reference **TRUTH_CONES_KEY_EXCERPTS.md** (exact quotes)
3. Elaborate with **TRUTH_CONES_ANALYSIS.md** (full context)

**I need to understand the methodology:**
1. Read **TRUTH_CONES_METHODOLOGY_COMPARISON.md** (side-by-side with refusal)
2. Check **TRUTH_CONES_KEY_EXCERPTS.md** (three-term loss definitions)
3. Reference **TRUTH_CONES_ANALYSIS.md** (experimental sections 3-5)

**I want to implement truth cones:**
1. Start with **TRUTH_CONES_METHODOLOGY_COMPARISON.md** (Section 13: Implementation template)
2. Reference **TRUTH_CONES_KEY_EXCERPTS.md** (loss definitions)
3. Consult **TRUTH_CONES_ANALYSIS.md** (experimental details)

**I need comprehensive understanding:**
1. **TRUTH_CONES_ANALYSIS.md** → Complete picture
2. **TRUTH_CONES_METHODOLOGY_COMPARISON.md** → Technical details
3. **TRUTH_CONES_KEY_EXCERPTS.md** → Verify claims
4. **TRUTH_CONES_BRIEF_SUMMARY.md** → Quick reference

**I'm checking relation to 2502.17420:**
1. **TRUTH_CONES_ANALYSIS.md** Section 1 (explicit citations)
2. **TRUTH_CONES_KEY_EXCERPTS.md** (6 citations throughout paper)
3. **TRUTH_CONES_METHODOLOGY_COMPARISON.md** (framework inheritance)

---

## Key Findings at a Glance

### What the Paper Does
- Extends Wollschläger et al.'s concept cone framework from **refusal** to **truth** domain
- Uses identical three-term loss optimization methodology
- Tests whether truth is multi-dimensional (like refusal proven to be)
- Discovers 2-5D truth cones in multiple models

### Key Evidence
1. **Causal interventions work:** 100% ASR on larger models at 5D
2. **Generalizes across models:** 5 models tested (Qwen 3B-14B, Gemma 2B-9B)
3. **Surgical precision:** KL divergence < 0.05 on unrelated tasks
4. **Orthogonal to linear methods:** v₂₊ have < 10⁻⁹ cosine similarity to DIM direction

### Validation of Concept Cones
- Shows concept cones work for **boolean/binary properties** (refusal AND truth)
- Suggests concept cones may be **general framework** for safety properties
- Framework boundaries identified: fails on gradient properties (sentiment, toxicity)

### Implications for Your Project
- **Validates** multi-dimensional refusal geometry framework
- **Extends** scope to other properties
- **Identifies** framework limitations (boolean only)
- **Strengthens** publication case (general framework beyond refusal)

---

## Citation Information

**APA Format:**
Yu, S., Bulusu, V., Yasunaga, O., Lau, C., Blondin, C., O'Brien, S., ... & Sharma, V. (2025). From directions to cones: Exploring multidimensional representations of propositional facts in LLMs. *arXiv preprint arXiv:2505.21800*.

**BibTeX:**
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

**Foundation Work:**
```bibtex
@article{wollschlager2025concept_cones,
  title={The Geometry of Refusal in Large Language Models: Concept Cones
         and Representational Independence},
  author={Wollschläger, Tilman and Elstner, Jonas and Geisler, Samuel
          and Cohen-Addad, Vincent and Günnemann, Stephan and Gasteiger, Johannes},
  journal={arXiv preprint arXiv:2502.17420},
  year={2025}
}
```

---

## Cross-References

### Papers Referenced
- **Arditi et al. (2024):** "Refusal in language models is mediated by a single direction"
  - Showed refusal is 1D direction
  - Yu et al. extends to multi-dimensional

- **Marks & Tegmark (2024):** "The geometry of truth: Emergent linear structure in large language model representations"
  - Showed truth is 1D direction
  - Yu et al. extends to multi-dimensional

- **Wollschläger et al. (2502.17420):** "The geometry of refusal in large language models: Concept cones and representational independence"
  - Introduced concept cone framework
  - Yu et al. applies to truth domain

### Related Work in Your Repository
- `/src/discovery/gradient_discovery.py` - Your gradient-based discovery
- `/src/discovery/adaptive_geometry_discovery.py` - Your GP-based methods
- `/CLAUDE.md` - Your project overview citing 2502.17420
- `/docs/discovery/GRADIENT_BASED_DISCOVERY.md` - Methodology documentation

---

## Statistics & Metrics

### Paper Coverage
- **Total sections in paper:** 7 (Intro, Background, Methodology, Experiments, Discussion, Conclusion, Limitations)
- **Total experiments:** 4 (Layer localization, Dimensionality testing, Retention, DIM alignment)
- **Models tested:** 5 (Qwen: 3B, 7B, 14B; Gemma: 2B, 9B)
- **Datasets:** 3 (cities, element_symb, animals_class) + Alpaca for retention
- **Cone dimensions tested:** 1-5D

### Analysis Coverage
- **Document count:** 4 comprehensive documents
- **Total lines:** 1,230 lines across all documents
- **Key excerpts:** 20+ direct quotations
- **Explicit citations of 2502.17420:** 6+ locations
- **Tables analyzed:** 8 (results + supplementary)

---

## Checklist: What You Now Have

- [x] Complete paper understanding
- [x] Explicit citation evidence (6 locations)
- [x] Framework inheritance documentation
- [x] Three-term loss breakdown (pseudocode + definition)
- [x] Experimental evidence (all 4 experiments)
- [x] Causal methodology (activation addition & ablation)
- [x] Multi-dimensional proof (Tables 1, 3, 8)
- [x] Layer localization validation
- [x] Surgical precision evidence (KL divergence)
- [x] Framework limitations (sentiment/toxicity failures)
- [x] Implementation template (Python code ready)
- [x] Comparison with refusal cones
- [x] Implications for your project
- [x] Citations (APA & BibTeX)

---

## Recommendations

### For Literature Review
Use in this order:
1. **TRUTH_CONES_BRIEF_SUMMARY.md** - For structure and claims
2. **TRUTH_CONES_KEY_EXCERPTS.md** - For exact citations
3. **TRUTH_CONES_ANALYSIS.md** Section 7 - For structured summary

### For Project Development
1. **TRUTH_CONES_METHODOLOGY_COMPARISON.md** Section 13 - Start implementation
2. **TRUTH_CONES_KEY_EXCERPTS.md** - Reference exact loss definitions
3. **TRUTH_CONES_ANALYSIS.md** Sections 3-4 - Understand experimental design

### For Publication
1. **TRUTH_CONES_ANALYSIS.md** - Full content for your paper
2. **TRUTH_CONES_KEY_EXCERPTS.md** - Verify all citations
3. **TRUTH_CONES_BRIEF_SUMMARY.md** - Abstract/intro reference

---

## Questions This Analysis Answers

**About Citations:**
- ✓ How many times does Yu et al. cite 2502.17420? (6+ locations)
- ✓ What framework does it inherit? (Three-term loss, causal methodology)
- ✓ How does it extend the work? (Applies to truth instead of refusal)

**About Innovation:**
- ✓ What's the key contribution? (Multi-dimensional truth cones, framework generalization)
- ✓ What's the evidence? (Four experiments with quantitative results)
- ✓ How does it validate concept cones? (Shows they work across domains)

**About Methodology:**
- ✓ What causal interventions are used? (Activation addition & directional ablation)
- ✓ How is dimensionality discovered? (Gradient-based optimization with orthogonality)
- ✓ How is precision ensured? (Three-term loss with retention term)

**About Limitations:**
- ✓ Where does the framework fail? (Gradient properties: sentiment, toxicity)
- ✓ What's unexplored? (Semantic interpretability of basis vectors)
- ✓ What scale issues exist? (Smaller models have simpler geometry)

**About Your Project:**
- ✓ Does this validate your approach? (Yes, framework generalizes)
- ✓ Can you extend it to other domains? (Yes, to other boolean properties)
- ✓ What are limitations? (Framework works for discrete, not gradient properties)

---

## Version & Last Updated

**Document Set Version:** 1.0
**Last Updated:** 2026-01-19
**Analysis Scope:** Complete paper (arXiv:2505.21800)
**Related Foundation:** Wollschläger et al. (arXiv:2502.17420)

---

## Access Notes

All documents are available in `/docs/`:
- `/docs/TRUTH_CONES_ANALYSIS.md`
- `/docs/TRUTH_CONES_BRIEF_SUMMARY.md`
- `/docs/TRUTH_CONES_KEY_EXCERPTS.md`
- `/docs/TRUTH_CONES_METHODOLOGY_COMPARISON.md`
- `/docs/TRUTH_CONES_INDEX.md` (this file)

All documents are standalone but cross-referenced.
Total reading time: 45-60 minutes (all documents)
Recommended entry point: TRUTH_CONES_BRIEF_SUMMARY.md (5-7 minutes)
