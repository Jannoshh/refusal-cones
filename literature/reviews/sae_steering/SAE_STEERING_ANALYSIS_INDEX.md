# arXiv:2411.11296 Analysis Index

**Paper:** Steering Language Model Refusal with Sparse Autoencoders

**Quick Links to Analysis Documents:**

## Entry Points (Pick One)

### For Quick Understanding (5-10 minutes)
→ Start with: **[2411_11296_SUMMARY.md](2411_11296_SUMMARY.md)**
- One-paragraph overview
- Key findings for RDO
- Critical questions for your project
- Success criteria

### For Literature Review Integration (20-30 minutes)
→ Start with: **[citation_review_SAE_steering.md](citation_review_SAE_steering.md)**
- Full academic format (how it cites prior work, methods, results)
- Structured sections: citations, key findings, implications
- Comparison to other approaches (ProCon, Arditi, etc.)
- Suitable for copying into your paper

### For Technical Deep-Dive (45-60 minutes)
→ Start with: **[SAE_vs_RDO_technical_analysis.md](SAE_vs_RDO_technical_analysis.md)**
- Why SAE steering fails (mechanistic analysis)
- Why RDO might succeed (theoretical basis)
- Quantitative comparison of perturbation magnitude
- Edge cases and failure modes
- Detailed hypotheses to test

### For Experiment Integration (30-45 minutes)
→ Start with: **[SAE_steering_implications_for_experiments.md](SAE_steering_implications_for_experiments.md)**
- Action items for E1-E6
- Specific code snippets to add
- New experiment (E7: Pareto analysis)
- Measurement checklist before publication
- Integration with existing codebase

---

## Document Purposes

| Document | Purpose | Audience | Time |
|----------|---------|----------|------|
| **2411_11296_SUMMARY** | Executive overview, decision-making | Everyone | 5-10 min |
| **citation_review_SAE_steering** | Academic synthesis, literature context | Researchers, paper authors | 20-30 min |
| **SAE_vs_RDO_technical_analysis** | Mechanistic understanding, theory | Technical team, theorists | 45-60 min |
| **SAE_steering_implications_for_experiments** | Practical action items, code changes | Experiment runners, engineers | 30-45 min |

---

## Key Takeaways Across All Documents

### The Problem (From SAE Paper)
- ✓ SAE steering improves jailbreak robustness (+42pp on Crescendo attacks)
- ✗ Severely degrades capabilities (-46pp on GSM8K, -33pp on MMLU)
- **Root cause:** Refusal features deeply entangled with general language ability

### The Implication for RDO
- Must measure MMLU, GSM8K, TruthfulQA to avoid hidden failures
- Multi-objective training (λ_retain) should prevent SAE's degradation
- If RDO preserves capabilities: major contribution to safety research
- If RDO also degrades: fundamental safety-capability tradeoff confirmed

### The Action Items
1. Modify E1-E6 to measure full capability suite
2. Add λ_retain sensitivity analysis
3. Test on unrelated domains (pure math, facts)
4. Create E7: Pareto frontier analysis
5. Compare results directly to SAE baseline

---

## Navigation by Use Case

### "I want to write a literature review section"
1. Read **citation_review_SAE_steering.md** (complete)
2. Copy-paste into your paper's related work section
3. Adapt citations to your bibliography style

### "I want to understand why SAE steering failed"
1. Start with **2411_11296_SUMMARY.md** (why section)
2. Read **SAE_vs_RDO_technical_analysis.md** (hypotheses 1-2)
3. Understand the mechanistic analysis and entanglement

### "I want to modify my experiments"
1. Start with **SAE_steering_implications_for_experiments.md** (action items)
2. Follow code snippets for each experiment
3. Use measurement checklist before publishing

### "I want to decide if this paper affects my project"
1. Read **2411_11296_SUMMARY.md** (critical questions section)
2. Answer the 5 critical questions for your team
3. Decide: yes, this changes priorities

### "I want the full academic treatment"
1. Read **citation_review_SAE_steering.md** (complete)
2. Read **SAE_vs_RDO_technical_analysis.md** (technical comparison)
3. Then integrate into your paper's contributions section

---

## Comparison Tables (Quick Reference)

### SAE Steering vs. RDO at a Glance

| Metric | SAE Steering | RDO (Expected) |
|--------|-------------|----------------|
| Safety improvement | ✓ +42pp | ✓ TBD |
| Capability preservation | ✗ -46pp | ✓ Expected <10pp |
| Multi-objective | ✗ No | ✓ Yes (λ_retain) |
| Per-layer control | ✗ No | ✓ Yes |
| Deployment ready | ✗ No | ? (Pending eval) |

### Refusal Direction Approaches

| Approach | Method | Capability Cost | Status |
|----------|--------|-----------------|--------|
| Arditi et al. | Single direction (empirical) | Unknown | Proven safe |
| SAE steering | Feature amplification | 46pp loss | Proven unsafe |
| RDO (ours) | Geometric projection + training | <10pp (predicted) | Pending evaluation |

---

## Critical Experiments to Run

From **SAE_steering_implications_for_experiments.md:**

### Essential (Before Publishing)
- [ ] Add MMLU measurement to all E1-E6 results
- [ ] Add GSM8K measurement to all E1-E6 results
- [ ] Add TruthfulQA measurement to all E1-E6 results
- [ ] Show λ_retain prevents degradation (sensitivity analysis)
- [ ] Test on unrelated domains (zero expected degradation)

### Recommended (Before First Conference Submission)
- [ ] E7: Pareto frontier analysis (sweep all λ values)
- [ ] Compare RDO directly to SAE baseline on same model
- [ ] Per-layer capability breakdown (E2 upgrade)
- [ ] Robustness test (Crescendo attacks like SAE paper)

### Nice to Have (Long-term)
- [ ] Align RDO cone with SAE features (validation)
- [ ] Theoretical analysis of why separation is possible
- [ ] Multi-model comparison (Llama, Mistral, etc.)

---

## Key Questions to Answer

From **2411_11296_SUMMARY.md:**

1. **Will RDO suffer SAE's degradation?**
   - Measurement: MMLU (expect <10pp vs SAE's 33pp)

2. **Is multi-objective training sufficient?**
   - Measurement: λ_retain ablation shows prevention

3. **Do RDO directions align with SAE features?**
   - Measurement: Cosine similarity of vectors

4. **How does per-layer targeting help?**
   - Measurement: E3 per-layer vs. global comparison

5. **Can you establish a safety-capability pareto frontier?**
   - Measurement: E7 Pareto sweep results

---

## Timeline Recommendation

**Week 1:** Read and understand (this document + 2411_11296_SUMMARY.md)

**Week 2:** Modify experiments (follow SAE_steering_implications_for_experiments.md)

**Week 3-4:** Run E1-E3 with capability measurements

**Week 5-6:** Analyze results, compare to SAE baseline

**Week 7:** Run E7 (Pareto analysis) if promising so far

**Week 8+:** Write up results in context of SAE steering paper

---

## How These Documents Were Created

All four documents synthesized from:
1. **Paper abstract:** arXiv:2411.11296
2. **Web search results:** Key findings and capability degradation measurements
3. **Comparative analysis:** Against Arditi et al., ProCon, and Refusal-Cones approach
4. **Integration planning:** Your existing experiments (E1-E6) and PEFT infrastructure

Cross-referenced with:
- Your CLAUDE.md (project instructions)
- Your citation_review.md (literature review template)
- Your experiments structure
- Your training infrastructure

---

## When to Update This Index

Update whenever:
- [ ] New related papers appear (add to citation_review_SAE_steering.md)
- [ ] E1-E3 results come in (update 2411_11296_SUMMARY.md with your results)
- [ ] New implications discovered (update implications_for_experiments.md)
- [ ] Methodological changes needed (update technical_analysis.md)

---

## Files Created

```
docs/
├── 2411_11296_SUMMARY.md                      (This paper - executive summary)
├── citation_review_SAE_steering.md             (Literature review format)
├── SAE_vs_RDO_technical_analysis.md            (Technical deep-dive)
├── SAE_steering_implications_for_experiments.md (Action items & code)
└── SAE_STEERING_ANALYSIS_INDEX.md              (This index)
```

All suitable for inclusion in your repository and paper.

---

**Created:** 2026-01-19
**Status:** Ready for immediate use
**Next Step:** Pick an entry point above and start reading!
