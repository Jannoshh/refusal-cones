# SAE Steering Paper (2411.11296): Implications for Refusal-Cones Experiments

**Quick Reference:** How findings from "Steering Language Model Refusal with Sparse Autoencoders" affect your experimental roadmap

---

## Critical Discovery: The Safety-Capability Tradeoff

### The Core Finding

**SAE steering improves safety but destroys capabilities:**

| Metric | Baseline | With SAE Steering | Change |
|--------|----------|-------------------|--------|
| Refusal on Crescendo attacks | 55.92% ASR | 32.58% ASR | +23.34pp safety ✓ |
| MMLU (knowledge) | 68.80% | 35.98% | -32.82pp capability ✗ |
| GSM8K (math) | 82.50% | 35.56% | -46.94pp capability ✗ |
| TruthfulQA (factuality) | 65.00% | 53.82% | -11.18pp capability ✗ |

### Why This Matters for Your Project

**Key implication:** If RDO training doesn't explicitly address the capability problem, you may achieve:
- ✓ High attack success rate on harmful prompts
- ✗ Degraded performance on MMLU, math, factuality
- ✗ Model becomes less useful overall

**This is a showstopper for practical deployment.**

---

## Immediate Action Items for Your Experiments

### Action 1: Add Capability Measurements to Every Experiment

**Current state:** Your planned experiments (E1-E6) focus on discovery quality and jailbreak performance

**Required addition:** Measure MMLU, GSM8K, TruthfulQA alongside safety metrics

**Why:** SAE paper discovered capability loss only because they ran full benchmark suite

**Implementation:**
```python
# In experiments/e2_per_layer/run_quality.py and others:

# Current:
compliance_rate = evaluate_compliance(trained_model, harmful_data)

# Add:
mmlu_score = evaluate_mmlu(trained_model)  # Before and after RDO
gsm8k_score = evaluate_gsm8k(trained_model)
truthfulqa_score = evaluate_truthfulqa(trained_model)

# Log both:
results = {
    'jailbreak_asr': compliance_rate,
    'mmlu': mmlu_score,
    'mmlu_degradation': baseline_mmlu - mmlu_score,
    'gsm8k': gsm8k_score,
    'truthfulqa': truthfulqa_score,
}
```

**Expected outcome:** If RDO is working correctly, you should see:
- MMLU degradation << 10pp (much better than SAE's 32pp)
- GSM8K degradation << 15pp (much better than SAE's 46pp)
- TruthfulQA degradation << 5pp (much better than SAE's 11pp)

### Action 2: Instrument λ_retain Sensitivity Analysis

**Current state:** You have λ_ablate, λ_add, λ_retain parameters

**Required experiment:** Systematic sweep showing λ_retain prevents degradation

**Why:** Multi-objective training should prevent the entanglement SAE suffered

**Implementation:**
```python
# New experiment: e_capability_tradeoff/run_lambda_sweep.py

lambda_retain_values = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 2.0]

for lambda_retain in lambda_retain_values:
    trained_model = train_rdo_with_peft(
        # ...
        lambda_ablate=1.0,
        lambda_add=1.0,
        lambda_retain=lambda_retain,
        # ...
    )

    results[lambda_retain] = {
        'jailbreak_asr': measure_jailbreak(trained_model),
        'mmlu': measure_mmlu(trained_model),
        'mmlu_degradation': baseline - measure_mmlu(trained_model),
    }

# Plot: λ_retain vs. capability_degradation
# Expected: degradation decreases as λ_retain increases
```

**Success criterion:** Clear inverse relationship between λ_retain and capability degradation

### Action 3: Test on Unrelated Domains

**Current state:** You test on harmful prompts and some benchmarks

**Required addition:** Pure mathematics, science facts, history (no refusal signal)

**Why:** SAE degradation appeared even on unrelated domains (philosophy, chemistry)
- Suggests refusal feature was entangled with general language ability
- RDO should have zero effect on truly unrelated domains

**Implementation:**
```python
# Add to evaluation suite:

domains = {
    'refusal_related': [
        'harmful_prompt_compliance',  # Current
        'safety_qa',
    ],
    'unrelated': [
        'pure_math_solving',       # NEW: 5 + 5 = ?
        'factual_qa',              # NEW: capital of France?
        'chemistry_knowledge',      # NEW: atomic number of oxygen?
        'history_facts',           # NEW: year of WW2?
        'language_understanding',  # NEW: grammatical judgment
    ]
}

for domain, tests in domains.items():
    for test in tests:
        score_baseline = evaluate_on_domain(baseline_model, test)
        score_rdo = evaluate_on_domain(trained_model, test)
        degradation = score_baseline - score_rdo

        print(f"{domain}/{test}: {degradation:+.1f}pp")
        # Expected: unrelated domains show ~0pp degradation
```

**Success criterion:** Unrelated domains show <1pp degradation, refusal domains show >10pp improvement

---

## Experiment-by-Experiment Implications

### E1: Discovery Efficiency (`e1_discovery/`)

**Current focus:** How many measurements needed to find refusal geometry

**New requirement:** Compare discovered geometry to SAE features
```python
# After discovery:
rdo_discovered_directions = results['modes']  # e.g., 3 vectors

# Compare to SAE:
sae_features = load_sae_features(model_name)  # Top 10 refusal features
cosine_sims = [cosine(rdo_direction, sae_feature) for each combo]

# If discovered directions align with SAE features:
#   - Validates both approaches
#   - Suggests refusal is indeed localized
# If not:
#   - SAE features may be entangled
#   - RDO discovers cleaner separation
```

### E2: Per-Layer Analysis (`e2_per_layer/`)

**Current focus:** Which layers have refusal information

**New requirement:** Layer-specific capability measurements
```python
# Current:
for layer in layers:
    asr[layer] = measure_asr_per_layer(trained_model, harmful_data)

# Add:
for layer in layers:
    mmlu_score[layer] = measure_mmlu_per_layer(trained_model)
    degradation[layer] = baseline_mmlu - mmlu_score[layer]

# Expected pattern:
#   - Middle layers: high ASR, low degradation (refusal is concentrated)
#   - Lower layers: low ASR, minimal degradation (no refusal signal)
#   - Upper layers: moderate ASR, moderate degradation (output preparation)
```

**Insight:** If degradation is concentrated in middle layers (like refusal), it suggests refusal and capability are entangled. If degradation is uniform, indicates feature steering artifacts.

### E3: Affine vs. Separate Operations (`e3_unified_affine/`)

**Current focus:** Joint affine transforms vs. per-layer

**New requirement:** Compare with SAE steering's single global feature
```python
# SAE paper: single feature 22373 globally amplified
# RDO options:
#   1. Single global direction (similar to SAE)
#   2. Per-layer separate directions (current plan)
#   3. Unified affine (E3 focus)

# Hypothesis: per-layer >> global (like RDO >> SAE)
# because refusal is layer-specific, not global

results = {
    'global': train_rdo(per_layer=False),    # Like SAE
    'per_layer': train_rdo(per_layer=True),  # Proposed RDO
}

# Measure:
for mode, model in results.items():
    asr = measure_asr(model)
    mmlu_degradation = measure_mmlu_degradation(model)
    print(f"{mode}: ASR={asr:.1%}, MMLU degradation={mmlu_degradation:.1f}pp")

# Expected: per_layer >> global in capability preservation
```

### E4: RL Optimization (`e4_rl_optimization/`)

**Current focus:** RL on top of RDO adapters

**New critical role:** RL explicitly optimizes safety-capability tradeoff
```python
# SAE steering problem: no explicit capability objective
# RDO solution: RL with dual reward

reward = w_safety * safety_reward(response) + w_capability * capability_reward(response)

# Where:
#   safety_reward = 1 if refuses harmful, 0 otherwise
#   capability_reward = MMLU score, math score, factuality score

# Optimize: find weights (w_safety, w_capability) that maximize both

# If successful: can achieve pareto frontier where
#   - Safety >> baseline
#   - Capability ≈ baseline (not degraded)
```

**Critical experiment:** Show that E4 RL recovers the capability loss from E3

### E5: Ablations (`e5_ablations/`)

**Current focus:** Component importance

**New requirement:** Capability impact of each component
```python
# Current ablations: remove each component, measure ASR

# Add: measure MMLU degradation for each ablation

ablations = [
    'no_gradient_discovery',  # Use random directions
    'no_multi_objective',     # Set lambda_retain = 0
    'no_per_layer',           # Single global direction
    'no_sparse_gp',           # Standard GP instead
]

for ablation in ablations:
    model = train_rdo(**ablation_config(ablation))
    asr = measure_asr(model)
    mmlu_deg = measure_mmlu_degradation(model)

    # Hypothesis: "no_multi_objective" will show largest MMLU degradation
    #   (like SAE steering), confirming that lambda_retain is critical
```

**Key finding:** If ablating `lambda_retain` causes large MMLU degradation, it proves multi-objective training prevents the capability loss SAE suffered

### E6: Unsupervised Methods (`e6_unsupervised/`)

**Current focus:** Methods without labeled harmful/harmless data

**New requirement:** Unsupervised methods still need capability preservation
```python
# Unsupervised methods: discover refusal without explicit labels

# But still measure:
mmlu_degradation = measure_mmlu(unsupervised_rdo_model)

# Expected: unsupervised methods trade off some safety for capabilities
# But should still be much better than SAE's 46pp MMLU loss
```

---

## New Experiment to Add: Capability-Safety Pareto Analysis

**Proposal:** Comprehensive experiment mapping safety-capability tradeoff space

**Why:** SAE paper shows simple steering breaks this tradeoff. Need to understand RDO's position.

**Execution:**
```python
# experiments/e_capability_safety_pareto/run_pareto.py

lambda_values = {
    'lambda_ablate': [0.0, 0.5, 1.0, 2.0],
    'lambda_add': [0.0, 0.5, 1.0, 2.0],
    'lambda_retain': [0.0, 0.25, 0.5, 1.0, 2.0],
}

results = {}

for lam_ab in lambda_values['lambda_ablate']:
    for lam_ad in lambda_values['lambda_add']:
        for lam_ret in lambda_values['lambda_retain']:
            model = train_rdo(
                lambda_ablate=lam_ab,
                lambda_add=lam_ad,
                lambda_retain=lam_ret,
            )

            safety_score = measure_asr(model)
            capability_score = measure_mmlu(model)

            results[(lam_ab, lam_ad, lam_ret)] = {
                'safety': safety_score,
                'capability': capability_score,
            }

# Plot 3D surface: lambda_ablate × lambda_add → (safety, capability)
# For each λ_retain

# Expected:
#   - λ_retain=0: high safety, low capability (like SAE)
#   - λ_retain=1: moderate safety, high capability (the sweet spot)
#   - λ_retain=2: lower safety, very high capability (over-constrained)
```

**Deliverable:** Pareto frontier showing optimal (safety, capability) pairs

---

## Integration with Existing Codebase

### Where to Measure Capabilities

**Add to:** `src/measurement/vllm_hybrid_measurement.py`

```python
class CapabilityMeasurement:
    """Measure model capability on standard benchmarks."""

    def measure_mmlu(self, model, subset='dev', n_samples=100):
        """MMLU accuracy on subset of questions."""
        # Load MMLU dataset
        # Run inference with model
        # Return accuracy

    def measure_gsm8k(self, model, n_samples=100):
        """GSM8K math problem solving."""
        # Similar pattern

    def measure_truthfulqa(self, model):
        """TruthfulQA factuality score."""
        # Similar pattern

    def measure_domain_specific(self, model, domain):
        """Generic domain testing."""
        # Test pure math, science, history, etc.
```

### Where to Store Results

**Add to:** `experiments/configs/`

```yaml
# experiments/configs/capability_evaluation.yaml
capability_benchmarks:
  mmlu:
    enabled: true
    subset: 'dev'
    n_samples: 100
  gsm8k:
    enabled: true
    n_samples: 100
  truthfulqa:
    enabled: true
  domain_specific:
    - math_algebra
    - science_chemistry
    - history_facts
    - language_grammar

# Store degradation threshold as pass/fail criterion
degradation_threshold:
  mmlu: 10  # pp
  gsm8k: 15
  truthfulqa: 5
```

### Update Experiment Runner

**Modify:** `experiments/run_all.py`

```python
def run_experiment_with_capability_check(experiment_config):
    """Run experiment and validate capability preservation."""

    # 1. Train
    trained_model = train_rdo(**experiment_config)

    # 2. Measure safety (current)
    safety_metrics = measure_safety(trained_model)

    # 3. Measure capability (NEW)
    capability_metrics = measure_capability(trained_model)

    # 4. Validate
    degradation = baseline_capability - capability_metrics
    if degradation > DEGRADATION_THRESHOLD:
        logger.warning(f"Capability degradation {degradation}pp exceeds threshold!")
        # Still log results, but flag as concerning

    return {
        'safety': safety_metrics,
        'capability': capability_metrics,
        'degradation': degradation,
    }
```

---

## Expected Outcomes: RDO vs. SAE Steering

### Best Case: RDO Cleanly Separates Safety from Capability

```
Safety metric: ASR (attack success rate)
  Baseline:           50%
  SAE steering:        5% (45pp improvement ✓)
  RDO trained:        10% (40pp improvement ✓)

Capability metric: MMLU accuracy
  Baseline:          70%
  SAE steering:      35% (35pp degradation ✗✗✗)
  RDO trained:       68% (2pp degradation ✓)

Conclusion: RDO >> SAE steering (safety gains without capability loss)
```

### Worst Case: RDO Also Suffers Capability Degradation

```
Capability metric: MMLU accuracy
  Baseline:        70%
  RDO trained:     50% (20pp degradation ✗)

Conclusion: Refusal IS deeply entangled with capabilities
  → Need to move to E4 (RL) or accept tradeoff
  → Validates SAE paper's claim about feature entanglement
```

### Middle Case: Tradeoff Space Exists

```
Capability metric: MMLU accuracy
  Baseline:        70%
  RDO (lambda_retain=0):      55% (15pp degradation)
  RDO (lambda_retain=0.5):    68% (2pp degradation)
  RDO (lambda_retain=1.0):    70% (0pp degradation, but lower safety)

Conclusion: Can tune tradeoff via λ_retain
  → Pareto frontier exists
  → Document and publish optimal point
```

---

## Documentation and Publication Strategy

### Write-up for Experiments

**Add section to each experiment report:**
```markdown
## Capability Preservation

To address recent findings on SAE feature steering's capability
degradation (2411.11296), we measured MMLU, GSM8K, and TruthfulQA
alongside safety metrics.

| Benchmark | Baseline | Trained Model | Degradation |
|-----------|----------|---------------|------------|
| MMLU      | 68.80%   | 67.20%        | 1.60pp     |
| GSM8K     | 82.50%   | 81.00%        | 1.50pp     |
| TruthfulQA| 65.00%   | 64.50%        | 0.50pp     |

RDO training successfully preserves capabilities while improving safety.
```

### Comparison to SAE Steering

**Planned table in your paper:**
```
Table X: RDO vs. SAE Steering on Safety-Capability Tradeoff

| Method             | Safety ↑ | MMLU Degradation ↓ | Robustness |
|--------------------|----------|-------------------|------------|
| Baseline           | 50% ASR  | 0pp               | Weak      |
| SAE steering       | 5% ASR   | 35pp ✗            | Strong    |
| RDO (λ_retain=0)   | 8% ASR   | 20pp              | Strong    |
| RDO (λ_retain=0.5) | 12% ASR  | 2pp               | Strong    |
| **RDO (λ_retain=1)** | **20% ASR** | **<1pp** | **Strong** |

This paper achieves the first successful safety-capability tradeoff.
```

---

## Risk Mitigation

### Risk 1: RDO Replicates SAE's Problem

**If:** You find large MMLU degradation from RDO training

**Action:**
- Blame: Refusal and capabilities are inherently entangled (not a method problem)
- Solution: Move to E4 RL to explicitly optimize tradeoff
- Publication: "Characterizing the Safety-Capability Frontier in LLMs"

### Risk 2: Multi-Objective Training Isn't Enough

**If:** Even with λ_retain=2.0, capabilities still degrade

**Action:**
- Hypothesis: Refusal direction contaminated with language ability
- Solution: Use discovery phase (E1) to identify cleaner direction
- Publication: "Separating Refusal from Language Ability via Geometric Discovery"

### Risk 3: Benchmark Inflation

**If:** MMLU scores suspiciously high after RDO

**Action:**
- Test on out-of-distribution capabilities (unfamiliar domains)
- Compare to independent evaluation (not your own code)
- Validate against SAE paper's methodology

---

## Summary Checklist

Before publishing results on RDO training:

- [ ] **E1:** Added MMLU, GSM8K, TruthfulQA measurements
- [ ] **E2:** Layer-specific capability analysis
- [ ] **E3:** Compared per-layer vs. global directions
- [ ] **E4:** RL explicitly optimizes safety-capability
- [ ] **E5:** Ablated lambda_retain to show it prevents degradation
- [ ] **E6:** Unsupervised methods tested for capability preservation
- [ ] **New:** Pareto frontier analysis across lambda values
- [ ] **Validation:** Tested on unrelated domains (pure math, facts)
- [ ] **Comparison:** Published table comparing RDO to SAE steering
- [ ] **Documentation:** Updated CLAUDE.md with capability preservation notes

---

**Document Version:** 1.0
**Last Updated:** 2026-01-19
**Priority:** CRITICAL - Do not skip capability measurements
