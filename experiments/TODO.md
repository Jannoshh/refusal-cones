# Experiment TODO List

## Paper: Beyond Concept Cones

**Status:** Planning → Experiments → Writing → Submission

---

## Quick Start

```bash
# Install dependencies
cd experiments
pip install -r requirements.txt

# List all experiments
python run_all.py --list

# Run specific experiment
python run_all.py -e e1.1 --model gemma-2-2b

# Run all experiments for a model
python run_all.py --all --model gemma-2-9b
```

---

## Experiment Checklist

### E1: Discovery Experiments (~15 GPU hours)

#### E1.1: Discovery Efficiency ⬜
**Goal:** Show gradient+GP is more efficient than pure GP

**Run:**
```bash
python e1_discovery/run_efficiency.py --model gemma-2-2b
python e1_discovery/run_efficiency.py --model qwen-2.5-7b
python e1_discovery/run_efficiency.py --model llama-3-8b
```

**Acceptance Criteria:**
- [ ] Gradient+GP reaches ASR > 0.7 in ≤100 measurements
- [ ] Pure GP requires ≥300 measurements for same ASR
- [ ] Speedup ≥ 3× documented
- [ ] Results for 3 model families

**Expected Output:**
| Method | Measurements | Best ASR |
|--------|-------------|----------|
| Gradient+GP | 50-70 | >0.7 |
| Pure GP | 300-500 | >0.7 |
| Random | 500+ | <0.6 |

---

#### E1.2: Discovery Quality ⬜
**Goal:** Show adaptive discovery finds better geometry than DIM

**Run:**
```bash
python e1_discovery/run_quality.py --model gemma-2-2b
python e1_discovery/run_quality.py --model gemma-2-9b
```

**Acceptance Criteria:**
- [ ] Adaptive discovery ASR > DIM ASR by ≥5%
- [ ] TruthfulQA side effects within 3% of DIM
- [ ] Results visualized in comparison plot

---

### E2: Per-Layer Experiments (~70 GPU hours)

#### E2.1: Dimensionality by Layer ⬜
**Goal:** Test inverted-U hypothesis for refusal dimensionality

**Run:**
```bash
python e2_per_layer/run_dimensionality.py --model gemma-2-9b --n_samples 100
python e2_per_layer/run_dimensionality.py --model qwen-2.5-14b --n_samples 100
python e2_per_layer/run_dimensionality.py --model llama-3-8b --n_samples 100
```

**Acceptance Criteria:**
- [ ] Plot intrinsic dimension vs layer for each model
- [ ] Test if middle_dim > early_dim AND middle_dim > late_dim
- [ ] Report peak dimension layer (expected: 40-60% depth)
- [ ] Statistical significance test

**Expected Result:**
```
Layer:    1    5   10   15   20   25   30
Dim:      2    3    5    7    6    4    2
```

---

#### E2.2: Per-Layer ASR ⬜
**Goal:** Identify which layers matter most for refusal

**Run:**
```bash
python e2_per_layer/run_layer_asr.py --model gemma-2-9b --n_steps 200
```

**Acceptance Criteria:**
- [ ] ASR plot by layer (bar chart)
- [ ] Identify top 3 most important layers
- [ ] Compare single-layer vs all-layer ablation
- [ ] Correlation between dimensionality and ASR per layer

---

### E3: Unified Affine Experiments (~20 GPU hours)

#### E3.1: Unified vs Separate ⬜
**Goal:** Show unified affine outperforms separate operations

**Run:**
```bash
python e3_unified_affine/run_comparison.py --model gemma-2-2b --num_epochs 10
python e3_unified_affine/run_comparison.py --model qwen-2.5-7b --num_epochs 10
```

**Acceptance Criteria:**
- [ ] Unified ASR ≥ Separate ASR
- [ ] Training time comparison (unified should be faster)
- [ ] Loss curve comparison (unified should be more stable)
- [ ] Side effects comparison

**Expected Result:**
| Method | ASR | Training Time | TruthfulQA |
|--------|-----|--------------|------------|
| Separate | 0.65 | 2x | 52% |
| Unified | 0.70 | 1x | 54% |

---

### E4: RL Optimization Experiments (~70 GPU hours)

#### E4.1: SFT → RL Pipeline ⬜
**Goal:** Measure RL gains over SFT baseline

**Run:**
```bash
python e4_rl_optimization/run_sft_rl_pipeline.py --model gemma-2-2b --sft_steps 300 --rl_steps 200
python e4_rl_optimization/run_sft_rl_pipeline.py --model gemma-2-9b --sft_steps 300 --rl_steps 200
```

**Acceptance Criteria:**
- [ ] Report ASR at each stage: Baseline → SFT → RL
- [ ] Compute RL gain = (RL ASR - SFT ASR)
- [ ] RL training curve (should plateau)
- [ ] Document if gain is marginal (<5%)

**Expected Result:**
| Stage | ASR | Δ |
|-------|-----|---|
| Baseline | 0.05 | - |
| SFT | 0.65 | +0.60 |
| RL | 0.68 | +0.03 |

---

#### E4.3: Ceiling Analysis ⬜
**Goal:** Distinguish between ceiling hypotheses

**Run:**
```bash
python e4_rl_optimization/run_ceiling_analysis.py --models qwen-2.5-7b qwen-2.5-14b
```

**Acceptance Criteria:**
- [ ] Compare ASR on 7B vs 14B with same steering
- [ ] Failure case breakdown (refuses / vague / low-score)
- [ ] Hypothesis determination:
  - If 14B >> 7B: Capability limit (B)
  - If 14B ≈ 7B: Steering limit (A) or Judge limit (C)
- [ ] Dominant failure mode identified

---

### E5: Ablation Studies (~20 GPU hours)

#### E5.1: Component Ablation ⬜
**Goal:** Measure contribution of each component

**Run:**
```bash
python e5_ablations/run_component_ablation.py --model gemma-2-9b
```

**Acceptance Criteria:**
- [ ] Full method ASR
- [ ] ASR without each component
- [ ] Rank components by importance
- [ ] Ablation plot

---

## Results Aggregation

After running all experiments, aggregate results:

```bash
# Collect all results
python scripts/aggregate_results.py --results_dir results/

# Generate paper figures
python scripts/generate_figures.py --results_dir results/
```

---

## Acceptance Criteria Summary

### For Paper Submission:

**Must have:**
- [ ] E1: Discovery is ≥3× more efficient
- [ ] E2: Per-layer analysis complete for ≥2 models
- [ ] E3: Unified affine comparison
- [ ] E4: RL pipeline with ceiling analysis
- [ ] E5: Component ablation

**Key claims to verify:**
- [ ] Claim 1: "Flexible geometry > cones" → E1.2 shows +5% ASR
- [ ] Claim 2: "Dimensionality varies by layer" → E2.1 shows inverted-U
- [ ] Claim 3: "Unified > separate" → E3.1 shows improvement
- [ ] Claim 4: "RL hits ceiling" → E4.1 shows <5% gain

**Figures needed:**
- [ ] Fig 1: Discovery efficiency comparison
- [ ] Fig 2: Dimensionality by layer (3 models)
- [ ] Fig 3: Per-layer ASR heatmap
- [ ] Fig 4: SFT → RL pipeline
- [ ] Fig 5: Ceiling analysis (model size comparison)
- [ ] Fig 6: Component ablation

---

## Compute Budget

| Experiment | GPU Hours (A100) |
|------------|------------------|
| E1 Discovery | 15 |
| E2 Per-layer | 70 |
| E3 Unified | 20 |
| E4 RL | 70 |
| E5 Ablations | 20 |
| **Total** | **~200** |

*Note: Estimates assume single runs. Add 50% for reruns/debugging.*

---

## Timeline

| Week | Focus | Experiments |
|------|-------|-------------|
| 1 | Setup + E1 | E1.1, E1.2 |
| 2-3 | Per-layer | E2.1, E2.2 |
| 4 | Unified | E3.1 |
| 5-6 | RL | E4.1, E4.3 |
| 7 | Ablations | E5.1 |
| 8-9 | Writing | - |
| 10 | Revision | - |

---

## Troubleshooting

### OOM Errors
```bash
# Use smaller batch size
python run_X.py --batch_size 2

# Use gradient checkpointing
# (enabled by default in most scripts)

# Use quantization
python run_X.py --load_in_8bit
```

### Slow Training
```bash
# Use smaller model for debugging
python run_X.py --model gemma-2-2b

# Reduce samples
python run_X.py --n_samples 20
```

### Judge Errors
```bash
# Fall back to keyword-based scoring
# (automatic if StrongREJECT fails to load)
```

---

## Notes

- All experiments save results to `results/{experiment}/{model}/{timestamp}/`
- Plots are saved as PNG files in the same directory
- Vectors are saved as `.pt` files for reuse
- Logs include full configuration for reproducibility
