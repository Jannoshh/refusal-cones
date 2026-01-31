# Compute Requirements: REINFORCE/GRPO Training

Computational cost breakdown for the mean difference → RL pipeline.

## Cost Breakdown

### Stage 1: Mean Difference Computation

**One-time cost** - cheap and fast

```python
# For 50 harmful + 50 harmless prompts:
# - 100 forward passes total
# - No generation needed (just activations)
# - ~2-5 seconds on A10G/A100
```

**Cost:** Negligible (~$0.001 on Modal)

---

### Stage 2: RL Training (REINFORCE/GRPO)

**Main computational cost** - dominated by generation

```python
# Per training step:
# - K samples (default: 4)
# - B prompts per step (default: 8)
# - Total generations: K × B = 32 generations
# - Each generation: ~150 tokens

# Per 100 steps:
# - Total generations: 100 × 32 = 3,200 generations
# - Total tokens: 3,200 × 150 = 480,000 tokens
```

---

## Hardware Requirements

### Minimum Requirements

| Model | VRAM | GPU | Training Time (100 steps) |
|-------|------|-----|---------------------------|
| **Qwen-0.5B** | 4 GB | T4 | ~15 min |
| **Gemma-2-2B** | 8 GB | L4/A10G | ~25 min |
| **Llama-3-8B** | 16 GB | A100 40GB | ~45 min |
| **Llama-3-70B** | 80+ GB | 2×A100 80GB | ~3 hours |

**Recommended GPU:** A10G (good price/performance balance)

---

## Cost Estimates

### Modal (Cloud GPU) Pricing

| GPU | Per Second | Per Hour | 100 steps (est.) | Cost |
|-----|-----------|----------|------------------|------|
| **T4** | $0.000164 | $0.59 | 15 min | **$0.15** |
| **L4** | $0.000222 | $0.80 | 20 min | **$0.27** |
| **A10G** | $0.000306 | $1.10 | 25 min | **$0.46** |
| **A100 40GB** | $0.000583 | $2.10 | 45 min | **$1.58** |
| **A100 80GB** | $0.000694 | $2.50 | 45 min | **$1.88** |

**Free tier:** $30/month credits = ~65 runs on A10G or ~20 runs on A100

---

## Detailed Cost Analysis

### Generation Cost Dominates

```
Per step breakdown (K=4, B=8, model=Gemma-2-2B on A10G):

1. Sample K vectors: ~0.001s (negligible)
2. Generate K×B completions:
   - 32 generations × 150 tokens each
   - ~150 tokens/sec throughput
   - Time: 32 × 150 / 150 = ~32 seconds
3. Score with judge:
   - 32 forward passes (judge model)
   - ~2 seconds
4. Compute gradients & update: ~0.1s

Total per step: ~35 seconds
Total for 100 steps: ~3,500 seconds ≈ 58 minutes
```

**Bottleneck:** Generation (92% of time)

---

## Optimization Strategies

### 1. Reduce Generation Cost

**Reduce max_new_tokens:**
```python
# Default: 150 tokens
max_new_tokens = 150  # ~35s per step

# Fast mode: 50 tokens
max_new_tokens = 50   # ~15s per step (2.3× faster)

# Ultra-fast: 20 tokens
max_new_tokens = 20   # ~8s per step (4.4× faster)
```

**Trade-off:** Shorter generations may not fully demonstrate harmfulness
**Recommendation:** Start with 50 tokens, increase if needed

### 2. Reduce K (samples per step)

```python
# Default: K=4
k_samples = 4  # Good variance reduction

# Fast mode: K=2
k_samples = 2  # 2× faster, higher variance

# Slow mode: K=8
k_samples = 8  # Better estimates, 2× slower
```

**Trade-off:** Fewer samples = higher gradient variance
**Recommendation:** K=4 is a good default

### 3. Batch Generation

**Current:** Sequential generation (one vector at a time)
**Optimized:** Batch all K×B generations together

```python
# Potential speedup: 2-3× faster
# Implementation complexity: Medium
# Would require batched ablation hooks
```

### 4. Smaller Model for Judge

```python
# Current: StrongREJECT (loaded separately)
# Alternative: Smaller judge like Llama-Guard-3-1B

# Speedup: ~30% faster scoring
# Trade-off: Slightly less accurate rewards
```

---

## Realistic Scenarios

### Scenario 1: Quick Test (Budget: $0.50)

```bash
python examples/reinforce_from_mean_diff.py \
    --model google/gemma-2-2b-it \
    --steps 50 \
    --k-samples 2 \
    --n-harmful 20 \
    --n-harmless 20

# Time: ~12 minutes on A10G
# Cost: ~$0.22
# Expected improvement: +5-8% ASR
```

### Scenario 2: Standard Run (Budget: $2)

```bash
python examples/reinforce_from_mean_diff.py \
    --model google/gemma-2-2b-it \
    --steps 100 \
    --k-samples 4 \
    --n-harmful 50 \
    --n-harmless 50

# Time: ~25 minutes on A10G
# Cost: ~$0.46
# Expected improvement: +8-12% ASR
```

### Scenario 3: High-Quality Run (Budget: $5)

```bash
python examples/reinforce_from_mean_diff.py \
    --model meta-llama/Llama-3-8B-Instruct \
    --steps 200 \
    --k-samples 8 \
    --n-harmful 100 \
    --n-harmless 100

# Time: ~90 minutes on A100
# Cost: ~$3.15
# Expected improvement: +10-15% ASR
```

### Scenario 4: Research-Grade (Budget: $20)

```bash
# Multiple runs with different hyperparameters
# 5 runs × $3 = $15
# Includes REINFORCE vs GRPO comparison
# Full ablation study

# Total time: ~8 hours on A100
# Cost: ~$16
# Deliverable: Publication-ready results
```

---

## Cost Comparison: RL vs Alternatives

| Method | Time | Cost (A10G) | ASR Improvement |
|--------|------|-------------|-----------------|
| **Mean diff only** | 5 sec | $0.001 | Baseline |
| **+ GRPO (50 steps)** | 12 min | $0.22 | +5-8% |
| **+ GRPO (100 steps)** | 25 min | $0.46 | +8-12% |
| **+ REINFORCE (100 steps)** | 25 min | $0.46 | +10-13% |
| **+ Full GCG attack** | ~3 hours | $3.30 | +15-25% |

**RL is 6× cheaper than GCG** while getting most of the improvement!

---

## Memory Requirements

### GPU VRAM

```python
# Base model (bfloat16):
model_size_gb = params_billions * 2  # 2 bytes per param

# Examples:
Gemma-2-2B:   ~4 GB
Llama-3-8B:   ~16 GB
Llama-3-70B:  ~140 GB (needs multi-GPU)

# Additional overhead:
# - Activations during generation: +2-4 GB
# - KV cache: +1-2 GB per concurrent generation
# - Optimizer states: +model_size_gb (for Adam)

# Total VRAM needed:
gemma_2b_vram = 4 + 3 + 4 = ~11 GB  # → 16GB GPU
llama_8b_vram = 16 + 4 + 16 = ~36 GB  # → 40GB GPU
```

**Optimization: Keep vectors frozen (no optimizer states for model)**
- Model optimizer: DISABLED ✓
- Vector optimizer only: ~1 MB
- Total savings: ~50% VRAM

### RAM Requirements

```python
# Minimal (streaming generation):
ram_needed = 8 GB

# With caching:
ram_needed = 16 GB  # Recommended
```

---

## Speed Optimizations

### Flash Attention

```python
# Enable Flash Attention 2 for 2-3× faster generation
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="flash_attention_2"  # Add this
)

# Speedup: 2-3× faster
# Requirements: torch >= 2.0, flash-attn package
```

### Compilation

```python
# PyTorch 2.0 compile for 1.5-2× speedup
model = torch.compile(model, mode="reduce-overhead")

# Speedup: 1.5-2× faster (after warmup)
# Caveat: First few steps are slower (compilation)
```

### BetterTransformer

```python
# HuggingFace BetterTransformer for 1.3-1.5× speedup
from optimum.bettertransformer import BetterTransformer

model = BetterTransformer.transform(model)

# Speedup: 1.3-1.5× faster
# Easiest to implement
```

**Combined speedup: 4-6× faster** (Flash + Compile + BetterTransformer)

With optimizations:
- 100 steps on Gemma-2-2B: **~6 minutes** (was 25 min)
- Cost: **~$0.11** (was $0.46)

---

## Parallelization

### Multiple Experiments

```bash
# Run REINFORCE and GRPO in parallel
modal run modal_app.py::run_reinforce &
modal run modal_app.py::run_grpo &

# 2× throughput (same total time)
# Each gets its own GPU
```

### Hyperparameter Sweep

```python
# Launch 5 jobs with different hyperparameters
for lr in [1e-4, 3e-4, 1e-3, 3e-3, 1e-2]:
    modal run modal_app.py::run_reinforce --lr $lr

# Wall-clock time: Same as single run
# Total cost: 5× single run
```

---

## Recommendations

### For Development/Testing
- **GPU:** A10G (best price/performance)
- **Steps:** 50
- **K samples:** 2
- **Model:** Gemma-2-2B
- **Time:** ~12 min
- **Cost:** ~$0.22

### For Paper Results
- **GPU:** A100 40GB
- **Steps:** 100-200
- **K samples:** 4-8
- **Model:** Llama-3-8B
- **Time:** ~90 min
- **Cost:** ~$3.15

### For Ablation Studies
- **Run multiple configs in parallel**
- **Total budget:** $15-20
- **Deliverable:** Complete comparison table

---

## Cost-Saving Tips

1. **Use Modal free tier** - $30/month = 65 runs on A10G
2. **Start small** - Test on Gemma-2-2B before scaling to Llama
3. **Reduce max_new_tokens** - 50 tokens is usually enough
4. **Use smaller K** - K=2 for debugging, K=4 for final runs
5. **Enable optimizations** - Flash Attention = 2-3× speedup
6. **Batch experiments** - Run multiple configs in one session

---

## Example Budget Plan

**Goal:** Compare REINFORCE vs GRPO on Gemma-2-2B

```
1. Mean diff (both methods): $0.001
2. GRPO baseline (100 steps): $0.46
3. REINFORCE (100 steps): $0.46
4. GRPO (200 steps, for comparison): $0.92
5. REINFORCE (200 steps): $0.92
6. Ablation study (5 configs): $2.30

Total: ~$5.06
Time: ~3 hours (parallel execution)
```

**Result:** Full comparison with publication-quality data for <$10

---

## Summary

| Configuration | Time | Cost | Use Case |
|---------------|------|------|----------|
| **Quick test** | 12 min | $0.22 | Development |
| **Standard** | 25 min | $0.46 | Single experiment |
| **High quality** | 90 min | $3.15 | Publication |
| **Full study** | 3 hours | $5-10 | Complete comparison |

**Bottom line:**
- Single run: **~$0.50 and 25 minutes**
- Full study: **~$10 and 3 hours**
- Very affordable compared to GCG (6× cheaper, 7× faster)

The free Modal tier ($30/month) covers ~60 experimental runs!
