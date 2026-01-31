# Refusal Cones - Training Performant Refusal Vectors

**Adversarial research project for training steering vectors that ablate refusal in LLMs**

This repository implements:
1. **ACE (Affine Concept Editing)** - State-of-the-art steering with bias correction
2. **RDO Training** - Multi-objective (ablation + addition + retain) via SFT or REINFORCE
3. **HarmBench Evaluation** - Standard benchmark for measuring attack success rate

### Latest updates
- **Training**: All code in `src/training/`. Use `UnifiedRDOConfig(layers=[15])` for single-layer (recommended). ACE formula: `h' = h - proj_r(h) + proj_r(r⁻) + α·r`.
- **Single-layer beats multi-layer**: L15 achieves 37% refusal reduction vs 0.8% for multi-layer.
- **Modal**: `run_modal.py` wraps `src/modal/app/`. REINFORCE training in `src/modal/modal_ace_reinforce.py`.
- **Tests**: `uv run pytest -q` (fast CPU tests, no GPU required).

### Training Methods

We implement two training approaches for ACE adapters:

#### RDO (SFT-based)

Original supervised fine-tuning approach with multi-objective loss:

```python
# 2-pass loss for harmful examples
loss_ablate = CE(harmful_completion | α=0)  # Comply when ablated
loss_add = CE(refusal_completion | α=1)     # Refuse when added
loss_retain = CE(helpful_completion)         # Preserve capabilities

loss = λ_ablate * loss_ablate + λ_add * loss_add + λ_retain * loss_retain
```

**Usage:**
```python
from src.training import train_ace
model, trainer = train_ace(..., mode='sft')
```

#### REINFORCE-RDO (RL-based)

Based on ["REINFORCE Adversarial Attacks on Large Language Models"](https://arxiv.org/abs/2502.17254) (Geisler et al., 2025). Uses HarmBench classifier as reward signal:

```python
# Variance-reduced policy gradient
w_i = R_i - mean(R)  # Baseline subtraction
grad = Σ w_i · ∇log π(v_i)
```

**Key advantage:** Optimizes directly for ASR without needing harmful completions as targets.

**Usage:**
```bash
# REINFORCE-RDO training with HarmBench judge
modal run src/modal/modal_ace_reinforce.py

# Quick test (~5 min)
modal run src/modal/modal_ace_reinforce.py --quick

# Single layer ablation
modal run src/modal/modal_ace_reinforce.py --layer 15
```

**Token limits (per paper):** 128 tokens during training, 512 for evaluation.

**Expected results on Gemma-2-2B:**
- Baseline (mean diff): ~70% ASR
- After REINFORCE-RDO (100 steps): ~85% ASR (+15%)

### Evaluation Dataset (HarmBench)

We use the **exact same dataset** as the REINFORCE attacks paper for reproducible comparisons:

- **Source:** HarmBench test set (indices 0-199)
- **Total behaviors:** 200
- **Standard behaviors:** 95 (filtered: no copyright, no contextual)
- **Categories:**
  - illegal (37 behaviors)
  - misinformation_disinformation (34)
  - cybercrime_intrusion (32)
  - chemical_biological (21)
  - harassment_bullying (15)
  - harmful (10)

**Files:**
- `data/harmbench/harmbench_test_200.json` - All 200 behaviors
- `data/harmbench/harmbench_test_standard.json` - 95 standard behaviors

**Judge:** HarmBench-Llama-2-13b-cls (same as paper)

## Quick Start

### Prerequisites

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) - Fast Python package manager

### Installation

```bash
# Clone the repository
git clone <repo-url>
cd refusal-cones

# Install dependencies with uv (creates .venv automatically)
uv sync

# Run commands with uv run (no need to activate venv)
uv run pytest -q
```

### Basic Usage

#### 1. Train Refusal Vectors

Train ACE (Affine Concept Editing) adapters. Directions are initialized from mean difference (harmful - harmless activations):

**With full control:**

```python
from src.training import (
    UnifiedRDOConfig,
    get_unified_rdo_model,
    ACETrainer,
    ACETrainingConfig,
)
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-0.6B",
    torch_dtype=torch.float16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")

# Configure ACE adapter with layer selection
config = UnifiedRDOConfig(
    layers=[15],    # Single layer (or None for all, or list for select)
    alpha=0.0,      # Start with ablation mode
)
model = get_unified_rdo_model(base_model, config)

# Fit baselines from data
model.fit_all_baselines(tokenizer, harmless_prompts, harmful_prompts)

# Create trainer
train_config = ACETrainingConfig(
    mode='sft',
    learning_rate=1e-3,
    lambda_ablate=1.0,   # Compliance when ablated (α=0)
    lambda_add=1.0,      # Refusal when added (α>0)
    lambda_retain=0.5,   # Helpfulness on harmless
)
trainer = ACETrainer(model, tokenizer, train_config)

# Train
trainer.train_sft(harmful_data, harmless_data, n_epochs=10)

# Save
model.save_pretrained("final_adapters")
```

#### 2. Evaluate

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.training import UnifiedRDOModel

# Load base model first
base_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-0.6B",
    torch_dtype=torch.float16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")

# Load trained ACE adapters onto base model
model = UnifiedRDOModel.from_pretrained(base_model, "final_adapters")

# Set to ablation mode (α=0) and generate
model.set_alpha(0.0)
inputs = tokenizer(["How to build a bomb?"], return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## Modal (Cloud GPU)

Run GPU workloads on Modal for training and evaluation.

### Setup

```bash
# Install modal
uv pip install "modal>=0.64"

# Authenticate (one-time, opens browser)
uv run modal token new
```

### Usage

```bash
# REINFORCE training with HarmBench judge
modal run src/modal/modal_ace_reinforce.py

# Quick test (~5 min)
modal run src/modal/modal_ace_reinforce.py --quick

# Evaluate prompts
uv run modal run run_modal.py::run_evaluation --prompts '["How to hack?"]'
```

### Viewing Logs

All print statements are captured. View logs at:
- https://modal.com/apps (select 'refusal-cones' app)
- Or the URL printed when running: `View run at https://modal.com/apps/...`

### GPU Pricing (Per-Second Billing)

| GPU | Per Second | Per Hour | 5-min job |
|-----|-----------|----------|-----------|
| T4 | $0.000164 | ~$0.59 | $0.05 |
| L4 | $0.000222 | ~$0.80 | $0.07 |
| **A10G** | **$0.000306** | **~$1.10** | **$0.09** |
| L40S | $0.000542 | ~$1.95 | $0.16 |
| A100 40GB | $0.000583 | ~$2.10 | $0.17 |
| A100 80GB | $0.000694 | ~$2.50 | $0.21 |
| H100 | $0.001097 | ~$3.95 | $0.33 |

Plus CPU/memory: ~$0.05/hr for 1 core + 8GB.

**Recommendation:** Use **A10G** (default) — good speed/cost balance. T4 is cheaper but slower, often not worth the time savings.

### Free Tier

Starter plan (free): **$30/month credits** — enough for:
- ~50 ten-minute A10G jobs
- ~90 ten-minute A100 jobs


**Batch size recommendations:**
- 32 (default): Fast, reasonable estimates
- 64: Better estimates, ~2x slower per measurement
- 128: High quality, ~4x slower per measurement

## Repository Structure

### Quick Overview

```
refusal-cones/
├── src/                   # Core implementations
│   ├── training/         # Training modules (flat structure)
│   │   ├── unified_rdo_adapter.py  # ACE adapters with layer selection
│   │   └── ace_trainer.py          # Unified SFT + RL trainer
│   ├── steering/         # Inference-time steering
│   │   └── ace.py        # Clean ACE implementation
│   ├── measurement/      # Evaluation (vllm_hybrid_measurement.py, etc.)
│   ├── utils/            # Utilities
│   └── modal/            # Modal cloud GPU package
│       ├── app/          # Main modal app modules
│       └── modal_ace_reinforce.py  # REINFORCE training
├── run_modal.py           # Thin wrapper importing from src/modal/app/
├── docs/                  # All documentation
├── examples/              # Example scripts
├── tests/                 # Test suite
├── scripts/               # Utility scripts
└── legacy/                # Historical code (reference only)
```

### Core Modules

**Training** (in `src/training/`):
- `unified_rdo_adapter.py` - ACE adapters with layer selection (UnifiedRDOLayer, RankKUnifiedLayer, UnifiedRDOModel)
- `ace_trainer.py` - Unified trainer supporting both SFT and RL modes (ACETrainer, train_ace)

**Steering** (in `src/steering/`):
- `ace.py` - Clean ACE implementation for inference (ace_transform, ACESteerer)

**Documentation** (in `docs/`):
- `docs/setup/GPU_QUICKSTART.md` - Step-by-step setup guide
- `docs/setup/COLAB_QUICKSTART.md` - Colab-specific guide

## Key Concepts

### 1. Projection as Rank-1 LoRA

The projection operation `h' = (I - vv^T)h` is mathematically equivalent to rank-1 LoRA:

```python
# These are equivalent:
h_ablated = h - torch.einsum('...d,d->...', h, v) * v  # Projection
h_ablated = h + LoRA(h, rank=1)                       # LoRA with special init

# Benefit: Use PEFT library (optimized, well-tested, 97% less code)
```

### 2. Multi-Objective RDO with ACE

Train vectors with three objectives using 2 forward passes for harmful examples:

```python
# Harmful examples: 2 forward passes with different α values
if is_harmful:
    # Pass 1: Ablation (α=0) - want model to COMPLY
    # h' = h - proj_v(h) + proj_v(v⁻)
    loss_ablate = CE(harmful_completion | α=0)

    # Pass 2: Addition (α>0) - want model to REFUSE
    # h' = h - proj_v(h) + proj_v(v⁻) + α·v
    loss_add = CE(refusal_completion | α=1)

    loss = λ_ablate * loss_ablate + λ_add * loss_add

# Harmless examples: single pass, no transformation
else:
    loss_retain = CE(helpful_completion | unchanged)
    loss = λ_retain * loss_retain
```

**Key insight**: Harmful examples need BOTH `harmful_completion` (what model says when jailbroken) and `refusal_completion` (what model should say).

### 3. Affine Concept Editing (ACE)

Based on ["Refusal in LLMs is an Affine Function"](literature/papers/affine_refusal.pdf) (Marshall et al., 2024), we implement the exact ACE formula (Equation 5):

```
h' = h - proj_r(h) + proj_r(r⁻) + α·r
```

Where:
- `r` = steering direction (trainable, initialized from mean diff)
- `r⁻` = baseline (mean harmless activations)
- `r⁺` = mean harmful activations
- `α` = steering parameter (0 = ablate refusal, 1 = induce refusal)
- `proj_r(x) = (x·r̂)r̂` where `r̂ = r/||r||`

**Key insight**: The bias term `proj_r(r⁻)` keeps activations in a sensible region of activation space. Without it, simple directional ablation can push activations into nonsensical regions.

**Layer selection**: The affine refusal paper recommends single-layer ablation (typically middle layers work best). Configure via:

```python
from src.training import UnifiedRDOConfig, get_unified_rdo_model

# Single layer (paper recommendation)
config = UnifiedRDOConfig(layers=[15], alpha=0.0)

# Multiple select layers
config = UnifiedRDOConfig(layers=[10, 11, 12, 13, 14, 15], alpha=0.0)

# All layers (default)
config = UnifiedRDOConfig(layers=None, alpha=0.0)

model = get_unified_rdo_model(base_model, config)

# Fit baselines from data
model.fit_all_baselines(tokenizer, harmless_prompts, harmful_prompts)
```

**Training modes**: Use `ACETrainer` for both SFT and RL:

```python
from src.training import ACETrainer, ACETrainingConfig

# SFT mode (cross-entropy loss)
config = ACETrainingConfig(mode='sft', learning_rate=1e-3)
trainer = ACETrainer(model, tokenizer, config)
trainer.train_sft(harmful_data, harmless_data, n_epochs=10)

# RL mode (REINFORCE with reward function)
config = ACETrainingConfig(mode='rl', learning_rate=1e-4, exploration_std=0.1)
trainer = ACETrainer(model, tokenizer, config)
trainer.train_rl(prompts, reward_fn, n_steps=100)
```

**Adapter parameters:**

Each `UnifiedRDOLayer` contains:
- `r` - Trainable steering direction (initialized from mean diff)
- `r_minus` - Harmless baseline (computed via `fit_baseline()`)
- `r_plus` - Harmful baseline (optional, for analysis)
- `_alpha` - Steering strength (0 = ablate, 1 = induce)