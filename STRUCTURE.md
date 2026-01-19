# Repository Structure

This document explains the organization of the refusal-cones repository.

## Directory Layout

```
refusal-cones/
├── README.md                    # Main project documentation
├── STRUCTURE.md                 # This file
│
├── docs/                        # All documentation
│   ├── setup/                   # Getting started guides
│   │   ├── GPU_QUICKSTART.md
│   │   ├── COLAB_QUICKSTART.md
│   │   └── NEXT_STEPS.md
│   ├── architecture/            # System design and architecture
│   │   ├── SUMMARY.md
│   │   ├── IMPLEMENTATION_SUMMARY.md
│   │   ├── PROJECTION_AS_LORA.md
│   │   └── WEIGHT_MODIFICATION_ANALYSIS.md
│   ├── discovery/               # Geometry discovery methods
│   │   ├── GRADIENT_BASED_DISCOVERY.md
│   │   ├── ADAPTIVE_GEOMETRY_DISCOVERY.md
│   │   ├── EFFICIENT_HYPERSPHERE_EXPLORATION.md
│   │   ├── EXPLORATION_METHODS_COMPARISON.md
│   │   ├── GEOMETRY_COMPARISON.md
│   │   └── SAMPLING_SPACE_EXPLAINED.md
│   ├── training/                # Training documentation
│   │   ├── RDO_WITH_PEFT.md
│   │   ├── SFT_WITH_PEFT.md
│   │   ├── PER_LAYER_TRAINING.md
│   │   └── ADVERSARIAL_RL_README.md
│   ├── integrations/            # Third-party integrations
│   │   ├── VLLM_INTEGRATION.md
│   │   ├── TRL_GRPO_INTEGRATION.md
│   │   └── RL_FRAMEWORK_COMPARISON.md
│   └── legacy/                  # Historical documentation
│       ├── PORTING_NOTES.md
│       ├── CONVERSION_TEST_RESULTS.md
│       └── COST_COMPARISON.md
│
├── src/                         # Core source code
│   ├── discovery/               # Geometry discovery implementations
│   │   ├── gradient_discovery.py           # Gradient-based (RECOMMENDED)
│   │   ├── efficient_discovery.py          # Pure GP with efficiency
│   │   ├── adaptive_geometry_discovery.py  # Base GP implementation
│   │   └── adaptive_to_training.py         # Discovery → training pipeline
│   ├── training/                # Training implementations
│   │   ├── rdo_peft_adapter.py            # PEFT adapters for RDO
│   │   ├── rdo_peft_trainer.py            # Multi-objective RDO trainer
│   │   ├── projection_adapter.py          # Simple projection adapter
│   │   ├── per_layer_training.py          # Per-layer vector training
│   │   ├── sft_peft_trainer.py            # SFT training
│   │   ├── rl_vector_optimization.py      # RL optimization
│   │   ├── rl_adversarial.py              # Adversarial RL
│   │   └── rl_grpo_trl_adapted.py        # GRPO/TRL integration
│   ├── measurement/             # Measurement and evaluation
│   │   ├── vllm_hybrid_measurement.py     # vLLM + HF hybrid
│   │   ├── scoring.py                      # Response scoring
│   │   └── example_measure_function.py     # Example implementations
│   └── utils/                   # Shared utilities
│       ├── model_utils.py                  # Model loading/handling
│       ├── generate_utils.py               # Generation utilities
│       └── conversion_utils.py             # nnsight → PyTorch conversion
│
├── examples/                    # Example usage scripts
│   ├── example_full_pipeline.py
│   ├── example_adversarial_training.py
│   ├── example_per_layer_training.py
│   ├── example_projection_adapter_training.py
│   ├── example_trl_comparison.py
│   └── example_weight_vs_hooks_training.py
│
├── tests/                       # Test suite
│   ├── test_port.py
│   ├── test_smooth_max.py
│   ├── test_subspaces.py
│   └── test_conversion_equivalence.py
│
├── scripts/                     # Utility scripts
│   └── visualize_geometry.py
│
└── legacy/                      # Legacy code (reference only)
    ├── directopt.py
    ├── old_directopt.py
    ├── crossovereffects.py
    ├── plots.py
    ├── sampling.py
    ├── properties.py
    ├── targets.py
    ├── surrogate_scores.py
    ├── repind_gcg.py
    ├── repind_gcg_run.py
    └── new_repind_gcg.py
```

## Quick Navigation

### Getting Started
- **First time?** → [docs/setup/GPU_QUICKSTART.md](docs/setup/GPU_QUICKSTART.md)
- **Using Colab?** → [docs/setup/COLAB_QUICKSTART.md](docs/setup/COLAB_QUICKSTART.md)
- **What to do next?** → [docs/setup/NEXT_STEPS.md](docs/setup/NEXT_STEPS.md)

### Understanding the Approach
- **High-level overview** → [docs/architecture/SUMMARY.md](docs/architecture/SUMMARY.md)
- **Why gradients?** → [docs/discovery/GRADIENT_BASED_DISCOVERY.md](docs/discovery/GRADIENT_BASED_DISCOVERY.md)
- **Cones vs adaptive?** → [docs/discovery/GEOMETRY_COMPARISON.md](docs/discovery/GEOMETRY_COMPARISON.md)
- **All approaches compared** → [docs/discovery/EXPLORATION_METHODS_COMPARISON.md](docs/discovery/EXPLORATION_METHODS_COMPARISON.md)

### Implementation
- **Discovery** → `src/discovery/gradient_discovery.py` (start here)
- **Training** → `src/training/rdo_peft_trainer.py`
- **Measurement** → `src/measurement/vllm_hybrid_measurement.py`
- **Full example** → `examples/example_full_pipeline.py`

## Module Import Examples

### Discovery
```python
from src.discovery import (
    GradientGeometryDiscovery,
    GradientDiscoveryConfig,
    run_discovery_pipeline
)
```

### Training
```python
from src.training import (
    get_rdo_model,
    train_rdo_with_peft,
    RDOConfig
)
```

### Measurement
```python
from src.measurement import (
    HybridMeasurement,
    HybridMeasurementConfig
)
```

### Utils
```python
from src.utils import (
    load_model,
    apply_projection_hook,
    generate_with_projection
)
```

## Workflow Paths

### Path 1: Standard Discovery + Training
```
1. docs/setup/GPU_QUICKSTART.md           # Setup
2. src/discovery/gradient_discovery.py    # Run discovery
3. src/training/rdo_peft_trainer.py      # Train with results
4. examples/example_full_pipeline.py      # See it all together
```

### Path 2: Quick Colab Demo
```
1. docs/setup/COLAB_QUICKSTART.md        # Colab setup
2. Copy-paste notebook cells              # Run discovery only
3. Visualize results                      # Analyze geometry
```

### Path 3: Understand the Theory
```
1. docs/architecture/SUMMARY.md                    # Overview
2. docs/discovery/EXPLORATION_METHODS_COMPARISON.md # Methods
3. docs/discovery/GRADIENT_BASED_DISCOVERY.md      # Why gradients
4. docs/discovery/IMPLEMENTATION_DETAILS.md        # Deep dive
```

## File Naming Conventions

- **UPPERCASE.md**: Documentation files
- **lowercase_with_underscores.py**: Python source files
- **example_*.py**: Runnable example scripts
- **test_*.py**: Unit tests

## Dependencies Between Modules

```
src/utils/
    ↓
src/measurement/ ← src/discovery/
    ↓                   ↓
src/training/ ←────────┘
```

- **utils**: No dependencies (foundational)
- **measurement**: Depends on utils
- **discovery**: Depends on utils + measurement
- **training**: Depends on utils + measurement (optionally discovery for initialization)

## Deprecation Status

### Active (use these)
- `src/discovery/gradient_discovery.py` ✓
- `src/training/rdo_peft_adapter.py` ✓
- `src/measurement/vllm_hybrid_measurement.py` ✓

### Legacy (reference only)
- `legacy/directopt.py` - Original nnsight implementation
- `legacy/old_directopt.py` - Even older version
- All other files in `legacy/` - Historical code

## Development

### Adding a New Feature

1. **Core functionality** → Add to appropriate `src/` subdirectory
2. **Documentation** → Add to appropriate `docs/` subdirectory
3. **Example** → Add to `examples/`
4. **Tests** → Add to `tests/`
5. **Update** → Update this STRUCTURE.md and README.md

### Running Tests
```bash
# Run all tests
python -m pytest tests/

# Run specific test
python -m pytest tests/test_port.py
```

### Running Examples
```bash
# From repository root
python examples/example_full_pipeline.py

# Or with module imports
python -m examples.example_full_pipeline
```

## Notes

- All imports should be from the new structure (e.g., `from src.discovery import ...`)
- The `legacy/` directory is for reference only - do not import from it
- Check `docs/setup/GPU_QUICKSTART.md` for environment setup with `uv`
