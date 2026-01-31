"""
Training modules for ACE (Affine Concept Editing).

Implements ACE from "Refusal in LLMs is an Affine Function" (Marshall et al., 2024).

The ACE formula (Equation 5):
    h' = h - proj_r(h) + proj_r(r⁻) + α·r

Where:
    r   = steering direction (trainable)
    r⁻  = baseline (mean harmless activations)
    α   = steering parameter (0 = no refusal, 1 = full refusal)

Quick start:
    from src.training import train_ace

    model, trainer = train_ace(
        model_name="Qwen/Qwen3-0.6B",
        harmful_data=harmful,
        harmless_data=harmless,
        output_dir="ace_model",
        mode='sft'  # or 'rl'
    )

Layer selection:
    # Single layer (paper recommends)
    config = UnifiedRDOConfig(layers=[15])

    # All layers
    config = UnifiedRDOConfig(layers=None)

    # Select layers
    config = UnifiedRDOConfig(layers=[10, 11, 12, 13, 14, 15])
"""

# ACE Adapters
from .unified_rdo_adapter import (
    UnifiedRDOConfig,
    UnifiedRDOLayer,
    RankKUnifiedLayer,
    UnifiedRDOModel,
    get_unified_rdo_model,
)

# ACE Trainer
from .ace_trainer import (
    ACETrainer,
    ACETrainingConfig,
    train_ace,
)

__all__ = [
    # ACE Adapters
    'UnifiedRDOConfig',
    'UnifiedRDOLayer',
    'RankKUnifiedLayer',
    'UnifiedRDOModel',
    'get_unified_rdo_model',
    # ACE Trainer
    'ACETrainer',
    'ACETrainingConfig',
    'train_ace',
]
