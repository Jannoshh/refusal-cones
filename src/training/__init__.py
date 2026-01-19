"""Training modules for refusal vectors.

Organized into:
- adapters/: Adapter implementations (unified_rdo_adapter.py)
- trainers/: Trainer implementations (UnifiedRDOTrainer, RL trainers)

Implements ACE (Affine Concept Editing) from "Refusal in LLMs is an Affine Function".
"""

# Re-export from adapters
from .adapters import (
    ProjectionConfig,
    UnifiedRDOConfig,
    UnifiedRDOLayer,
    RankKUnifiedLayer,
    UnifiedRDOModel,
    get_unified_rdo_model,
)

# Re-export from trainers
from .trainers import (
    # Unified RDO trainer (ACE-based)
    UnifiedRDOTrainer,
    prepare_unified_dataset,
    train_unified_rdo,
    # Per-layer training
    PerLayerRefusalVectors,
    train_per_layer_vectors,
    smooth_max_loss,
    weighted_smooth_max_loss,
    compute_ce_loss,
    projection_einops,
    apply_per_layer_ablation,
    # RL trainers
    VectorPolicyGradient,
    PPOVectorOptimizer,
    GRPOAdversarialTrainer,
    HarmfulnessRewardModel,
    VectorGRPOTrainer,
)

__all__ = [
    # Adapters
    'ProjectionConfig',
    'UnifiedRDOConfig',
    'UnifiedRDOLayer',
    'RankKUnifiedLayer',
    'UnifiedRDOModel',
    'get_unified_rdo_model',
    # Trainers
    'UnifiedRDOTrainer',
    'prepare_unified_dataset',
    'train_unified_rdo',
    'PerLayerRefusalVectors',
    'train_per_layer_vectors',
    'smooth_max_loss',
    'weighted_smooth_max_loss',
    'compute_ce_loss',
    'projection_einops',
    'apply_per_layer_ablation',
    'VectorPolicyGradient',
    'PPOVectorOptimizer',
    'GRPOAdversarialTrainer',
    'HarmfulnessRewardModel',
    'VectorGRPOTrainer',
]
