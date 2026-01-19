"""Trainer implementations for refusal vector training."""

from .unified_rdo_trainer import UnifiedRDOTrainer, prepare_unified_dataset, train_unified_rdo
from .per_layer_training import (
    PerLayerRefusalVectors,
    train_per_layer_vectors,
    smooth_max_loss,
    weighted_smooth_max_loss,
    compute_ce_loss,
    projection_einops,
    apply_per_layer_ablation
)

# RL-based trainers (experimental)
from .rl_vector_optimization import VectorPolicyGradient, PPOVectorOptimizer
from .rl_adversarial import GRPOAdversarialTrainer, HarmfulnessRewardModel
from .rl_grpo_trl_adapted import VectorGRPOTrainer

__all__ = [
    # Unified RDO trainer (ACE-based)
    'UnifiedRDOTrainer',
    'prepare_unified_dataset',
    'train_unified_rdo',
    # Per-layer training
    'PerLayerRefusalVectors',
    'train_per_layer_vectors',
    'smooth_max_loss',
    'weighted_smooth_max_loss',
    'compute_ce_loss',
    'projection_einops',
    'apply_per_layer_ablation',
    # RL trainers
    'VectorPolicyGradient',
    'PPOVectorOptimizer',
    'GRPOAdversarialTrainer',
    'HarmfulnessRewardModel',
    'VectorGRPOTrainer',
]
