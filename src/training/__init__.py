"""Training modules for refusal vectors."""

from .rdo_peft_adapter import RDOConfig, get_rdo_model
from .rdo_peft_trainer import train_rdo_with_peft, RDOTrainer
from .projection_adapter import ProjectionConfig, ProjectionLayer, get_projection_model
from .per_layer_training import (
    PerLayerRefusalVectors,
    train_per_layer_vectors,
    smooth_max_loss,
    weighted_smooth_max_loss
)
from .sft_peft_trainer import train_sft_with_peft

# Unified Affine RDO modules
from .unified_rdo_adapter import (
    UnifiedRDOConfig,
    UnifiedRDOLayer,
    RankKUnifiedLayer,
    UnifiedRDOModel,
    get_unified_rdo_model
)
from .unified_rdo_trainer import (
    UnifiedRDOTrainer,
    prepare_unified_dataset,
    train_unified_rdo
)
from .unified_rdo_eval import (
    UnifiedRDOEvaluator,
    EvaluationResults,
    evaluate_unified_rdo
)
from .refusal_token_eval import (
    RefusalTokenEvaluator,
    RefusalTokenResults,
    evaluate_with_refusal_tokens,
    get_refusal_tokens
)

__all__ = [
    # Standard RDO
    'RDOConfig',
    'get_rdo_model',
    'train_rdo_with_peft',
    'RDOTrainer',
    'ProjectionConfig',
    'ProjectionLayer',
    'get_projection_model',
    'PerLayerRefusalVectors',
    'train_per_layer_vectors',
    'smooth_max_loss',
    'weighted_smooth_max_loss',
    'train_sft_with_peft',
    # Unified Affine RDO
    'UnifiedRDOConfig',
    'UnifiedRDOLayer',
    'RankKUnifiedLayer',
    'UnifiedRDOModel',
    'get_unified_rdo_model',
    'UnifiedRDOTrainer',
    'prepare_unified_dataset',
    'train_unified_rdo',
    'UnifiedRDOEvaluator',
    'EvaluationResults',
    'evaluate_unified_rdo',
    'RefusalTokenEvaluator',
    'RefusalTokenResults',
    'evaluate_with_refusal_tokens',
    'get_refusal_tokens'
]
