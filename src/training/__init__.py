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

__all__ = [
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
    'train_sft_with_peft'
]
