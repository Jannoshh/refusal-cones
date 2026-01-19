"""Utility functions for model handling and generation."""

from .model_utils import load_model, apply_projection_hook
from .generate_utils import generate_with_projection
from .conversion_utils import convert_nnsight_to_pytorch
from .data_utils import (
    load_cb_harmful_train,
    load_harmless_data,
    prepare_rdo_datasets,
    format_prompt_for_model
)

__all__ = [
    'load_model',
    'apply_projection_hook',
    'generate_with_projection',
    'convert_nnsight_to_pytorch',
    'load_cb_harmful_train',
    'load_harmless_data',
    'prepare_rdo_datasets',
    'format_prompt_for_model'
]
