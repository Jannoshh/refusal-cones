"""Utility functions for model handling and generation."""

from .model_utils import HookedModel, get_layer_activations, apply_intervention_to_layers
from .generate_utils import (
    projection_einops,
    generate_completions,
    intervene_with_fn_vector_ablation,
    intervene_with_fn_vector_addition
)
from .conversion_utils import (
    get_projection_matrix,
    VectorModifiedLayer,
    convert_model_to_vector_modified,
    get_trainable_vector_parameters
)
from .data_utils import (
    load_cb_harmful_train,
    load_harmless_data,
    prepare_rdo_datasets,
    format_prompt_for_model
)
from .mean_difference import compute_mean_difference_vector

__all__ = [
    'HookedModel',
    'get_layer_activations',
    'apply_intervention_to_layers',
    'projection_einops',
    'generate_completions',
    'intervene_with_fn_vector_ablation',
    'intervene_with_fn_vector_addition',
    'get_projection_matrix',
    'VectorModifiedLayer',
    'convert_model_to_vector_modified',
    'get_trainable_vector_parameters',
    'load_cb_harmful_train',
    'load_harmless_data',
    'prepare_rdo_datasets',
    'format_prompt_for_model',
    'compute_mean_difference_vector'
]
