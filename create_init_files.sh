#!/bin/bash

# Create __init__.py files for src packages
touch src/__init__.py

cat > src/discovery/__init__.py << 'PYEOF'
"""Geometry discovery methods for refusal vectors."""

from .gradient_discovery import (
    GradientGeometryDiscovery,
    GradientDiscoveryConfig,
    riemannian_gradient_ascent,
    sample_von_mises_fisher
)
from .efficient_discovery import EfficientDiscovery
from .adaptive_geometry_discovery import (
    AdaptiveGeometryDiscovery,
    SimpleGP,
    estimate_intrinsic_dimension
)
from .adaptive_to_training import run_discovery_pipeline

__all__ = [
    'GradientGeometryDiscovery',
    'GradientDiscoveryConfig',
    'riemannian_gradient_ascent',
    'sample_von_mises_fisher',
    'EfficientDiscovery',
    'AdaptiveGeometryDiscovery',
    'SimpleGP',
    'estimate_intrinsic_dimension',
    'run_discovery_pipeline'
]
PYEOF

cat > src/training/__init__.py << 'PYEOF'
"""Training modules for refusal vectors."""

from .rdo_peft_adapter import (
    RDOConfig,
    get_rdo_model,
    get_cone_model,
    set_operation_mode
)
from .rdo_peft_trainer import train_rdo_with_peft
from .projection_adapter import ProjectionAdapter
from .per_layer_training import train_per_layer_vectors
from .sft_peft_trainer import train_sft_with_peft

__all__ = [
    'RDOConfig',
    'get_rdo_model',
    'get_cone_model',
    'set_operation_mode',
    'train_rdo_with_peft',
    'ProjectionAdapter',
    'train_per_layer_vectors',
    'train_sft_with_peft'
]
PYEOF

cat > src/measurement/__init__.py << 'PYEOF'
"""Measurement functions for refusal evaluation."""

from .vllm_hybrid_measurement import (
    HybridMeasurement,
    HybridMeasurementConfig
)
from .scoring import score_responses

__all__ = [
    'HybridMeasurement',
    'HybridMeasurementConfig',
    'score_responses'
]
PYEOF

cat > src/utils/__init__.py << 'PYEOF'
"""Utility functions for model handling and generation."""

from .model_utils import load_model, apply_projection_hook
from .generate_utils import generate_with_projection
from .conversion_utils import convert_nnsight_to_pytorch

__all__ = [
    'load_model',
    'apply_projection_hook',
    'generate_with_projection',
    'convert_nnsight_to_pytorch'
]
PYEOF

echo "Created __init__.py files"
