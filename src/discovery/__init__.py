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
