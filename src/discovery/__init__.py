"""Geometry discovery methods for refusal vectors."""

from .gradient_discovery import (
    GradientGeometryDiscovery,
    GradientDiscoveryConfig,
    riemannian_gradient_ascent,
    sample_von_mises_fisher
)
from .efficient_discovery import EfficientGeometryDiscovery
from .adaptive_geometry_discovery import (
    RefusalGeometryDiscovery,
    SimpleGP
)
from .boundary_discovery import (
    BoundaryGeometryDiscovery,
    BoundaryDiscoveryConfig,
    BoundaryDiscoveryResults
)
from .pareto_boundary_discovery import (
    ParetoGeometryDiscovery,
    ParetoDiscoveryConfig,
    ParetoDiscoveryResults,
    MultiObjectiveScorer,
    discover_pareto_boundary
)
from .adaptive_to_training import adaptive_discovery_to_training_pipeline

__all__ = [
    'GradientGeometryDiscovery',
    'GradientDiscoveryConfig',
    'riemannian_gradient_ascent',
    'sample_von_mises_fisher',
    'EfficientGeometryDiscovery',
    'RefusalGeometryDiscovery',
    'SimpleGP',
    'BoundaryGeometryDiscovery',
    'BoundaryDiscoveryConfig',
    'BoundaryDiscoveryResults',
    'ParetoGeometryDiscovery',
    'ParetoDiscoveryConfig',
    'ParetoDiscoveryResults',
    'MultiObjectiveScorer',
    'discover_pareto_boundary',
    'adaptive_discovery_to_training_pipeline'
]
