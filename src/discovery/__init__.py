"""Geometry discovery methods for refusal vectors."""

from .gradient_discovery import (
    GradientGeometryDiscovery,
    GradientDiscoveryConfig,
    riemannian_gradient_ascent,
    sample_von_mises_fisher
)
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

__all__ = [
    'GradientGeometryDiscovery',
    'GradientDiscoveryConfig',
    'riemannian_gradient_ascent',
    'sample_von_mises_fisher',
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
]
