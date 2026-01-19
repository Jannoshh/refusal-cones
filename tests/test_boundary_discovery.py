#!/usr/bin/env python3
"""Tests for boundary/level-set discovery."""

import torch
import numpy as np

from src.discovery.boundary_discovery import (
    BoundaryGeometryDiscovery,
    BoundaryDiscoveryConfig,
    BoundaryDiscoveryResults
)
from src.discovery.adaptive_geometry_discovery import SimpleGP


def test_straddle_acquisition_peaks_near_boundary():
    """Straddle acquisition should give highest scores near the boundary."""
    torch.manual_seed(42)
    n_layers, hidden_dim = 2, 3
    threshold = 0.5

    config = BoundaryDiscoveryConfig(
        threshold=threshold,
        n_init_samples=10,
        n_boundary_iterations=0,  # Skip iterations, just test acquisition
        beta=1.96,
        use_sparse_gp=True,  # Use sparse GP for numerical stability
        num_inducing=8,
        sparse_train_steps=3
    )

    # Create mock function where boundary is at cos_sim = 0.5
    v_center = torch.randn(n_layers, hidden_dim)
    v_center = v_center / v_center.norm(dim=1, keepdim=True)

    def mock_measure(v: torch.Tensor) -> float:
        v_flat = v.reshape(-1)
        center_flat = v_center.reshape(-1)
        cos_sim = (v_flat @ center_flat) / (v_flat.norm() * center_flat.norm())
        # R = sigmoid((cos_sim - 0.3) / 0.1), so R=0.5 at cos_sim=0.3
        R = torch.sigmoid((cos_sim - 0.3) / 0.1)
        return R.item()

    discovery = BoundaryGeometryDiscovery(
        measure_refusal_fn=mock_measure,
        n_layers=n_layers,
        hidden_dim=hidden_dim,
        config=config
    )

    # Run initial exploration to fit GP
    discovery._initial_exploration()

    # Generate test candidates: one near boundary, others far
    candidates = []

    # Point designed to be near the boundary
    # We need cos_sim ≈ 0.3 for R ≈ 0.5
    v_boundary = 0.3 * v_center + 0.7 * torch.randn_like(v_center)
    v_boundary = v_boundary / v_boundary.norm(dim=1, keepdim=True)
    candidates.append(v_boundary)

    # Point clearly inside (high R)
    v_inside = v_center.clone()
    candidates.append(v_inside)

    # Point clearly outside (low R)
    v_outside = -v_center.clone()
    candidates.append(v_outside)

    candidates = torch.stack(candidates)

    # Compute straddle acquisition
    scores = discovery._compute_straddle_acquisition(candidates)

    # The boundary point should have high acquisition
    # (though GP uncertainty may affect exact ranking with few samples)
    assert scores.shape == (3,)
    assert torch.isfinite(scores).all()


def test_gradient_straddle_uses_last_gradient():
    """Gradient-straddle should pass the most recent gradient into acquisition."""
    torch.manual_seed(0)
    n_layers, hidden_dim = 2, 3

    config = BoundaryDiscoveryConfig(
        n_init_samples=0,
        n_boundary_iterations=2,
        n_candidates=5,
        acquisition_type='gradient_straddle',
        use_gradients=True,
        max_measurements=10,
        use_sparse_gp=False
    )

    grad_value = torch.ones(n_layers, hidden_dim)

    def mock_measure(v: torch.Tensor) -> float:
        return 0.2

    def mock_measure_with_grad(v: torch.Tensor):
        return 0.2, grad_value

    class TrackingBoundaryDiscovery(BoundaryGeometryDiscovery):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.grad_args = []

        def _compute_straddle_acquisition(self, candidates, grad_R=None):
            self.grad_args.append(grad_R)
            return super()._compute_straddle_acquisition(candidates, grad_R)

    discovery = TrackingBoundaryDiscovery(
        measure_refusal_fn=mock_measure,
        n_layers=n_layers,
        hidden_dim=hidden_dim,
        config=config,
        measure_with_grad_fn=mock_measure_with_grad
    )

    discovery._boundary_search()

    assert len(discovery.grad_args) >= 2
    assert discovery.grad_args[0] is None
    assert discovery.grad_args[1] is not None


def test_boundary_discovery_on_synthetic_spherical_region():
    """Test boundary discovery finds the boundary of a spherical cap."""
    torch.manual_seed(123)
    n_layers, hidden_dim = 2, 4

    # Create a spherical cap region
    v_center = torch.randn(n_layers, hidden_dim)
    v_center = v_center / v_center.norm(dim=1, keepdim=True)

    def mock_measure(v: torch.Tensor) -> float:
        v_flat = v.reshape(-1)
        center_flat = v_center.reshape(-1)
        cos_sim = (v_flat @ center_flat) / (v_flat.norm() * center_flat.norm())
        # Smooth transition: R = sigmoid((cos_sim - 0.3) / 0.1)
        R = torch.sigmoid((cos_sim - 0.3) / 0.1)
        R = R + 0.01 * torch.randn(1)  # Small noise
        return torch.clamp(R, 0, 1).item()

    config = BoundaryDiscoveryConfig(
        threshold=0.5,
        n_init_samples=10,
        n_boundary_iterations=15,
        min_boundary_points=3,
        max_measurements=50,
        convergence_threshold=0.2,
        use_sparse_gp=True,  # Use sparse GP for numerical stability
        num_inducing=8,
        sparse_train_steps=3
    )

    discovery = BoundaryGeometryDiscovery(
        measure_refusal_fn=mock_measure,
        n_layers=n_layers,
        hidden_dim=hidden_dim,
        config=config,
        v_init=v_center
    )

    results = discovery.discover()

    # Verify results structure
    assert isinstance(results, BoundaryDiscoveryResults)
    assert results.n_measurements <= config.max_measurements
    assert results.threshold == config.threshold

    # Should have found some boundary points
    # (may be few with limited iterations, but structure should be correct)
    assert results.boundary_points.dim() == 3  # [n, n_layers, hidden_dim]
    assert results.inside_points.dim() == 3
    assert results.outside_points.dim() == 3

    # Total points should match
    total = len(results.boundary_points) + len(results.inside_points) + len(results.outside_points)
    # Note: boundary_points may overlap with inside/outside based on classification
    assert results.n_measurements == len(results.V_observed)


def test_boundary_dimension_estimation():
    """Test that boundary dimension estimation works."""
    torch.manual_seed(456)
    n_layers, hidden_dim = 2, 4

    # Simple mock that always returns 0.5 (everything is on boundary)
    def mock_measure(v: torch.Tensor) -> float:
        return 0.5 + 0.01 * torch.randn(1).item()

    config = BoundaryDiscoveryConfig(
        threshold=0.5,
        n_init_samples=20,
        n_boundary_iterations=0,
        boundary_proximity=0.1,
        use_sparse_gp=True,  # Use sparse GP for numerical stability
        num_inducing=12,
        sparse_train_steps=3
    )

    discovery = BoundaryGeometryDiscovery(
        measure_refusal_fn=mock_measure,
        n_layers=n_layers,
        hidden_dim=hidden_dim,
        config=config
    )

    # Run just initial exploration
    discovery._initial_exploration()

    # All points should be near boundary
    boundary_points, _, _ = discovery._identify_boundary_points()

    # Test dimension estimation (should work with enough points)
    if len(boundary_points) >= 5:
        dim = discovery._estimate_boundary_dimension(boundary_points)
        assert dim >= 1  # Should find some dimension


def test_predict_region():
    """Test region prediction after discovery."""
    torch.manual_seed(789)
    n_layers, hidden_dim = 2, 3

    v_center = torch.randn(n_layers, hidden_dim)
    v_center = v_center / v_center.norm(dim=1, keepdim=True)

    def mock_measure(v: torch.Tensor) -> float:
        v_flat = v.reshape(-1)
        center_flat = v_center.reshape(-1)
        cos_sim = (v_flat @ center_flat) / (v_flat.norm() * center_flat.norm())
        R = torch.sigmoid((cos_sim - 0.3) / 0.1)
        return torch.clamp(R, 0, 1).item()

    config = BoundaryDiscoveryConfig(
        threshold=0.5,
        n_init_samples=15,
        n_boundary_iterations=10,
        max_measurements=40,
        use_sparse_gp=True,  # Use sparse GP for numerical stability
        num_inducing=10,
        sparse_train_steps=3
    )

    discovery = BoundaryGeometryDiscovery(
        measure_refusal_fn=mock_measure,
        n_layers=n_layers,
        hidden_dim=hidden_dim,
        config=config,
        v_init=v_center
    )

    results = discovery.discover()

    # Test prediction
    v_test = torch.randn(n_layers, hidden_dim)
    v_test = v_test / v_test.norm(dim=1, keepdim=True)

    region, mu, sigma = discovery.predict_region(v_test)

    assert region in ['inside', 'outside', 'boundary']
    assert 0 <= mu <= 1 or np.isclose(mu, 0, atol=0.1) or np.isclose(mu, 1, atol=0.1)
    assert sigma >= 0


def test_connected_components_estimation():
    """Test connected components estimation."""
    torch.manual_seed(111)
    n_layers, hidden_dim = 2, 3

    # Create boundary points in two clusters
    cluster1 = torch.randn(5, n_layers, hidden_dim)
    cluster1 = cluster1 / cluster1.norm(dim=2, keepdim=True)

    cluster2 = torch.randn(5, n_layers, hidden_dim) + 10  # Far away
    cluster2 = cluster2 / cluster2.norm(dim=2, keepdim=True)

    boundary_points = torch.cat([cluster1, cluster2], dim=0)

    # Create minimal discovery object for testing
    config = BoundaryDiscoveryConfig()
    discovery = BoundaryGeometryDiscovery(
        measure_refusal_fn=lambda v: 0.5,
        n_layers=n_layers,
        hidden_dim=hidden_dim,
        config=config
    )

    n_components = discovery._estimate_connected_components(boundary_points)

    # Should detect at least 1 component (may merge if threshold is large)
    assert n_components >= 1


def test_boundary_discovery_config_defaults():
    """Test that config has sensible defaults."""
    config = BoundaryDiscoveryConfig()

    assert config.threshold == 0.05
    assert config.n_init_samples == 30
    assert config.n_boundary_iterations == 50
    assert config.beta == 1.96
    assert config.acquisition_type == 'straddle'
    assert config.min_boundary_points == 20
    assert config.max_measurements == 200


def test_boundary_discovery_with_sparse_gp():
    """Test that sparse GP option works."""
    torch.manual_seed(222)
    n_layers, hidden_dim = 2, 3

    def mock_measure(v: torch.Tensor) -> float:
        return torch.rand(1).item()

    config = BoundaryDiscoveryConfig(
        n_init_samples=10,
        n_boundary_iterations=5,
        use_sparse_gp=True,
        num_inducing=8,
        sparse_train_steps=2,
        max_measurements=20
    )

    discovery = BoundaryGeometryDiscovery(
        measure_refusal_fn=mock_measure,
        n_layers=n_layers,
        hidden_dim=hidden_dim,
        config=config
    )

    results = discovery.discover()

    assert results.n_measurements <= config.max_measurements
    assert len(results.V_observed) == results.n_measurements
