#!/usr/bin/env python3
"""
Boundary/Level-Set Discovery for Refusal Geometry

Finds the boundary of the refusal region (where R(v) ≈ threshold) instead of
finding peaks (argmax R(v)). This characterizes the complete region on the
hypersphere where refusal is mediated.

Key insight: Knowing the boundary = knowing everything about the region.

Uses straddle acquisition: acquisition(v) = beta * sigma(v) - |mu(v) - threshold|
- High score when uncertain AND near threshold crossing
- Optionally enhanced with gradient info for sampling along contours
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Callable, List, Dict, Optional, Tuple, Union
from dataclasses import dataclass, field

from .adaptive_geometry_discovery import SimpleGP, AdaptiveSparseGP


@dataclass
class BoundaryDiscoveryConfig:
    """Configuration for boundary/level-set discovery."""

    # Threshold for the boundary (R(v) level to find)
    threshold: float = 0.05  # Stricter boundary by default

    # Phase 1: Initial exploration
    n_init_samples: int = 30  # Initial stratified samples

    # Phase 2: Boundary search
    n_boundary_iterations: int = 50  # Boundary tracing iterations
    acquisition_type: str = 'straddle'  # 'straddle' or 'gradient_straddle'
    beta: float = 1.96  # Exploration parameter (95% confidence)
    n_candidates: int = 500  # Candidates per iteration

    # Gradient enhancement (for 'gradient_straddle')
    use_gradients: bool = True  # Use gradient info for sampling
    gradient_alpha: float = 0.5  # Weight for perpendicular bonus

    # Convergence
    convergence_threshold: float = 0.05  # Max uncertainty at boundary
    min_boundary_points: int = 20  # Minimum boundary points needed
    max_measurements: int = 200  # Hard limit on measurements

    # GP settings
    kernel_type: str = 'rbf'
    kernel_lengthscale: float = 0.3
    use_sparse_gp: bool = False
    num_inducing: int = 64
    sparse_train_steps: int = 15
    sparse_lr: float = 0.05
    sparse_max_train_points: int = 256
    sparse_jitter: float = 1e-5

    # Sampling near boundary
    boundary_proximity: float = 0.1  # How close to classify as "near boundary"
    near_boundary_sample_ratio: float = 0.5  # Fraction of candidates from near boundary


@dataclass
class BoundaryDiscoveryResults:
    """Results from boundary discovery."""

    # Boundary characterization
    boundary_points: torch.Tensor  # [n_boundary, n_layers, hidden_dim]
    boundary_uncertainties: torch.Tensor  # [n_boundary]
    boundary_R_values: torch.Tensor  # [n_boundary] actual R values

    # Region classification
    inside_points: torch.Tensor  # R > threshold
    outside_points: torch.Tensor  # R < threshold

    # Geometry
    boundary_dimension: int  # Estimated dimension of boundary manifold
    n_connected_components: int  # Number of separate boundary regions

    # GP model (for interpolation)
    gp: Union[SimpleGP, AdaptiveSparseGP]

    # Settings used
    threshold: float
    n_measurements: int

    # All observations
    V_observed: torch.Tensor
    R_observed: torch.Tensor


class BoundaryGeometryDiscovery:
    """
    Discover the boundary of the refusal region adaptively.

    Uses straddle acquisition to efficiently find where R(v) ≈ threshold,
    characterizing the complete region on the hypersphere.
    """

    def __init__(
        self,
        measure_refusal_fn: Callable,
        n_layers: int,
        hidden_dim: int,
        config: Optional[BoundaryDiscoveryConfig] = None,
        measure_with_grad_fn: Optional[Callable] = None,
        v_init: Optional[torch.Tensor] = None
    ):
        """
        Initialize boundary discovery.

        Args:
            measure_refusal_fn: Function that takes v and returns R(v) ∈ [0, 1]
            n_layers: Number of layers
            hidden_dim: Hidden dimension
            config: Configuration
            measure_with_grad_fn: Optional function returning (R, grad_R) for
                                  gradient-enhanced acquisition
            v_init: Optional initial direction (e.g., existing refusal vector)
                    to bias initial sampling
        """
        self.measure_refusal = measure_refusal_fn
        self.measure_with_grad = measure_with_grad_fn
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.config = config or BoundaryDiscoveryConfig()
        self.v_init = v_init
        self._last_grad: Optional[torch.Tensor] = None

        # Observations
        self.V_observed: List[torch.Tensor] = []
        self.R_observed: List[float] = []

        # Boundary tracking
        self.boundary_candidates: List[torch.Tensor] = []

        # GP model
        if self.config.use_sparse_gp:
            self.gp = AdaptiveSparseGP(
                kernel_type=self.config.kernel_type,
                lengthscale=self.config.kernel_lengthscale,
                num_inducing=self.config.num_inducing,
                train_steps=self.config.sparse_train_steps,
                lr=self.config.sparse_lr,
                max_train_points=self.config.sparse_max_train_points,
                jitter=self.config.sparse_jitter
            )
        else:
            self.gp = SimpleGP(
                kernel_type=self.config.kernel_type,
                lengthscale=self.config.kernel_lengthscale
            )

    def discover(self) -> BoundaryDiscoveryResults:
        """
        Run boundary discovery.

        Returns:
            BoundaryDiscoveryResults with boundary characterization
        """
        print("=" * 70)
        print("Boundary/Level-Set Discovery")
        print("=" * 70)
        print(f"\nTarget: Find boundary where R(v) ≈ {self.config.threshold}")
        print(f"Dimension: d = {self.n_layers} × {self.hidden_dim} = {self.n_layers * self.hidden_dim:,}")

        # ========================================
        # Phase 1: Initial Stratified Sampling
        # ========================================
        print("\n" + "=" * 70)
        print(f"Phase 1: Initial Exploration ({self.config.n_init_samples} samples)")
        print("=" * 70)
        self._initial_exploration()

        # ========================================
        # Phase 2: Boundary Search with Straddle
        # ========================================
        print("\n" + "=" * 70)
        print(f"Phase 2: Boundary Search ({self.config.n_boundary_iterations} iterations)")
        print("=" * 70)
        print(f"\nUsing {self.config.acquisition_type} acquisition")
        self._boundary_search()

        # ========================================
        # Phase 3: Boundary Characterization
        # ========================================
        print("\n" + "=" * 70)
        print("Phase 3: Boundary Characterization")
        print("=" * 70)
        results = self._characterize_boundary()

        # ========================================
        # Summary
        # ========================================
        print("\n" + "=" * 70)
        print("Discovery Complete")
        print("=" * 70)
        print(f"\nTotal measurements: {results.n_measurements}")
        print(f"Boundary points found: {len(results.boundary_points)}")
        print(f"Boundary dimension: {results.boundary_dimension}")
        print(f"Connected components: {results.n_connected_components}")
        print(f"Inside region: {len(results.inside_points)} points")
        print(f"Outside region: {len(results.outside_points)} points")

        return results

    def _initial_exploration(self):
        """Phase 1: Initial stratified sampling on hypersphere."""

        # Sample uniformly on hypersphere
        for i in range(self.config.n_init_samples):
            # Stratified: mix of uniform and near-prior (if available)
            if self.v_init is not None and i < self.config.n_init_samples // 3:
                # Sample near initial direction
                v = self.v_init + 0.5 * torch.randn_like(self.v_init)
            else:
                # Uniform on sphere
                v = torch.randn(self.n_layers, self.hidden_dim)

            # Normalize to unit sphere
            v = v / v.norm(dim=1, keepdim=True)

            # Measure
            R = self.measure_refusal(v)

            self.V_observed.append(v)
            self.R_observed.append(R)

            if i % 10 == 0:
                print(f"  Sample {i+1}/{self.config.n_init_samples}: R = {R:.4f}")

        # Fit initial GP
        self._update_gp()

        # Report initial findings
        R_tensor = torch.tensor(self.R_observed)
        n_inside = (R_tensor > self.config.threshold).sum().item()
        n_boundary = ((R_tensor - self.config.threshold).abs() < self.config.boundary_proximity).sum().item()

        print(f"\n  Initial statistics:")
        print(f"    Inside (R > {self.config.threshold}): {n_inside}")
        print(f"    Near boundary: {n_boundary}")
        print(f"    R range: [{R_tensor.min():.4f}, {R_tensor.max():.4f}]")

    def _boundary_search(self):
        """Phase 2: Boundary search using straddle acquisition."""

        converged = False

        for iteration in range(self.config.n_boundary_iterations):
            # Check termination
            if len(self.R_observed) >= self.config.max_measurements:
                print(f"\n  Reached max measurements ({self.config.max_measurements})")
                break

            # Generate candidates
            candidates = self._generate_candidates()

            # Compute acquisition scores
            grad_for_acq = None
            if self.config.use_gradients and self.config.acquisition_type == 'gradient_straddle':
                grad_for_acq = self._last_grad
            scores = self._compute_straddle_acquisition(candidates, grad_for_acq)

            # Select best candidate
            best_idx = scores.argmax()
            v_next = candidates[best_idx]

            # Measure
            if self.config.use_gradients and self.measure_with_grad is not None:
                R_next, grad_next = self.measure_with_grad(v_next)
            else:
                R_next = self.measure_refusal(v_next)
                grad_next = None
            if grad_next is not None:
                self._last_grad = grad_next.detach().to(v_next)

            self.V_observed.append(v_next)
            self.R_observed.append(R_next)

            # Track boundary candidates
            distance_to_boundary = abs(R_next - self.config.threshold)
            if distance_to_boundary < self.config.boundary_proximity:
                self.boundary_candidates.append(v_next)

            # Update GP
            self._update_gp()

            # Check convergence
            converged = self._check_convergence()

            if iteration % 10 == 0 or converged:
                n_boundary = len(self.boundary_candidates)
                print(f"  Iter {iteration}: R = {R_next:.4f}, "
                      f"|R - threshold| = {distance_to_boundary:.4f}, "
                      f"boundary pts = {n_boundary}, "
                      f"acq = {scores[best_idx]:.4f}")

            if converged:
                print(f"\n  Converged! Sufficient boundary points with low uncertainty.")
                break

    def _generate_candidates(self) -> torch.Tensor:
        """Generate candidate directions for evaluation."""

        n_candidates = self.config.n_candidates
        candidates = []

        # Mix of sampling strategies
        n_near_boundary = int(n_candidates * self.config.near_boundary_sample_ratio)
        n_uniform = n_candidates - n_near_boundary

        # 1. Sample uniformly on hypersphere
        for _ in range(n_uniform):
            v = torch.randn(self.n_layers, self.hidden_dim)
            v = v / v.norm(dim=1, keepdim=True)
            candidates.append(v)

        # 2. Sample near existing boundary candidates
        if len(self.boundary_candidates) > 0:
            for _ in range(n_near_boundary):
                # Pick random boundary candidate
                idx = np.random.randint(len(self.boundary_candidates))
                v_base = self.boundary_candidates[idx]

                # Perturb
                v = v_base + 0.2 * torch.randn_like(v_base)
                v = v / v.norm(dim=1, keepdim=True)
                candidates.append(v)
        else:
            # No boundary candidates yet, use uniform
            for _ in range(n_near_boundary):
                v = torch.randn(self.n_layers, self.hidden_dim)
                v = v / v.norm(dim=1, keepdim=True)
                candidates.append(v)

        return torch.stack(candidates)

    def _compute_straddle_acquisition(
        self,
        candidates: torch.Tensor,
        grad_R: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute straddle acquisition function.

        Straddle heuristic: acquisition(v) = beta * sigma(v) - |mu(v) - threshold|

        High scores when:
        - High uncertainty (exploring unknown regions)
        - Near threshold crossing (exploring boundary)

        Args:
            candidates: [n_candidates, n_layers, hidden_dim]
            grad_R: Optional gradient for enhancement

        Returns:
            scores: [n_candidates] acquisition scores
        """
        # Flatten candidates for GP
        V_flat = candidates.reshape(len(candidates), -1)

        # GP predictions
        mu, sigma = self.gp.predict(V_flat)

        # Base straddle acquisition
        distance_to_threshold = torch.abs(mu - self.config.threshold)
        scores = self.config.beta * sigma - distance_to_threshold

        # Optional gradient enhancement
        if (self.config.acquisition_type == 'gradient_straddle'
                and grad_R is not None):
            # Prefer sampling perpendicular to gradient (along contour)
            grad_on_device = grad_R.to(candidates)
            grad_normalized = grad_on_device / (grad_on_device.norm() + 1e-8)
            grad_flat = grad_normalized.reshape(-1)

            # Compute alignment with gradient for each candidate
            candidates_flat = candidates.reshape(len(candidates), -1)
            alignments = torch.abs(candidates_flat @ grad_flat)

            # Perpendicular bonus: 1 - |alignment|
            perpendicular_bonus = 1 - alignments
            scores = scores + self.config.gradient_alpha * perpendicular_bonus

        return scores

    def _identify_boundary_points(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Identify points near the boundary from observations.

        Returns:
            boundary_points: Points where |R - threshold| is small
            boundary_uncertainties: GP uncertainty at boundary points
            boundary_R_values: Actual R values at boundary points
        """
        V_tensor = torch.stack(self.V_observed)
        R_tensor = torch.tensor(self.R_observed)

        # Find points near threshold
        distance_to_threshold = torch.abs(R_tensor - self.config.threshold)
        boundary_mask = distance_to_threshold < self.config.boundary_proximity

        boundary_points = V_tensor[boundary_mask]
        boundary_R_values = R_tensor[boundary_mask]

        # Get GP uncertainty at boundary points
        if len(boundary_points) > 0:
            V_flat = boundary_points.reshape(len(boundary_points), -1)
            _, boundary_uncertainties = self.gp.predict(V_flat)
        else:
            boundary_uncertainties = torch.tensor([])

        return boundary_points, boundary_uncertainties, boundary_R_values

    def _check_convergence(self) -> bool:
        """Check if boundary discovery has converged."""

        boundary_points, boundary_uncertainties, _ = self._identify_boundary_points()

        # Need minimum number of boundary points
        if len(boundary_points) < self.config.min_boundary_points:
            return False

        # Check uncertainty at boundary
        if len(boundary_uncertainties) > 0:
            max_uncertainty = boundary_uncertainties.max().item()
            if max_uncertainty > self.config.convergence_threshold:
                return False

        return True

    def _characterize_boundary(self) -> BoundaryDiscoveryResults:
        """Phase 3: Characterize the discovered boundary."""

        V_tensor = torch.stack(self.V_observed)
        R_tensor = torch.tensor(self.R_observed)

        # Identify boundary, inside, outside points
        boundary_points, boundary_uncertainties, boundary_R_values = self._identify_boundary_points()

        inside_mask = R_tensor > self.config.threshold
        outside_mask = R_tensor <= self.config.threshold

        inside_points = V_tensor[inside_mask]
        outside_points = V_tensor[outside_mask]

        print(f"  Boundary points: {len(boundary_points)}")
        print(f"  Inside points: {len(inside_points)}")
        print(f"  Outside points: {len(outside_points)}")

        # Estimate boundary dimension using PCA
        boundary_dimension = self._estimate_boundary_dimension(boundary_points)
        print(f"  Boundary dimension: {boundary_dimension}")

        # Estimate connected components
        n_components = self._estimate_connected_components(boundary_points)
        print(f"  Connected components: {n_components}")

        return BoundaryDiscoveryResults(
            boundary_points=boundary_points,
            boundary_uncertainties=boundary_uncertainties,
            boundary_R_values=boundary_R_values,
            inside_points=inside_points,
            outside_points=outside_points,
            boundary_dimension=boundary_dimension,
            n_connected_components=n_components,
            gp=self.gp,
            threshold=self.config.threshold,
            n_measurements=len(self.R_observed),
            V_observed=V_tensor,
            R_observed=R_tensor
        )

    def _estimate_boundary_dimension(self, boundary_points: torch.Tensor) -> int:
        """Estimate intrinsic dimension of boundary using PCA."""

        if len(boundary_points) < 5:
            return 0

        # Flatten points
        points_flat = boundary_points.reshape(len(boundary_points), -1).numpy()

        # PCA
        from sklearn.decomposition import PCA
        pca = PCA()
        pca.fit(points_flat)

        # Find number of components for 95% variance
        cumsum = np.cumsum(pca.explained_variance_ratio_)
        intrinsic_dim = int(np.argmax(cumsum > 0.95) + 1)

        return intrinsic_dim

    def _estimate_connected_components(self, boundary_points: torch.Tensor) -> int:
        """Estimate number of connected components in boundary."""

        if len(boundary_points) < 2:
            return len(boundary_points)

        # Simple clustering based on pairwise distances
        points_flat = boundary_points.reshape(len(boundary_points), -1)
        dists = torch.cdist(points_flat, points_flat)

        # Use single-linkage clustering with distance threshold
        threshold = 0.5  # Connectivity threshold

        # Simple connected components via union-find
        n = len(boundary_points)
        parent = list(range(n))

        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py

        for i in range(n):
            for j in range(i + 1, n):
                if dists[i, j] < threshold:
                    union(i, j)

        # Count unique components
        components = set(find(i) for i in range(n))
        return len(components)

    def _update_gp(self):
        """Update GP with all observations."""
        if len(self.V_observed) > 0:
            V_flat = torch.stack(self.V_observed).reshape(len(self.V_observed), -1)
            R_tensor = torch.tensor(self.R_observed)
            self.gp.fit(V_flat, R_tensor)

    def predict_region(self, v: torch.Tensor) -> Tuple[str, float, float]:
        """
        Predict whether a point is inside, outside, or on boundary.

        Args:
            v: Direction [n_layers, hidden_dim]

        Returns:
            region: 'inside', 'outside', or 'boundary'
            mu: Predicted R(v)
            uncertainty: Prediction uncertainty
        """
        V_flat = v.reshape(1, -1)
        mu, sigma = self.gp.predict(V_flat)
        mu, sigma = mu.item(), sigma.item()

        distance_to_threshold = abs(mu - self.config.threshold)

        # Account for uncertainty in classification
        if distance_to_threshold < 2 * sigma:
            region = 'boundary'
        elif mu > self.config.threshold:
            region = 'inside'
        else:
            region = 'outside'

        return region, mu, sigma


# Example usage and demo
if __name__ == '__main__':
    """Demo: Discover boundary of a synthetic refusal region."""

    print("=" * 70)
    print("Boundary Discovery Demo")
    print("=" * 70)

    # Small dimensions for demo
    n_layers, hidden_dim = 2, 4
    d = n_layers * hidden_dim

    # Create synthetic refusal region: spherical cap
    # R(v) = 1 if v · v_center > cos(angle), else 0
    # This creates a circular boundary on the sphere

    v_center = torch.randn(n_layers, hidden_dim)
    v_center = v_center / v_center.norm(dim=1, keepdim=True)
    cap_angle = 0.5  # radians

    def mock_measure_refusal(v: torch.Tensor) -> float:
        """Mock: R(v) based on alignment with v_center."""
        v_flat = v.reshape(-1)
        center_flat = v_center.reshape(-1)

        # Cosine similarity
        cos_sim = (v_flat @ center_flat) / (v_flat.norm() * center_flat.norm())

        # Smooth transition at boundary
        # R transitions from 1 (inside) to 0 (outside) around cos(cap_angle)
        boundary_cos = np.cos(cap_angle)
        width = 0.1  # Transition width

        R = torch.sigmoid((cos_sim - boundary_cos) / width)

        # Add small noise
        R = R + 0.02 * torch.randn(1)
        R = torch.clamp(R, 0, 1)

        return R.item()

    # Configure discovery
    config = BoundaryDiscoveryConfig(
        threshold=0.5,  # Boundary where R = 0.5
        n_init_samples=15,
        n_boundary_iterations=30,
        beta=1.96,
        convergence_threshold=0.1,
        min_boundary_points=10,
        max_measurements=100
    )

    # Run discovery
    discovery = BoundaryGeometryDiscovery(
        measure_refusal_fn=mock_measure_refusal,
        n_layers=n_layers,
        hidden_dim=hidden_dim,
        config=config,
        v_init=v_center  # Start near the inside
    )

    results = discovery.discover()

    # Verify results
    print("\n" + "=" * 70)
    print("Verification")
    print("=" * 70)

    # Check boundary points are near threshold
    if len(results.boundary_R_values) > 0:
        boundary_distances = (results.boundary_R_values - config.threshold).abs()
        print(f"\nBoundary R values: {results.boundary_R_values[:5].tolist()}")
        print(f"Mean |R - threshold|: {boundary_distances.mean():.4f}")

    # Test prediction on new points
    print("\nPrediction test:")
    for _ in range(3):
        v_test = torch.randn(n_layers, hidden_dim)
        v_test = v_test / v_test.norm(dim=1, keepdim=True)

        region, mu, sigma = discovery.predict_region(v_test)
        R_true = mock_measure_refusal(v_test)

        print(f"  Predicted: {region} (μ={mu:.3f}, σ={sigma:.3f}), True R={R_true:.3f}")
