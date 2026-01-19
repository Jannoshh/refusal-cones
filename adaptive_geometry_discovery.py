#!/usr/bin/env python3
"""
Adaptive Geometry Discovery for Refusal Subspaces

Instead of fixed cones, this discovers the true geometry of refusal
directions using:
- Gaussian Process modeling of refusal landscape R(v)
- Active exploration with acquisition functions
- Automatic geometry extraction (intrinsic dimension, boundaries, modes)

Theoretical foundation in ADAPTIVE_GEOMETRY_DISCOVERY.md
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Optional, Callable
from dataclasses import dataclass


@dataclass
class GeometryConfig:
    """Configuration for adaptive geometry discovery."""

    # Exploration
    n_init_random: int = 20  # Initial random samples
    n_iterations: int = 100  # Active exploration iterations
    n_candidates: int = 1000  # Candidates per iteration

    # Acquisition
    acquisition_type: str = 'ucb'  # 'ucb', 'ei', or 'boundary'
    beta: float = 2.0  # UCB exploration parameter

    # GP (if using sklearn/gpytorch)
    kernel_type: str = 'rbf'  # 'rbf', 'linear', or 'rbf+linear'
    kernel_lengthscale: float = 1.0

    # Stopping
    convergence_threshold: float = 0.01

    # Geometry extraction
    boundary_threshold: float = 0.5  # R(v) threshold for refusal
    min_intrinsic_dim_samples: int = 20


class SimpleGP:
    """
    Simplified Gaussian Process for refusal strength R(v).

    Uses RBF kernel with sklearn for simplicity.
    For production, use gpytorch for better scaling.
    """

    def __init__(self, kernel_type='rbf', lengthscale=1.0):
        self.kernel_type = kernel_type
        self.lengthscale = lengthscale

        # Will store training data
        self.V_train = None
        self.R_train = None

        # Kernel parameters (can be learned)
        self.noise_var = 0.01

    def fit(self, V: torch.Tensor, R: torch.Tensor):
        """
        Fit GP to observations.

        Args:
            V: Observed directions [n, d]
            R: Observed refusal strengths [n]
        """
        self.V_train = V
        self.R_train = R

    def predict(self, V_test: torch.Tensor) -> tuple:
        """
        Predict R(v) for test directions.

        Args:
            V_test: Test directions [m, d]

        Returns:
            mean: Predicted R(v) [m]
            std: Uncertainty (standard deviation) [m]
        """
        if self.V_train is None:
            # Prior: mean 0, high uncertainty
            mean = torch.zeros(len(V_test))
            std = torch.ones(len(V_test))
            return mean, std

        # Compute kernels
        K_train_train = self._kernel(self.V_train, self.V_train)
        K_test_train = self._kernel(V_test, self.V_train)
        K_test_test = self._kernel(V_test, V_test)

        # Add noise to diagonal
        K_train_train_noisy = K_train_train + self.noise_var * torch.eye(len(K_train_train))

        # Solve for weights
        L = torch.linalg.cholesky(K_train_train_noisy)
        alpha = torch.cholesky_solve(self.R_train.unsqueeze(-1), L).squeeze(-1)

        # Predictive mean
        mean = K_test_train @ alpha

        # Predictive variance
        v = torch.triangular_solve(K_test_train.T, L, upper=False)[0]
        var = torch.diag(K_test_test) - (v ** 2).sum(dim=0)

        # Ensure non-negative variance
        var = torch.clamp(var, min=1e-6)
        std = torch.sqrt(var)

        return mean, std

    def _kernel(self, V1: torch.Tensor, V2: torch.Tensor) -> torch.Tensor:
        """
        Compute kernel matrix.

        Args:
            V1: Directions [n, d]
            V2: Directions [m, d]

        Returns:
            K: Kernel matrix [n, m]
        """
        if self.kernel_type == 'rbf':
            # RBF kernel on unit sphere (geodesic distance)
            # k(v, v') = exp(-dist(v, v')^2 / 2σ^2)

            # Dot products (cosine similarity for unit vectors)
            dots = V1 @ V2.T

            # Clamp for numerical stability
            dots = torch.clamp(dots, -1, 1)

            # Geodesic distance on sphere
            dist_sq = 2 * (1 - dots)  # Squared chord distance (proportional to geodesic)

            K = torch.exp(-dist_sq / (2 * self.lengthscale ** 2))

        elif self.kernel_type == 'linear':
            # Linear kernel (dot product)
            K = V1 @ V2.T

        elif self.kernel_type == 'rbf+linear':
            # Composite kernel
            K_rbf = self._kernel_rbf(V1, V2)
            K_linear = V1 @ V2.T
            K = 0.5 * K_rbf + 0.5 * K_linear

        else:
            raise ValueError(f"Unknown kernel: {self.kernel_type}")

        return K


class RefusalGeometryDiscovery:
    """
    Discover the geometry of refusal subspace adaptively.

    Uses Gaussian Process + active exploration to find:
    - Principal refusal directions
    - Intrinsic dimension
    - Boundary of refusal subspace
    """

    def __init__(
        self,
        measure_refusal_fn: Callable,  # Function to measure R(v) for direction v
        n_layers: int,
        hidden_dim: int,
        config: Optional[GeometryConfig] = None
    ):
        """
        Initialize geometry discovery.

        Args:
            measure_refusal_fn: Function that takes vectors and returns refusal strength
            n_layers: Number of layers
            hidden_dim: Hidden dimension
            config: Configuration
        """
        self.measure_refusal = measure_refusal_fn
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.config = config or GeometryConfig()

        # Will store observations
        self.V_observed = []
        self.R_observed = []

        # GP model
        self.gp = SimpleGP(
            kernel_type=self.config.kernel_type,
            lengthscale=self.config.kernel_lengthscale
        )

    def discover(self) -> Dict:
        """
        Run adaptive geometry discovery.

        Returns:
            results: Dictionary with discovered geometry
        """
        print("=" * 70)
        print("Adaptive Geometry Discovery")
        print("=" * 70)

        # Phase 1: Random initialization
        print(f"\nPhase 1: Random initialization ({self.config.n_init_random} samples)")
        self._initialize_random()

        # Phase 2: Active exploration
        print(f"\nPhase 2: Active exploration ({self.config.n_iterations} iterations)")
        self._active_exploration()

        # Phase 3: Geometry extraction
        print("\nPhase 3: Geometry extraction")
        geometry = self._extract_geometry()

        # Package results
        results = {
            'V_observed': torch.stack(self.V_observed),
            'R_observed': torch.tensor(self.R_observed),
            'gp': self.gp,
            'geometry': geometry
        }

        return results

    def _initialize_random(self):
        """Initialize with random directions."""

        for i in range(self.config.n_init_random):
            # Random direction
            v = torch.randn(self.n_layers, self.hidden_dim)
            v = v / v.norm(dim=1, keepdim=True)  # Normalize per layer

            # Measure refusal strength
            R = self.measure_refusal(v)

            self.V_observed.append(v)
            self.R_observed.append(R)

            print(f"  Sample {i+1}/{self.config.n_init_random}: R = {R:.3f}")

        # Fit initial GP
        V_tensor = torch.stack(self.V_observed)
        R_tensor = torch.tensor(self.R_observed)
        self.gp.fit(V_tensor, R_tensor)

    def _active_exploration(self):
        """Active exploration using acquisition functions."""

        for iteration in range(self.config.n_iterations):
            # Generate candidate directions
            candidates = self._generate_candidates(self.config.n_candidates)

            # Compute acquisition scores
            scores = self._compute_acquisition(candidates)

            # Select best candidate
            best_idx = scores.argmax()
            v_next = candidates[best_idx]

            # Measure refusal strength
            R_next = self.measure_refusal(v_next)

            # Update observations
            self.V_observed.append(v_next)
            self.R_observed.append(R_next)

            # Update GP
            V_tensor = torch.stack(self.V_observed)
            R_tensor = torch.tensor(self.R_observed)
            self.gp.fit(V_tensor, R_tensor)

            print(f"  Iteration {iteration+1}: R = {R_next:.3f}, Acq = {scores[best_idx]:.3f}")

            # Check convergence
            if scores.max() < self.config.convergence_threshold:
                print("  Converged!")
                break

    def _generate_candidates(self, n: int) -> torch.Tensor:
        """Generate candidate directions to evaluate."""

        # Random sampling on sphere
        candidates = torch.randn(n, self.n_layers, self.hidden_dim)
        candidates = candidates / candidates.norm(dim=2, keepdim=True)

        # Could add: Sampling near observed high-R points, boundary points, etc.

        return candidates

    def _compute_acquisition(self, candidates: torch.Tensor) -> torch.Tensor:
        """Compute acquisition function scores."""

        # Reshape for GP: [n_candidates, n_layers * hidden_dim]
        V_flat = candidates.reshape(len(candidates), -1)

        # GP predictions
        mean, std = self.gp.predict(V_flat)

        # Compute acquisition
        if self.config.acquisition_type == 'ucb':
            # Upper Confidence Bound
            scores = mean + self.config.beta * std

        elif self.config.acquisition_type == 'ei':
            # Expected Improvement
            R_max = max(self.R_observed)
            improvement = mean - R_max
            Z = improvement / (std + 1e-8)

            # Standard normal CDF and PDF
            from scipy.stats import norm
            cdf = torch.tensor([norm.cdf(z.item()) for z in Z])
            pdf = torch.tensor([norm.pdf(z.item()) for z in Z])

            scores = improvement * cdf + std * pdf

        elif self.config.acquisition_type == 'boundary':
            # Boundary exploration
            distance_to_threshold = torch.abs(mean - self.config.boundary_threshold)
            scores = -distance_to_threshold * std  # Want high uncertainty near threshold

        else:
            raise ValueError(f"Unknown acquisition: {self.config.acquisition_type}")

        return scores

    def _extract_geometry(self) -> Dict:
        """Extract geometric properties from GP."""

        geometry = {}

        V_tensor = torch.stack(self.V_observed)
        R_tensor = torch.tensor(self.R_observed)

        # 1. Find principal directions (local maxima of R)
        print("  Finding principal refusal directions...")
        principal_dirs = self._find_principal_directions(V_tensor, R_tensor)
        geometry['principal_directions'] = principal_dirs
        geometry['n_modes'] = len(principal_dirs)
        print(f"    Found {len(principal_dirs)} principal directions")

        # 2. Estimate intrinsic dimension
        print("  Estimating intrinsic dimension...")
        high_R_points = V_tensor[R_tensor > self.config.boundary_threshold]

        if len(high_R_points) >= self.config.min_intrinsic_dim_samples:
            intrinsic_dim = self._estimate_intrinsic_dimension(high_R_points)
            geometry['intrinsic_dimension'] = intrinsic_dim
            print(f"    Intrinsic dimension: {intrinsic_dim}")
        else:
            print(f"    Not enough high-R samples for dimension estimation")

        # 3. Characterize spread
        print("  Analyzing spread...")
        if len(high_R_points) > 0:
            # Pairwise distances
            dists = torch.cdist(high_R_points.reshape(len(high_R_points), -1),
                               high_R_points.reshape(len(high_R_points), -1))
            geometry['avg_intra_distance'] = dists.mean().item()
            geometry['max_intra_distance'] = dists.max().item()
            print(f"    Avg distance: {geometry['avg_intra_distance']:.3f}")

        return geometry

    def _find_principal_directions(self, V_observed: torch.Tensor, R_observed: torch.Tensor) -> List[torch.Tensor]:
        """Find principal refusal directions (local maxima)."""

        # Simple approach: Return top-k observed directions
        # More sophisticated: Optimize to local maxima

        top_k = min(5, len(R_observed))
        topk_values, topk_indices = R_observed.topk(top_k)

        principal_dirs = [V_observed[idx] for idx in topk_indices if topk_values[topk_indices == idx] > self.config.boundary_threshold]

        return principal_dirs

    def _estimate_intrinsic_dimension(self, points: torch.Tensor) -> int:
        """Estimate intrinsic dimension using PCA."""

        # Flatten points
        points_flat = points.reshape(len(points), -1).numpy()

        # PCA
        from sklearn.decomposition import PCA
        pca = PCA()
        pca.fit(points_flat)

        # Find number of components for 95% variance
        cumsum = np.cumsum(pca.explained_variance_ratio_)
        intrinsic_dim = int(np.argmax(cumsum > 0.95) + 1)

        return intrinsic_dim


# Example usage
if __name__ == '__main__':
    """
    Example: Discover geometry of refusal subspace.
    """

    print("=" * 70)
    print("Adaptive Geometry Discovery - Demo")
    print("=" * 70)

    # Mock refusal measurement function
    def mock_measure_refusal(v: torch.Tensor) -> float:
        """
        Mock refusal strength measurement.

        Simulates a refusal subspace with:
        - 2 principal modes (two different refusal directions)
        - Intrinsic dimension ~3
        """
        # Define two "true" refusal directions
        true_dir1 = torch.randn(26, 2048)
        true_dir1 = true_dir1 / true_dir1.norm(dim=1, keepdim=True)

        true_dir2 = torch.randn(26, 2048)
        true_dir2 = true_dir2 / true_dir2.norm(dim=1, keepdim=True)

        # Measure alignment with true directions
        v_flat = v.reshape(-1)
        dir1_flat = true_dir1.reshape(-1)
        dir2_flat = true_dir2.reshape(-1)

        alignment1 = (v_flat @ dir1_flat) / (v_flat.norm() * dir1_flat.norm())
        alignment2 = (v_flat @ dir2_flat) / (v_flat.norm() * dir2_flat.norm())

        # Refusal strength: high if aligned with either direction
        R = max(abs(alignment1.item()), abs(alignment2.item()))

        # Add noise
        R = R + 0.1 * torch.randn(1).item()
        R = np.clip(R, 0, 1)

        return R

    # Run discovery
    config = GeometryConfig(
        n_init_random=10,  # Small for demo
        n_iterations=20,
        n_candidates=100,
        acquisition_type='ucb',
        beta=2.0
    )

    discovery = RefusalGeometryDiscovery(
        measure_refusal_fn=mock_measure_refusal,
        n_layers=26,
        hidden_dim=2048,
        config=config
    )

    results = discovery.discover()

    # Print results
    print("\n" + "=" * 70)
    print("Results")
    print("=" * 70)

    geometry = results['geometry']

    print(f"\nDiscovered geometry:")
    print(f"  Principal directions: {geometry['n_modes']}")
    if 'intrinsic_dimension' in geometry:
        print(f"  Intrinsic dimension: {geometry['intrinsic_dimension']}")
    if 'avg_intra_distance' in geometry:
        print(f"  Avg spread: {geometry['avg_intra_distance']:.3f}")

    print(f"\nTotal measurements: {len(results['R_observed'])}")
    print(f"Max refusal strength: {results['R_observed'].max():.3f}")

    print("\n" + "=" * 70)
    print("Next steps:")
    print("  1. Use discovered principal directions for cone initialization")
    print("  2. Use intrinsic dimension to set cone rank")
    print("  3. Refine with full RDO training")
