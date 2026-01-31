#!/usr/bin/env python3
"""
Single-Layer Discovery: Optimize (direction, layer_index) jointly.

Instead of searching over [n_layers, hidden_dim] matrices, we search over
(d, i) pairs where d ∈ R^{hidden_dim} and i ∈ {1, ..., n_layers}.

The GP learns:
- Which directions work well
- Which layers are most effective
- How nearby layers relate

We then extract per-layer boundaries and Pareto frontiers.
"""

import torch
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import numpy as np


@dataclass
class SingleLayerDiscoveryConfig:
    """Configuration for single-layer discovery."""

    # Boundary discovery
    use_dynamic_threshold: bool = True
    boundary_fraction: float = 0.05
    boundary_threshold: float = 0.0
    n_boundary_init: int = 30
    n_boundary_iterations: int = 50
    boundary_beta: float = 1.96

    # Pareto discovery
    n_pareto_init: int = 20
    n_pareto_iterations: int = 50
    pareto_beta: float = 0.5

    # Layer selection
    layer_start: int = 10  # First layer to consider
    layer_end: int = 24    # Last layer to consider

    # GP settings
    direction_lengthscale: float = 0.3
    layer_lengthscale: float = 2.0  # How correlated nearby layers are
    noise_var: float = 0.01

    # Total budget
    max_measurements: int = 200


@dataclass
class PerLayerResults:
    """Results for a single layer."""
    layer: int
    boundary_points: List[torch.Tensor]  # Directions near boundary
    inside_points: List[torch.Tensor]    # Directions where ablation works
    pareto_points: List[torch.Tensor]    # Pareto-optimal directions
    pareto_scores: Dict[str, List[float]]  # Scores for Pareto points
    best_refusal_score: float
    best_kl_score: float


@dataclass
class SingleLayerDiscoveryResults:
    """Full results from single-layer discovery."""
    per_layer: Dict[int, PerLayerResults]
    best_layer: int
    best_direction: torch.Tensor
    best_scores: Dict[str, float]
    all_observations: List[Tuple[torch.Tensor, int, Dict[str, float]]]  # (direction, layer, scores)
    hypervolume_per_layer: Dict[int, float]


class DirectionLayerGP:
    """
    GP over (direction, layer_index) pairs.

    Kernel factorizes as:
        K((d1, i), (d2, j)) = K_dir(d1, d2) * K_layer(i, j)

    This allows information sharing across layers while modeling
    layer-specific effects.
    """

    def __init__(
        self,
        hidden_dim: int,
        n_layers: int,
        direction_lengthscale: float = 0.3,
        layer_lengthscale: float = 2.0,
        noise_var: float = 0.01,
    ):
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.direction_lengthscale = direction_lengthscale
        self.layer_lengthscale = layer_lengthscale
        self.noise_var = noise_var

        # Training data
        self.D_train: List[torch.Tensor] = []  # Directions
        self.L_train: List[int] = []           # Layer indices
        self.R_train: List[float] = []         # Scores

        # Cached for prediction
        self.K_inv = None
        self.alpha = None

    def add_observation(self, d: torch.Tensor, layer: int, score: float):
        """Add a new observation."""
        self.D_train.append(d.detach().cpu())
        self.L_train.append(layer)
        self.R_train.append(score)
        self.K_inv = None  # Invalidate cache

    def _direction_kernel(self, d1: torch.Tensor, d2: torch.Tensor) -> float:
        """Compute direction kernel (geodesic RBF on sphere)."""
        # Move to CPU and convert to float32 for stable computation
        d1 = d1.detach().cpu().float()
        d2 = d2.detach().cpu().float()

        d1_norm = d1 / (d1.norm() + 1e-8)
        d2_norm = d2 / (d2.norm() + 1e-8)

        cos_sim = torch.dot(d1_norm.flatten(), d2_norm.flatten())
        cos_sim = torch.clamp(cos_sim, -1, 1)

        # Geodesic distance squared
        dist_sq = 2 * (1 - cos_sim)

        return torch.exp(-dist_sq / (2 * self.direction_lengthscale ** 2)).item()

    def _layer_kernel(self, i: int, j: int) -> float:
        """Compute layer kernel (RBF on layer index)."""
        dist_sq = (i - j) ** 2
        return np.exp(-dist_sq / (2 * self.layer_lengthscale ** 2))

    def _kernel(self, d1: torch.Tensor, l1: int, d2: torch.Tensor, l2: int) -> float:
        """Compute full kernel K((d1, l1), (d2, l2))."""
        return self._direction_kernel(d1, d2) * self._layer_kernel(l1, l2)

    def _compute_kernel_matrix(self) -> torch.Tensor:
        """Compute kernel matrix for training data."""
        n = len(self.D_train)
        K = torch.zeros(n, n)

        for i in range(n):
            for j in range(i, n):
                k = self._kernel(self.D_train[i], self.L_train[i],
                                self.D_train[j], self.L_train[j])
                K[i, j] = k
                K[j, i] = k

        return K

    def fit(self):
        """Fit GP to current observations."""
        if len(self.D_train) == 0:
            return

        K = self._compute_kernel_matrix()
        K = K + self.noise_var * torch.eye(len(self.D_train))

        # Cholesky for numerical stability
        try:
            L = torch.linalg.cholesky(K)
            R = torch.tensor(self.R_train)
            self.alpha = torch.cholesky_solve(R.unsqueeze(1), L).squeeze()
            self.K_inv = torch.cholesky_inverse(L)
        except:
            # Fallback to direct inverse
            self.K_inv = torch.linalg.inv(K)
            self.alpha = self.K_inv @ torch.tensor(self.R_train)

    def predict(self, d: torch.Tensor, layer: int) -> Tuple[float, float]:
        """
        Predict mean and std for a (direction, layer) pair.

        Returns:
            mu: Predicted mean score
            sigma: Predicted standard deviation
        """
        if len(self.D_train) == 0:
            return 0.0, 1.0

        if self.alpha is None:
            self.fit()

        # Compute kernel vector to training points (use float32 consistently)
        k_star = torch.tensor([
            self._kernel(d, layer, self.D_train[i], self.L_train[i])
            for i in range(len(self.D_train))
        ], dtype=torch.float32)

        # Mean prediction
        mu = torch.dot(k_star, self.alpha.float()).item()

        # Variance prediction
        k_star_star = self._kernel(d, layer, d, layer)
        var = k_star_star - k_star @ self.K_inv.float() @ k_star
        sigma = np.sqrt(max(var.item(), 1e-8))

        return mu, sigma

    def predict_batch(
        self,
        directions: List[torch.Tensor],
        layers: List[int]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict for multiple (direction, layer) pairs."""
        mus = []
        sigmas = []

        for d, l in zip(directions, layers):
            mu, sigma = self.predict(d, l)
            mus.append(mu)
            sigmas.append(sigma)

        return torch.tensor(mus), torch.tensor(sigmas)


class SingleLayerDiscovery:
    """
    Two-phase discovery over (direction, layer) space.

    Phase 1: Find boundary for each layer (where ablation starts working)
    Phase 2: Find Pareto frontier for each layer (refusal vs KL tradeoff)
    """

    def __init__(
        self,
        scorer,  # MultiObjectiveScorer with single-layer ablation support
        hidden_dim: int,
        n_layers: int,
        config: Optional[SingleLayerDiscoveryConfig] = None,
        r_init: Optional[Dict[int, torch.Tensor]] = None,  # Initial directions per layer
    ):
        self.scorer = scorer
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.config = config or SingleLayerDiscoveryConfig()
        self.r_init = r_init  # Difference-in-means directions

        # Determine device from r_init or default to cpu
        if r_init is not None and len(r_init) > 0:
            first_key = next(iter(r_init.keys()))
            self.device = r_init[first_key].device
        else:
            self.device = torch.device('cpu')

        # Determine layers to search
        self.layers = list(range(
            self.config.layer_start,
            min(self.config.layer_end, n_layers)
        ))

        # GPs for each objective
        self.refusal_gp = DirectionLayerGP(
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            direction_lengthscale=self.config.direction_lengthscale,
            layer_lengthscale=self.config.layer_lengthscale,
            noise_var=self.config.noise_var,
        )
        self.kl_gp = DirectionLayerGP(
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            direction_lengthscale=self.config.direction_lengthscale,
            layer_lengthscale=self.config.layer_lengthscale,
            noise_var=self.config.noise_var,
        )

        # All observations: (direction, layer, scores)
        self.observations: List[Tuple[torch.Tensor, int, Dict[str, float]]] = []

    def _score(self, d: torch.Tensor, layer: int) -> Dict[str, float]:
        """Score a direction at a specific layer."""
        # Create a per-layer vector (zeros except at target layer)
        # Match device and dtype of input direction
        v = torch.zeros(self.n_layers, self.hidden_dim, device=d.device, dtype=d.dtype)
        v[layer] = d

        scores = self.scorer.score(v)
        return scores

    def _add_observation(self, d: torch.Tensor, layer: int, scores: Dict[str, float]):
        """Add observation to GPs and storage."""
        self.observations.append((d.detach().cpu(), layer, scores))
        self.refusal_gp.add_observation(d, layer, scores['refusal_score'])
        self.kl_gp.add_observation(d, layer, scores.get('kl_score', 0.0))

    def _generate_candidates(
        self,
        layer: int,
        n_candidates: int = 50,
        near_boundary: bool = False,
    ) -> List[torch.Tensor]:
        """Generate candidate directions for a specific layer."""
        candidates = []

        # Strategy 1: Perturbations of r_init
        if self.r_init is not None and layer in self.r_init:
            r = self.r_init[layer]
            for _ in range(n_candidates // 3):
                d = r + 0.3 * torch.randn_like(r)
                d = d / (d.norm() + 1e-8)
                candidates.append(d)

        # Strategy 2: Random directions on sphere
        for _ in range(n_candidates // 3):
            d = torch.randn(self.hidden_dim, device=self.device)
            d = d / (d.norm() + 1e-8)
            candidates.append(d)

        # Strategy 3: Perturbations of best observed at this layer
        layer_obs = [(d, s) for d, l, s in self.observations if l == layer]
        if layer_obs:
            # Sort by refusal score
            layer_obs.sort(key=lambda x: x[1]['refusal_score'])
            best_d = layer_obs[0][0]
            for _ in range(n_candidates - len(candidates)):
                d = best_d + 0.2 * torch.randn_like(best_d)
                d = d / (d.norm() + 1e-8)
                candidates.append(d)
        else:
            # Fill with random
            while len(candidates) < n_candidates:
                d = torch.randn(self.hidden_dim, device=self.device)
                d = d / (d.norm() + 1e-8)
                candidates.append(d)

        return candidates

    def _boundary_phase(self) -> Dict[int, Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]]:
        """
        Phase 1: Find boundary for each layer.

        Returns:
            Dict mapping layer -> (boundary_points, inside_points, outside_points)
        """
        threshold = self.config.boundary_threshold

        print(f"Phase 1: Boundary Discovery")
        print(f"  Layers: {self.layers[0]} to {self.layers[-1]}")
        print(f"  Threshold: {threshold:.3f}")

        # Initial exploration: sample across layers
        print(f"  Initial exploration ({self.config.n_boundary_init} samples)...")
        for i in range(self.config.n_boundary_init):
            # Pick a random layer
            layer = self.layers[i % len(self.layers)]

            # Generate direction
            if self.r_init is not None and layer in self.r_init and i < self.config.n_boundary_init // 2:
                # Perturbation of r_init
                d = self.r_init[layer] + 0.3 * torch.randn_like(self.r_init[layer])
            else:
                # Random
                d = torch.randn(self.hidden_dim, device=self.device)
            d = d / (d.norm() + 1e-8)

            scores = self._score(d, layer)
            self._add_observation(d, layer, scores)

            if i % 10 == 0:
                print(f"    Sample {i+1}: layer={layer}, refusal={scores['refusal_score']:.4f}")

        # Fit GPs
        self.refusal_gp.fit()
        self.kl_gp.fit()

        # Boundary iterations using straddle acquisition
        print(f"  Straddle search ({self.config.n_boundary_iterations} iterations)...")
        for iteration in range(self.config.n_boundary_iterations):
            if len(self.observations) >= self.config.max_measurements // 2:
                break

            # Generate candidates for each layer and select best globally
            best_acquisition = -float('inf')
            best_d = None
            best_layer = None

            for layer in self.layers:
                candidates = self._generate_candidates(layer, n_candidates=30)

                for d in candidates:
                    mu, sigma = self.refusal_gp.predict(d, layer)
                    # Straddle acquisition: β*σ - |μ - threshold|
                    acq = self.config.boundary_beta * sigma - abs(mu - threshold)

                    if acq > best_acquisition:
                        best_acquisition = acq
                        best_d = d
                        best_layer = layer

            # Evaluate best candidate
            scores = self._score(best_d, best_layer)
            self._add_observation(best_d, best_layer, scores)

            # Update GP
            self.refusal_gp.fit()

            if iteration % 10 == 0:
                print(f"    Iter {iteration}: layer={best_layer}, "
                      f"refusal={scores['refusal_score']:.4f}, acq={best_acquisition:.4f}")

        # Classify points per layer
        return self._classify_points_per_layer()

    def _classify_points_per_layer(
        self
    ) -> Dict[int, Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]]:
        """Classify observations per layer as inside/outside/boundary."""
        threshold = self.config.boundary_threshold
        boundary_margin = 1.0

        results = {}
        for layer in self.layers:
            inside = []
            outside = []
            boundary = []

            for d, l, scores in self.observations:
                if l != layer:
                    continue

                r = scores['refusal_score']
                if r < threshold - boundary_margin:
                    inside.append(d)
                elif r > threshold + boundary_margin:
                    outside.append(d)
                else:
                    boundary.append(d)

            results[layer] = (boundary, inside, outside)

        return results

    def _pareto_phase(
        self,
        boundary_results: Dict[int, Tuple[List, List, List]]
    ) -> Dict[int, Tuple[List[torch.Tensor], Dict[str, List[float]], float]]:
        """
        Phase 2: Find Pareto frontier for each layer.

        Returns:
            Dict mapping layer -> (pareto_directions, pareto_scores, hypervolume)
        """
        print(f"\nPhase 2: Pareto Discovery (per layer)")

        results = {}

        for layer in self.layers:
            boundary, inside, outside = boundary_results[layer]

            print(f"\n  Layer {layer}: {len(inside)} inside, {len(boundary)} boundary, {len(outside)} outside")

            if len(inside) == 0:
                print(f"    No inside points, skipping Pareto search")
                results[layer] = ([], {'refusal_score': [], 'kl_score': []}, 0.0)
                continue

            # Pareto search for this layer
            layer_obs = [(d, s) for d, l, s in self.observations if l == layer]

            # Find Pareto frontier from existing observations
            pareto_directions, pareto_scores = self._compute_pareto_frontier(layer_obs)
            hypervolume = self._compute_hypervolume(pareto_scores)

            print(f"    Pareto points: {len(pareto_directions)}, HV: {hypervolume:.4f}")

            results[layer] = (pareto_directions, pareto_scores, hypervolume)

        return results

    def _compute_pareto_frontier(
        self,
        observations: List[Tuple[torch.Tensor, Dict[str, float]]]
    ) -> Tuple[List[torch.Tensor], Dict[str, List[float]]]:
        """Extract Pareto-optimal points from observations."""
        if not observations:
            return [], {'refusal_score': [], 'kl_score': []}

        # Extract scores
        points = [(d, s['refusal_score'], s.get('kl_score', 0.0)) for d, s in observations]

        # Find Pareto frontier (minimize both refusal and KL)
        pareto = []
        for i, (d_i, r_i, k_i) in enumerate(points):
            dominated = False
            for j, (d_j, r_j, k_j) in enumerate(points):
                if i != j:
                    # j dominates i if j is better in both objectives
                    if r_j <= r_i and k_j <= k_i and (r_j < r_i or k_j < k_i):
                        dominated = True
                        break
            if not dominated:
                pareto.append((d_i, r_i, k_i))

        directions = [p[0] for p in pareto]
        scores = {
            'refusal_score': [p[1] for p in pareto],
            'kl_score': [p[2] for p in pareto],
        }

        return directions, scores

    def _compute_hypervolume(
        self,
        pareto_scores: Dict[str, List[float]],
        ref_point: Tuple[float, float] = (10.0, 10.0)
    ) -> float:
        """Compute hypervolume indicator for Pareto frontier."""
        if not pareto_scores['refusal_score']:
            return 0.0

        points = list(zip(pareto_scores['refusal_score'], pareto_scores['kl_score']))

        # Simple 2D hypervolume computation
        # Sort by first objective
        points.sort(key=lambda p: p[0])

        hv = 0.0
        prev_k = ref_point[1]

        for r, k in points:
            if r < ref_point[0] and k < ref_point[1]:
                width = ref_point[0] - r
                height = prev_k - k
                if height > 0:
                    hv += width * height
                    prev_k = k

        return hv

    def discover(self) -> SingleLayerDiscoveryResults:
        """Run full discovery pipeline."""

        print("=" * 70)
        print("Single-Layer Discovery: (direction, layer) optimization")
        print("=" * 70)

        # Compute dynamic threshold if needed
        if self.config.use_dynamic_threshold:
            baseline_refusal, baseline_harmless = self.scorer.compute_baseline_refusal_scores()
            gap = baseline_refusal - baseline_harmless
            self.config.boundary_threshold = baseline_refusal - self.config.boundary_fraction * gap
            print(f"Dynamic threshold: {baseline_refusal:.3f} - {self.config.boundary_fraction:.0%}*{gap:.3f} "
                  f"= {self.config.boundary_threshold:.3f}")

        # Phase 1: Boundary discovery
        boundary_results = self._boundary_phase()

        # Phase 2: Pareto discovery per layer
        pareto_results = self._pareto_phase(boundary_results)

        # Compile per-layer results
        per_layer = {}
        for layer in self.layers:
            boundary, inside, outside = boundary_results[layer]
            pareto_dirs, pareto_scores, hv = pareto_results[layer]

            # Find best scores for this layer
            layer_obs = [(s['refusal_score'], s.get('kl_score', 0.0))
                        for d, l, s in self.observations if l == layer]

            best_refusal = min([r for r, k in layer_obs]) if layer_obs else float('inf')
            best_kl = min([k for r, k in layer_obs]) if layer_obs else float('inf')

            per_layer[layer] = PerLayerResults(
                layer=layer,
                boundary_points=boundary,
                inside_points=inside,
                pareto_points=pareto_dirs,
                pareto_scores=pareto_scores,
                best_refusal_score=best_refusal,
                best_kl_score=best_kl,
            )

        # Find best layer (by hypervolume)
        hypervolumes = {l: pareto_results[l][2] for l in self.layers}
        best_layer = max(hypervolumes.keys(), key=lambda l: hypervolumes[l])

        # Find overall best direction
        best_obs = min(self.observations, key=lambda x: x[2]['refusal_score'])

        print("\n" + "=" * 70)
        print("Results Summary")
        print("=" * 70)
        print(f"Total measurements: {len(self.observations)}")
        print(f"Best layer (by HV): {best_layer} (HV={hypervolumes[best_layer]:.4f})")
        print(f"Best refusal score: {best_obs[2]['refusal_score']:.4f} at layer {best_obs[1]}")

        print("\nPer-layer summary:")
        for layer in self.layers:
            pr = per_layer[layer]
            hv = hypervolumes[layer]
            print(f"  Layer {layer}: inside={len(pr.inside_points)}, "
                  f"pareto={len(pr.pareto_points)}, HV={hv:.4f}, "
                  f"best_r={pr.best_refusal_score:.4f}")

        return SingleLayerDiscoveryResults(
            per_layer=per_layer,
            best_layer=best_layer,
            best_direction=best_obs[0],
            best_scores=best_obs[2],
            all_observations=self.observations,
            hypervolume_per_layer=hypervolumes,
        )
