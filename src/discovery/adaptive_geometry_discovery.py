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
    n_init_random: int = 10  # Initial random samples
    n_iterations: int = 30  # Active exploration iterations
    n_candidates: int = 100  # Candidates per iteration (keep low to avoid OOM)

    # Acquisition
    acquisition_type: str = 'ucb'  # 'ucb', 'ei', or 'boundary'
    beta: float = 2.0  # UCB exploration parameter

    # GP type: 'sparse', 'structured', or 'simple'
    gp_type: str = 'structured'  # 'structured' models layer dependencies

    # GP kernel settings (for 'simple' and 'sparse')
    kernel_type: str = 'rbf'  # 'rbf', 'linear', or 'rbf+linear'
    kernel_lengthscale: float = 1.0

    # Sparse GP settings (gp_type='sparse')
    use_sparse_gp: bool = True  # Legacy flag; only used if gp_type is unset/invalid
    num_inducing: int = 64
    sparse_train_steps: int = 15
    sparse_lr: float = 0.05
    sparse_max_train_points: int = 256
    sparse_jitter: float = 1e-5

    # Structured GP settings (gp_type='structured')
    # Models layer smoothness + learns layer importance via ARD
    layer_lengthscale: float = 3.0  # Smoothness across ~3 adjacent layers
    feature_lengthscale: float = 1.0  # RBF lengthscale for features
    init_layer_weights: str = 'middle'  # 'middle', 'uniform', or 'learned'
    learn_layer_weights: bool = True  # Learn which layers matter via ARD
    structured_train_steps: int = 20

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
        self.noise_var = 0.1  # Higher noise for numerical stability

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

        # Add noise to diagonal with increasing jitter for numerical stability
        jitter = self.noise_var
        K_train_train_noisy = K_train_train + jitter * torch.eye(len(K_train_train), device=K_train_train.device)

        # Solve for weights with fallback jitter
        for attempt in range(5):
            try:
                L = torch.linalg.cholesky(K_train_train_noisy)
                break
            except torch._C._LinAlgError:
                jitter *= 10
                K_train_train_noisy = K_train_train + jitter * torch.eye(len(K_train_train), device=K_train_train.device)
        else:
            # Last resort: use pseudoinverse instead of cholesky
            mean = K_test_train @ torch.linalg.lstsq(K_train_train_noisy, self.R_train.unsqueeze(-1)).solution.squeeze(-1)
            std = torch.ones(len(V_test), device=V_test.device) * 0.5
            return mean, std
        alpha = torch.cholesky_solve(self.R_train.unsqueeze(-1), L).squeeze(-1)

        # Predictive mean
        mean = K_test_train @ alpha

        # Predictive variance
        v = torch.linalg.solve_triangular(L, K_test_train.T, upper=False)
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


class AdaptiveSparseGP:
    """
    Variationally updated sparse GP with learnable inducing points.

    Uses a low-rank Nyström feature map with a ridge-regression objective
    to adapt inducing locations toward informative regions. This keeps
    uncertainty estimates while scaling beyond dense GP costs.
    """

    def __init__(
        self,
        kernel_type='rbf',
        lengthscale: float = 1.0,
        noise_var: float = 0.01,
        num_inducing: int = 64,
        train_steps: int = 15,
        lr: float = 0.05,
        max_train_points: int = 256,
        jitter: float = 1e-5
    ):
        self.kernel_type = kernel_type
        self.noise_var = noise_var
        self.train_steps = train_steps
        self.lr = lr
        self.max_train_points = max_train_points
        self.jitter = jitter
        self.num_inducing = num_inducing

        # Learnable parameters
        self.log_lengthscale = torch.tensor(float(lengthscale)).log().requires_grad_()
        self.inducing_points = None

        # Cached fit state
        self.w = None
        self.A_inv = None
        self.L_mm = None

    @property
    def lengthscale(self):
        return torch.nn.functional.softplus(self.log_lengthscale) + 1e-6

    def _kernel(self, V1: torch.Tensor, V2: torch.Tensor) -> torch.Tensor:
        if self.kernel_type == 'rbf':
            # RBF kernel on flattened vectors
            dots = V1 @ V2.T
            dots = torch.clamp(dots, -1, 1)
            dist_sq = 2 * (1 - dots)  # chord distance proxy
            K = torch.exp(-dist_sq / (2 * self.lengthscale ** 2))
        elif self.kernel_type == 'linear':
            K = V1 @ V2.T
        else:
            raise ValueError(f"Unknown kernel: {self.kernel_type}")
        return K

    def _ensure_inducing(self, V: torch.Tensor, m: int):
        if self.inducing_points is None or self.inducing_points.shape[0] != m:
            # Initialize or resize with random subset (or all if small)
            idx = torch.randperm(len(V))[:m]
            init = V[idx].detach().clone()
            self.inducing_points = torch.nn.Parameter(init)

    def _stable_cholesky(self, K_base: torch.Tensor) -> torch.Tensor:
        """Compute a stable Cholesky factor with jitter/eigen clamping."""
        device = K_base.device
        L = None
        for scale in (1.0, 10.0, 1e2, 1e3):
            K = K_base + (scale * self.jitter) * torch.eye(K_base.size(0), device=device)
            try:
                L = torch.linalg.cholesky(K)
                break
            except torch.linalg.LinAlgError:
                L = None
        if L is None:
            # Last resort: symmetrize and clamp eigenvalues
            K_sym = 0.5 * (K_base + K_base.T)
            evals, evecs = torch.linalg.eigh(K_sym)
            evals_clamped = torch.clamp(evals, min=self.jitter)
            K = (evecs @ torch.diag(evals_clamped) @ evecs.T)
            L = torch.linalg.cholesky(K)
        return L

    def fit(self, V: torch.Tensor, R: torch.Tensor):
        """
        Fit sparse GP with adaptive inducing points.

        Args:
            V: Observed directions [n, d]
            R: Observed refusal strengths [n]
        """
        if V.dim() != 2:
            raise ValueError("V must be [n, d] flattened directions")

        device = V.device
        V_train = V[-self.max_train_points:].to(device)
        R_train = R[-self.max_train_points:].to(device)

        m = min(self.num_inducing, len(V_train))
        self._ensure_inducing(V_train, m)

        params = [self.inducing_points, self.log_lengthscale]
        optimizer = torch.optim.Adam(params, lr=self.lr)

        for _ in range(self.train_steps):
            optimizer.zero_grad()

            Z = self.inducing_points
            K_mm_base = self._kernel(Z, Z)
            L_mm = self._stable_cholesky(K_mm_base)

            K_nm = self._kernel(V_train, Z)  # [n, m]
            # Φ = K_nm K_mm^{-1/2}
            tmp = torch.linalg.solve_triangular(L_mm, K_nm.T, upper=False)  # [m, n]
            phi = tmp.T  # [n, m]

            lambda_ = self.noise_var
            A = phi.T @ phi + lambda_ * torch.eye(len(Z), device=device)
            A = A + self.jitter * torch.eye(len(Z), device=device)
            L_A = torch.linalg.cholesky(A)

            rhs = phi.T @ R_train
            w = torch.cholesky_solve(rhs.unsqueeze(-1), L_A).squeeze(-1)

            pred = phi @ w
            residual = R_train - pred

            data_loss = (residual ** 2).mean()
            reg_loss = lambda_ * (w ** 2).mean()
            logdet = 2 * torch.log(torch.diag(L_A) + 1e-12).sum() / len(R_train)

            loss = data_loss + reg_loss + 1e-3 * logdet
            loss.backward()
            optimizer.step()

        # Cache final state for prediction
        with torch.no_grad():
            Z = self.inducing_points
            K_mm_base = self._kernel(Z, Z)
            self.L_mm = self._stable_cholesky(K_mm_base)

            K_nm = self._kernel(V_train, Z)
            tmp = torch.linalg.solve_triangular(self.L_mm, K_nm.T, upper=False)
            phi = tmp.T

            A = phi.T @ phi + self.noise_var * torch.eye(len(Z), device=device)
            A = A + self.jitter * torch.eye(len(Z), device=device)
            L_A = torch.linalg.cholesky(A)
            self.A_inv = torch.cholesky_inverse(L_A)

            rhs = phi.T @ R_train
            self.w = torch.cholesky_solve(rhs.unsqueeze(-1), L_A).squeeze(-1)

    def predict(self, V_test: torch.Tensor) -> tuple:
        """
        Predict mean and std for test directions.

        Args:
            V_test: [m, d] flattened directions
        """
        if self.inducing_points is None or self.w is None:
            mean = torch.zeros(len(V_test), device=V_test.device)
            std = torch.ones(len(V_test), device=V_test.device)
            return mean, std

        device = self.inducing_points.device
        Vt = V_test.to(device)

        K_sm = self._kernel(Vt, self.inducing_points)  # [m, M]
        # Φ_* = K_sm K_mm^{-1/2}
        tmp = torch.linalg.solve_triangular(self.L_mm, K_sm.T, upper=False)
        phi_star = tmp.T  # [m, M]

        mean = phi_star @ self.w

        # Predictive variance: λ + φ_* A^{-1} φ_*^T (diagonal only)
        A_inv_phiT = phi_star @ self.A_inv  # [m, M]
        var = self.noise_var + (A_inv_phiT * phi_star).sum(dim=1)
        var = torch.clamp(var, min=self.jitter)
        std = torch.sqrt(var)

        return mean, std


class StructuredLayerGP:
    """
    GP with structure-aware kernel for [n_layers, hidden_dim] directions.

    Implements:
    1. Layer smoothness: Adjacent layers have correlated directions
    2. ARD layer weights: Learns which layers matter (middle layers typically)
    3. Feature kernel: RBF on hidden_dim features

    The kernel factorizes as:
        K(v, v') = K_layer(i, j) * K_feature(v[i], v'[j])

    Where K_layer encodes layer smoothness and importance.
    """

    def __init__(
        self,
        n_layers: int,
        hidden_dim: int,
        feature_lengthscale: float = 1.0,
        layer_lengthscale: float = 3.0,  # Smoothness across ~3 layers
        noise_var: float = 0.01,
        learn_layer_weights: bool = True,
        init_layer_weights: str = 'middle',  # 'middle', 'uniform', or 'learned'
        train_steps: int = 20,
        lr: float = 0.05,
    ):
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.noise_var = noise_var
        self.learn_layer_weights = learn_layer_weights
        self.train_steps = train_steps
        self.lr = lr

        # Learnable parameters
        self.log_feature_lengthscale = torch.tensor(feature_lengthscale).log()
        self.log_layer_lengthscale = torch.tensor(layer_lengthscale).log()

        # Layer importance weights (ARD-style)
        # Initialize based on strategy
        if init_layer_weights == 'middle':
            # Bell curve centered on middle layers
            layer_idx = torch.arange(n_layers, dtype=torch.float32)
            middle = (n_layers - 1) / 2
            weights = torch.exp(-0.5 * ((layer_idx - middle) / (n_layers / 4)) ** 2)
            weights = weights / weights.sum() * n_layers  # Normalize to sum=n_layers
        elif init_layer_weights == 'uniform':
            weights = torch.ones(n_layers)
        else:  # 'learned' - start uniform
            weights = torch.ones(n_layers)

        self.log_layer_weights = weights.log()

        if learn_layer_weights:
            self.log_layer_weights.requires_grad_(True)
            self.log_feature_lengthscale.requires_grad_(True)
            self.log_layer_lengthscale.requires_grad_(True)

        # Training data
        self.V_train = None  # [n, n_layers, hidden_dim]
        self.R_train = None  # [n]

        # Cached for prediction
        self.L = None
        self.alpha = None

    @property
    def feature_lengthscale(self):
        return self.log_feature_lengthscale.exp()

    @property
    def layer_lengthscale(self):
        return self.log_layer_lengthscale.exp()

    @property
    def layer_weights(self):
        # Softmax to ensure positive and normalized
        return torch.softmax(self.log_layer_weights, dim=0) * self.n_layers

    def _layer_kernel(self) -> torch.Tensor:
        """
        Compute layer-layer kernel matrix [n_layers, n_layers].

        Encodes: adjacent layers are correlated, weighted by importance.
        """
        idx = torch.arange(self.n_layers, dtype=torch.float32)
        # Layer distance matrix
        layer_dist_sq = (idx.unsqueeze(0) - idx.unsqueeze(1)) ** 2
        # RBF for smoothness
        K_smooth = torch.exp(-layer_dist_sq / (2 * self.layer_lengthscale ** 2))
        # Weight by layer importance (outer product)
        w = self.layer_weights.sqrt()
        K_weighted = K_smooth * w.unsqueeze(0) * w.unsqueeze(1)
        return K_weighted

    def _feature_kernel(self, V1: torch.Tensor, V2: torch.Tensor, layer: int) -> torch.Tensor:
        """
        Compute feature kernel for a single layer.

        Args:
            V1: [n1, hidden_dim] features at layer for first set
            V2: [n2, hidden_dim] features at layer for second set

        Returns:
            K: [n1, n2] kernel matrix
        """
        # Normalize to unit sphere
        V1_norm = V1 / (V1.norm(dim=1, keepdim=True) + 1e-8)
        V2_norm = V2 / (V2.norm(dim=1, keepdim=True) + 1e-8)

        # Cosine similarity -> geodesic distance
        dots = V1_norm @ V2_norm.T
        dots = torch.clamp(dots, -1, 1)
        dist_sq = 2 * (1 - dots)

        K = torch.exp(-dist_sq / (2 * self.feature_lengthscale ** 2))
        return K

    def _full_kernel(self, V1: torch.Tensor, V2: torch.Tensor) -> torch.Tensor:
        """
        Compute full structured kernel.

        Args:
            V1: [n1, n_layers, hidden_dim]
            V2: [n2, n_layers, hidden_dim]

        Returns:
            K: [n1, n2] kernel matrix
        """
        n1, n2 = len(V1), len(V2)
        K_layer = self._layer_kernel()  # [n_layers, n_layers]

        # Sum over layers with layer kernel weighting
        K = torch.zeros(n1, n2)
        for i in range(self.n_layers):
            for j in range(self.n_layers):
                if K_layer[i, j] > 0.01:  # Skip negligible contributions
                    K_feat = self._feature_kernel(V1[:, i, :], V2[:, j, :], i)
                    K = K + K_layer[i, j] * K_feat

        # Normalize
        K = K / self.n_layers
        return K

    def fit(self, V: torch.Tensor, R: torch.Tensor):
        """
        Fit GP to observations.

        Args:
            V: [n, n_layers, hidden_dim] observed directions
            R: [n] observed refusal strengths
        """
        # Handle flattened input (reshape if needed)
        if V.dim() == 2:
            V = V.reshape(-1, self.n_layers, self.hidden_dim)

        # Ensure tensors are float and on correct device (don't need grad for data)
        device = V.device
        self.V_train = V.float().detach().cpu()  # Keep on CPU for GP
        self.R_train = R.float().detach().cpu()

        # Move learnable parameters to CPU (where computation happens)
        if self.log_layer_weights.device != torch.device('cpu'):
            self.log_layer_weights = self.log_layer_weights.cpu()
            self.log_feature_lengthscale = self.log_feature_lengthscale.cpu()
            self.log_layer_lengthscale = self.log_layer_lengthscale.cpu()
            if self.learn_layer_weights:
                self.log_layer_weights.requires_grad_(True)
                self.log_feature_lengthscale.requires_grad_(True)
                self.log_layer_lengthscale.requires_grad_(True)

        if self.learn_layer_weights and len(V) > 5:
            self._optimize_hyperparameters()

        self._update_cache()

    def _optimize_hyperparameters(self):
        """Optimize kernel hyperparameters via marginal likelihood."""
        # Ensure params have grad enabled
        if not self.log_layer_weights.requires_grad:
            self.log_layer_weights.requires_grad_(True)
            self.log_feature_lengthscale.requires_grad_(True)
            self.log_layer_lengthscale.requires_grad_(True)

        params = [self.log_layer_weights, self.log_feature_lengthscale, self.log_layer_lengthscale]
        optimizer = torch.optim.Adam(params, lr=self.lr)

        for _ in range(self.train_steps):
            optimizer.zero_grad()

            K = self._full_kernel(self.V_train, self.V_train)
            K = K + self.noise_var * torch.eye(len(K))
            K = K + 1e-5 * torch.eye(len(K))  # Jitter

            try:
                L = torch.linalg.cholesky(K)
            except torch.linalg.LinAlgError:
                continue

            # Log marginal likelihood
            alpha = torch.cholesky_solve(self.R_train.unsqueeze(-1), L).squeeze(-1)
            data_fit = -0.5 * (self.R_train @ alpha)
            complexity = -torch.diag(L).log().sum()
            lml = data_fit + complexity

            try:
                (-lml).backward()  # Maximize LML
                optimizer.step()
            except RuntimeError as e:
                # Gradient computation failed - skip optimization
                if "does not require grad" in str(e):
                    return
                raise

    def _update_cache(self):
        """Update cached Cholesky and alpha for prediction."""
        K = self._full_kernel(self.V_train, self.V_train)
        K = K + self.noise_var * torch.eye(len(K))
        K = K + 1e-5 * torch.eye(len(K))

        self.L = torch.linalg.cholesky(K)
        self.alpha = torch.cholesky_solve(self.R_train.unsqueeze(-1), self.L).squeeze(-1)

    def predict(self, V_test: torch.Tensor) -> tuple:
        """
        Predict R(v) for test directions.

        Args:
            V_test: [m, n_layers * hidden_dim] or [m, n_layers, hidden_dim]

        Returns:
            mean: [m] predicted values
            std: [m] standard deviations
        """
        # Handle flattened input
        if V_test.dim() == 2 and V_test.shape[1] == self.n_layers * self.hidden_dim:
            V_test = V_test.reshape(-1, self.n_layers, self.hidden_dim)

        if self.V_train is None:
            return torch.zeros(len(V_test)), torch.ones(len(V_test))

        K_test_train = self._full_kernel(V_test, self.V_train)
        K_test_test = self._full_kernel(V_test, V_test)

        # Predictive mean
        mean = K_test_train @ self.alpha

        # Predictive variance
        v = torch.linalg.solve_triangular(self.L, K_test_train.T, upper=False)
        var = torch.diag(K_test_test) - (v ** 2).sum(dim=0)
        var = torch.clamp(var, min=1e-6)
        std = torch.sqrt(var)

        return mean, std

    def get_layer_importance(self) -> torch.Tensor:
        """Return learned layer importance weights."""
        return self.layer_weights.detach()


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

        # GP model selection
        gp_type = self._resolve_gp_type()
        if gp_type == 'structured':
            self.gp = StructuredLayerGP(
                n_layers=n_layers,
                hidden_dim=hidden_dim,
                feature_lengthscale=self.config.feature_lengthscale,
                layer_lengthscale=self.config.layer_lengthscale,
                learn_layer_weights=self.config.learn_layer_weights,
                init_layer_weights=self.config.init_layer_weights,
                train_steps=self.config.structured_train_steps,
            )
        elif gp_type == 'sparse':
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

    def _resolve_gp_type(self) -> str:
        """Resolve GP backend from explicit gp_type with legacy fallback."""
        gp_type = (self.config.gp_type or '').strip().lower()
        if gp_type in ('structured', 'sparse', 'simple'):
            return gp_type
        return 'sparse' if self.config.use_sparse_gp else 'simple'

    def _estimate_memory_usage(self) -> Dict:
        """
        Estimate memory usage for the discovery run.

        Returns:
            Dict with memory estimates in MB
        """
        d = self.n_layers * self.hidden_dim  # Flattened dimension
        n_total = self.config.n_init_random + self.config.n_iterations
        m = self.config.n_candidates

        gp_type = self._resolve_gp_type()

        # Effective matrix size depends on GP type
        if gp_type == 'structured':
            M = n_total  # Dense but with layer structure
        elif gp_type == 'sparse':
            M = self.config.num_inducing
        else:
            M = n_total

        bytes_per_float = 4  # float32

        # Candidate storage per iteration
        candidates_mb = (m * d * bytes_per_float) / (1024 ** 2)

        # GP kernel matrices
        if gp_type == 'sparse':
            # Sparse: K_nm [n, M], K_mm [M, M], K_sm [m, M]
            gp_train_mb = (n_total * M + M * M) * bytes_per_float / (1024 ** 2)
            gp_predict_mb = (m * M) * bytes_per_float / (1024 ** 2)
        else:
            # Dense: K_train [n, n], K_test_train [m, n], K_test_test [m, m]
            gp_train_mb = (n_total ** 2) * bytes_per_float / (1024 ** 2)
            gp_predict_mb = (m * n_total + m * m) * bytes_per_float / (1024 ** 2)

        # Observations storage
        obs_mb = (n_total * d * bytes_per_float) / (1024 ** 2)

        peak_mb = candidates_mb + gp_predict_mb + obs_mb

        return {
            'candidates_mb': candidates_mb,
            'gp_train_mb': gp_train_mb,
            'gp_predict_mb': gp_predict_mb,
            'observations_mb': obs_mb,
            'peak_estimate_mb': peak_mb,
            'dimension': d,
            'n_observations': n_total,
        }

    def discover(self) -> Dict:
        """
        Run adaptive geometry discovery.

        Returns:
            results: Dictionary with discovered geometry
        """
        print("=" * 70)
        print("Adaptive Geometry Discovery")
        print("=" * 70)

        # Memory estimation and warning
        mem = self._estimate_memory_usage()
        gp_type = self._resolve_gp_type()
        gp_type_str = gp_type.capitalize()
        if gp_type == 'structured':
            gp_type_str += f" (layer smoothness σ={self.config.layer_lengthscale}, ARD={self.config.learn_layer_weights})"

        print(f"\nMemory estimate (peak ~{mem['peak_estimate_mb']:.1f} MB):")
        print(f"  Candidates/iter: {mem['candidates_mb']:.1f} MB ({self.config.n_candidates} × {mem['dimension']} floats)")
        print(f"  GP predict:      {mem['gp_predict_mb']:.1f} MB")
        print(f"  Observations:    {mem['observations_mb']:.1f} MB")
        print(f"  GP type:         {gp_type_str}")

        if mem['peak_estimate_mb'] > 1000:
            print(f"\n  WARNING: Estimated memory > 1GB. Consider reducing n_candidates.")
        if gp_type == 'simple' and mem['gp_predict_mb'] > 500:
            print(f"\n  WARNING: Dense GP will be slow. Set gp_type='sparse' or 'structured'.")

        # Phase 1: Random initialization
        print(f"\nPhase 1: Random initialization ({self.config.n_init_random} samples)")
        self._initialize_random()

        # Phase 2: Active exploration
        print(f"\nPhase 2: Active exploration ({self.config.n_iterations} iterations)")
        self._active_exploration()

        # Report learned layer importance (if using structured GP)
        if gp_type == 'structured' and hasattr(self.gp, 'get_layer_importance'):
            layer_weights = self.gp.get_layer_importance()
            print("\n  Learned layer importance (ARD):")
            top_k = min(5, len(layer_weights))
            top_layers = layer_weights.topk(top_k)
            for idx, weight in zip(top_layers.indices, top_layers.values):
                print(f"    Layer {idx.item():2d}: {weight.item():.3f}")

        # Phase 3: Geometry extraction
        print("\nPhase 3: Geometry extraction")
        geometry = self._extract_geometry()

        # Phase 4: Recommend representation
        print("\nPhase 4: Representation recommendation")
        recommendation = self.recommend_representation(geometry)
        print(f"  Use cone: {recommendation.get('use_cone', 'unknown')}")
        print(f"  Reasoning: {recommendation.get('reasoning', 'N/A')}")
        if recommendation.get('use_cone'):
            print(f"  Cone rank: {recommendation.get('cone_rank', 'N/A')}")
            if 'efficiency_gain' in recommendation:
                print(f"  Efficiency: {recommendation['efficiency_gain']}")
        else:
            print(f"  Alternative: {recommendation.get('alternative', 'N/A')}")

        # Package results
        results = {
            'V_observed': torch.stack(self.V_observed),
            'R_observed': torch.tensor(self.R_observed),
            'gp': self.gp,
            'geometry': geometry,
            'recommendation': recommendation
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
        V_tensor = torch.stack(self.V_observed).reshape(len(self.V_observed), -1)
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
            V_tensor = torch.stack(self.V_observed).reshape(len(self.V_observed), -1)
            R_tensor = torch.tensor(self.R_observed)
            self.gp.fit(V_tensor, R_tensor)

            print(f"  Iteration {iteration+1}: R = {R_next:.3f}, Acq = {scores[best_idx]:.3f}")

            # Check convergence
            if scores.max() < self.config.convergence_threshold:
                print("  Converged!")
                break

    def _generate_candidates(self, n: int) -> torch.Tensor:
        """
        Generate candidate directions to evaluate.

        Samples from FULL hypersphere S^(d-1) where d = n_layers × hidden_dim.

        NOTE: This is different from cone sampling!
        - Cones: Sample from S^(k-1) within k-dimensional subspace
        - Adaptive: Sample from S^(d-1) = entire hypersphere

        Returns:
            candidates: [n, n_layers, hidden_dim] unit vectors
        """

        # Random sampling on FULL hypersphere S^(d-1)
        # NOT restricted to any subspace!
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
            # Boundary exploration using straddle heuristic
            # High scores when: (1) high uncertainty AND (2) near threshold crossing
            distance_to_threshold = torch.abs(mean - self.config.boundary_threshold)
            scores = self.config.beta * std - distance_to_threshold

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

        principal_dirs = [
            V_observed[idx]
            for idx, value in zip(topk_indices, topk_values)
            if value > self.config.boundary_threshold
        ]

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

    def recommend_representation(self, geometry: Dict) -> Dict:
        """
        Recommend whether to use cones or GP/field based on discovered geometry.

        Args:
            geometry: Discovered geometry from _extract_geometry()

        Returns:
            recommendation: Dict with:
                - use_cone: bool
                - reasoning: str
                - cone_rank: int (if use_cone)
                - alternative: str (if not use_cone)
        """
        recommendation = {}

        intrinsic_dim = geometry.get('intrinsic_dimension', None)
        n_modes = geometry.get('n_modes', 0)

        # Decision criteria
        LOW_DIM_THRESHOLD = 10
        SIMPLE_THRESHOLD = 3  # For "very simple" geometry

        if intrinsic_dim is None:
            recommendation['use_cone'] = False
            recommendation['reasoning'] = "Could not estimate intrinsic dimension (insufficient data)"
            recommendation['alternative'] = "Collect more data or use GP with uncertainty"
            return recommendation

        # Check if geometry is low-dimensional enough for cones
        if intrinsic_dim <= SIMPLE_THRESHOLD:
            # Very simple - definitely use cone
            recommendation['use_cone'] = True
            recommendation['cone_rank'] = intrinsic_dim
            recommendation['reasoning'] = (
                f"Intrinsic dimension ({intrinsic_dim}) is very low. "
                f"Geometry is simple enough for efficient cone representation."
            )
            recommendation['efficiency_gain'] = f"{intrinsic_dim}/{self.n_layers * self.hidden_dim} = {100 * intrinsic_dim / (self.n_layers * self.hidden_dim):.6f}% of full space"

        elif intrinsic_dim <= LOW_DIM_THRESHOLD:
            # Moderately low - check curvature/modes
            if n_modes <= intrinsic_dim:
                # Single connected region
                recommendation['use_cone'] = True
                recommendation['cone_rank'] = intrinsic_dim
                recommendation['reasoning'] = (
                    f"Intrinsic dimension ({intrinsic_dim}) is low and geometry appears "
                    f"single-mode. Cone approximation should work well."
                )
            else:
                # Multiple disconnected modes
                recommendation['use_cone'] = False
                recommendation['reasoning'] = (
                    f"Multiple modes ({n_modes}) detected despite low intrinsic dimension ({intrinsic_dim}). "
                    f"Geometry is likely multi-modal - cone approximation may be poor."
                )
                recommendation['alternative'] = "Use GP with multi-modal kernel or neural field"

        else:
            # High-dimensional - don't use cone
            recommendation['use_cone'] = False
            recommendation['reasoning'] = (
                f"Intrinsic dimension ({intrinsic_dim}) is too high for efficient cone. "
                f"Geometry is complex."
            )
            recommendation['alternative'] = "Use GP for smooth interpolation or neural implicit field for capacity"

        return recommendation


# Example usage
if __name__ == '__main__':
    """
    Example: Discover geometry of refusal subspace.
    """

    print("=" * 70)
    print("Adaptive Geometry Discovery - Demo")
    print("=" * 70)

    # Using smaller dimensions for fast demo
    demo_n_layers = 4
    demo_hidden_dim = 256

    # Fixed "true" directions for reproducible demo
    torch.manual_seed(42)
    TRUE_DIR1 = torch.randn(demo_n_layers, demo_hidden_dim)
    TRUE_DIR1 = TRUE_DIR1 / TRUE_DIR1.norm(dim=1, keepdim=True)
    TRUE_DIR2 = torch.randn(demo_n_layers, demo_hidden_dim)
    TRUE_DIR2 = TRUE_DIR2 / TRUE_DIR2.norm(dim=1, keepdim=True)

    def mock_measure_refusal(v: torch.Tensor) -> float:
        """
        Mock refusal strength measurement.

        Simulates a refusal subspace with:
        - 2 principal modes (two different refusal directions)
        """
        v_flat = v.reshape(-1)
        dir1_flat = TRUE_DIR1.reshape(-1)
        dir2_flat = TRUE_DIR2.reshape(-1)

        alignment1 = (v_flat @ dir1_flat) / (v_flat.norm() * dir1_flat.norm())
        alignment2 = (v_flat @ dir2_flat) / (v_flat.norm() * dir2_flat.norm())

        # Refusal strength: high if aligned with either direction
        R = max(abs(alignment1.item()), abs(alignment2.item()))

        # Add noise
        R = R + 0.1 * torch.randn(1).item()
        R = np.clip(R, 0, 1)

        return R

    # Run discovery with defaults (now sane: sparse GP, 100 candidates, 30 iterations)

    discovery = RefusalGeometryDiscovery(
        measure_refusal_fn=mock_measure_refusal,
        n_layers=demo_n_layers,
        hidden_dim=demo_hidden_dim,
        # config=GeometryConfig()  # Uses sane defaults
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
