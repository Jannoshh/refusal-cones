#!/usr/bin/env python3
"""
Unified RDO Adapter - Affine Concept Editing

Implements the unified affine transformation from "Refusal in LLMs is an Affine Function"
(https://arxiv.org/abs/2411.09003v3)

Key insight: Combine projection and addition into single affine transformation:
    h' = P @ h + α·v

Where:
    P = (I - vv^T) or more generally (I - L@R) for rank-k
    v = steering vector
    α = scaling factor

This is more expressive than separate ablate/add modes.

This module consolidates all adapter functionality:
- ProjectionConfig (base config with PEFT compatibility)
- UnifiedRDOConfig (extends with affine parameters)
- UnifiedRDOLayer (rank-1 affine transformation)
- RankKUnifiedLayer (rank-k affine transformation)

Backward-compatible aliases are provided for RDOConfig, get_rdo_model, etc.
"""

import torch
import torch.nn as nn
from typing import Optional, List, Literal
from dataclasses import dataclass, field

# Try importing PEFT (graceful fallback if not installed)
try:
    from peft import PeftConfig, PeftModel, PeftType
    from peft.tuners import BaseTuner, BaseTunerLayer
    from peft.utils import PeftType, _get_submodules
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    # Fallback definitions
    class PeftConfig:
        pass
    class BaseTuner:
        pass
    class BaseTunerLayer:
        pass


@dataclass
class ProjectionConfig(PeftConfig if PEFT_AVAILABLE else object):
    """
    Base configuration for projection adapters.

    This is a PEFT-compatible config for projection adapters.
    Think of it as "rank-1 LoRA but with subtraction instead of addition".

    Args:
        target_modules: Which modules to add projection to (e.g., ["layers"])
        projection_alpha: Scaling factor for projection (default: 1.0)
        projection_init_std: Initialization std for vectors
        normalize_vectors: Whether to normalize projection vectors
    """

    target_modules: Optional[List[str]] = field(default=None)
    projection_alpha: float = 1.0
    projection_init_std: float = 0.01
    normalize_vectors: bool = True

    if PEFT_AVAILABLE:
        def __post_init__(self):
            self.peft_type = "PROJECTION"
            super().__post_init__()


@dataclass
class UnifiedRDOConfig(ProjectionConfig):
    """
    Configuration for unified affine RDO.

    Key difference from standard RDO:
    - Single affine transformation instead of separate ablate/add modes
    - Projection and addition happen simultaneously
    """

    # Affine transformation parameters
    projection_alpha: float = 1.0  # Scaling for projection (usually 1.0)
    addition_alpha: float = 1.0    # Scaling for addition component

    # Operation mode (simplified)
    operation: Literal['affine', 'project_only', 'add_only', 'none'] = 'affine'

    # Advanced: rank-k projection (for discovered geometry)
    enable_rank_k: bool = False
    rank_k: int = 1  # Number of vectors in subspace

    # Affine reference point (baseline)
    use_baseline: bool = False  # Use baseline from harmless prompts
    train_magnitudes: bool = False  # Train magnitudes or compute from data

    # Multi-dimensional ACE options
    force_orthogonal: bool = True  # Force orthogonalization (faster, more stable)
    per_direction_steering: bool = False  # Separate α for each direction (k alphas vs 1)

    # Loss weights
    lambda_harmful: float = 1.0    # Weight for harmful examples
    lambda_harmless: float = 0.5   # Weight for harmless examples


class UnifiedRDOLayer(nn.Module):
    """
    Unified affine transformation layer with baseline support.

    Two modes:

    1. Standard (use_baseline=False):
       h' = (I - β·vv^T)h + α·v

    2. Affine with baseline (use_baseline=True):
       h' = h - β·((h - h0)·v_norm)·v_norm + α·steering_mag·v_norm

    Where:
        h0 = baseline (mean activation on harmless prompts)
        v_norm = normalized direction (only trainable param)
        steering_mag = magnitude (auto-computed from data)

    This is the true affine formulation from the paper.
    """

    def __init__(
        self,
        base_layer: nn.Module,
        dim: int,
        projection_alpha: float = 1.0,
        addition_alpha: float = 1.0,
        normalize: bool = True,
        use_baseline: bool = False,
        train_magnitudes: bool = False
    ):
        super().__init__()

        self.base_layer = base_layer
        self.projection_alpha = projection_alpha
        self.addition_alpha = addition_alpha
        self.normalize = normalize
        self.use_baseline = use_baseline
        self.train_magnitudes = train_magnitudes

        # Freeze base layer
        for param in base_layer.parameters():
            param.requires_grad = False

        # Trainable steering vector (normalized direction)
        self.vector = nn.Parameter(torch.randn(dim) * 0.01)

        # Baseline and magnitudes (non-trainable by default, computed from data)
        self.register_buffer('baseline', torch.zeros(dim))
        self.register_buffer('baseline_component', torch.tensor(0.0))
        self.register_buffer('steering_magnitude', torch.tensor(1.0))

        # Optionally make magnitudes trainable
        if train_magnitudes:
            self.baseline_component = nn.Parameter(self.baseline_component)
            self.steering_magnitude = nn.Parameter(self.steering_magnitude)

    def forward(self, x, *args, **kwargs):
        """
        Forward with unified affine transformation.

        Standard mode: h' = (I - β·vv^T)h + α·v
        Baseline mode: h' = h - β·((h - h0)·v)v + α·mag·v
        """
        # Base layer forward
        result = self.base_layer(x, *args, **kwargs)

        # Handle tuple outputs
        if isinstance(result, tuple):
            activations = result[0]
            extra_outputs = result[1:]
        else:
            activations = result
            extra_outputs = None

        # Get normalized vector
        v = self.vector
        if self.normalize:
            v = v / (v.norm() + 1e-8)

        if self.use_baseline:
            # Affine mode with baseline reference point
            # h' = h - β·((h - h0)·v)v + α·mag·v

            # Subtract baseline (center relative to harmless mean)
            h_centered = activations - self.baseline

            # Project out centered component
            projection_magnitude = torch.einsum('...d,d->...', h_centered, v)
            projection = torch.einsum('...,d->...d', projection_magnitude, v)

            # Add back steering with learned magnitude
            modified = (activations - self.projection_alpha * projection +
                       self.addition_alpha * self.steering_magnitude * v)
        else:
            # Standard mode (no baseline)
            # h' = h - β·(h·v)v + α·v

            # Projection component: β·(h·v)v
            projection_magnitude = torch.einsum('...d,d->...', activations, v)
            projection = torch.einsum('...,d->...d', projection_magnitude, v)

            # Affine transformation
            modified = activations - self.projection_alpha * projection + self.addition_alpha * v

        # Reconstruct output
        if extra_outputs is not None:
            return (modified,) + extra_outputs
        return modified

    def fit_baseline_and_magnitude(
        self,
        harmless_activations: torch.Tensor,
        harmful_activations: torch.Tensor
    ):
        """
        Fit baseline and steering magnitude from data.

        This computes:
        1. baseline (h0) = mean activation on harmless prompts
        2. steering_magnitude = mean(h_harmful · v) - mean(h_harmless · v)

        Args:
            harmless_activations: Activations on harmless prompts [n_harmless, dim]
            harmful_activations: Activations on harmful prompts [n_harmful, dim]

        Usage:
            # After initializing layer
            layer.fit_baseline_and_magnitude(harmless_acts, harmful_acts)
            # Now baseline and steering_magnitude are set automatically!
        """
        # Get normalized direction
        v = self.vector.detach()
        if self.normalize:
            v = v / (v.norm() + 1e-8)

        # Compute baseline (mean harmless activation)
        baseline = harmless_activations.mean(dim=0)
        self.baseline.copy_(baseline)

        # Compute mean dot products
        harmless_dots = (harmless_activations @ v).mean()
        harmful_dots = (harmful_activations @ v).mean()

        # Steering magnitude = difference in mean projections
        steering_mag = harmful_dots - harmless_dots
        self.steering_magnitude.copy_(steering_mag)

        # Baseline component (for reference, usually close to 0 if centered)
        baseline_comp = ((harmless_activations - baseline) @ v).mean()
        self.baseline_component.copy_(baseline_comp)

        print(f"  Fitted affine parameters:")
        print(f"    Baseline norm: {baseline.norm().item():.4f}")
        print(f"    Baseline component: {baseline_comp.item():.4f}")
        print(f"    Steering magnitude: {steering_mag.item():.4f}")

    def get_projection_matrix(self) -> torch.Tensor:
        """
        Get the projection matrix P = I - β·vv^T.

        Returns:
            Projection matrix [dim, dim]
        """
        v = self.vector
        if self.normalize:
            v = v / (v.norm() + 1e-8)

        dim = len(v)
        I = torch.eye(dim, device=v.device, dtype=v.dtype)
        P = I - self.projection_alpha * torch.outer(v, v)
        return P

    def get_affine_components(self):
        """
        Get affine transformation components.

        Returns:
            (P, b) where h' = P @ h + b
        """
        v = self.vector
        if self.normalize:
            v = v / (v.norm() + 1e-8)

        P = self.get_projection_matrix()
        b = self.addition_alpha * v

        return P, b


class RankKUnifiedLayer(nn.Module):
    """
    Rank-k unified affine transformation with ACE baseline.

    Two modes:

    1. Orthogonal (force_orthogonal=True, default):
       h' = h - VV^T(h - v⁻) + V·α
       where V is orthonormalized via Gram-Schmidt

    2. Non-orthogonal (force_orthogonal=False):
       h' = h - V(V^T V)^{-1}V^T(h - v⁻) + V·α
       where V can be arbitrary (uses pseudoinverse projection)

    Args:
        base_layer: The layer to wrap
        dim: Hidden dimension
        rank_k: Number of subspace directions
        projection_alpha: β scaling for projection (default 1.0)
        addition_alpha: Scaling for steering term (default 1.0)
        normalize: Whether to normalize vectors (only used if force_orthogonal=True)
        use_baseline: Use ACE baseline restoration v⁻ (recommended)
        force_orthogonal: Enforce orthogonality via Gram-Schmidt (faster, more stable)
        per_direction_steering: Use vector α ∈ R^k instead of scalar (more expressive)
    """

    def __init__(
        self,
        base_layer: nn.Module,
        dim: int,
        rank_k: int = 3,
        projection_alpha: float = 1.0,
        addition_alpha: float = 1.0,
        normalize: bool = True,
        use_baseline: bool = False,
        force_orthogonal: bool = True,
        per_direction_steering: bool = False
    ):
        super().__init__()

        self.base_layer = base_layer
        self.rank_k = rank_k
        self.projection_alpha = projection_alpha
        self.addition_alpha = addition_alpha
        self.normalize = normalize
        self.use_baseline = use_baseline
        self.force_orthogonal = force_orthogonal
        self.per_direction_steering = per_direction_steering

        # Freeze base layer
        for param in base_layer.parameters():
            param.requires_grad = False

        # Trainable subspace vectors [rank_k, dim]
        self.subspace_vectors = nn.Parameter(
            torch.randn(rank_k, dim) * 0.01
        )

        # ACE baseline (v⁻): mean harmless activations
        # Registered as buffer (not trained, computed from data)
        self.register_buffer('v_minus', torch.zeros(dim))
        self._baseline_fitted = False

        # Steering coefficients
        if per_direction_steering:
            # Vector α ∈ R^k (one per direction)
            self.alpha = nn.Parameter(torch.zeros(rank_k))
        else:
            # Scalar α (scaled v_mean)
            self.alpha = nn.Parameter(torch.tensor(0.0))

    def get_orthonormal_basis(self):
        """
        Get orthonormalized subspace basis using Gram-Schmidt.

        Returns:
            Orthonormal basis [rank_k, dim]
        """
        vectors = self.subspace_vectors

        # Gram-Schmidt orthogonalization
        ortho_vectors = []
        for i in range(len(vectors)):
            v = vectors[i].clone()

            # Subtract projections onto previous vectors
            for prev in ortho_vectors:
                v = v - (v @ prev) * prev

            # Normalize
            if self.normalize:
                v = v / (v.norm() + 1e-8)

            ortho_vectors.append(v)

        return torch.stack(ortho_vectors)

    def _project_orthogonal(self, h: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        """
        Orthogonal projection: P_V(h) = VV^T h

        Args:
            h: Input [..., dim]
            V: Orthonormal basis [k, dim]

        Returns:
            Projection [..., dim]
        """
        # V^T h
        VT_h = torch.einsum('kd,...d->...k', V, h)  # [..., k]

        # V (V^T h)
        proj = torch.einsum('...k,kd->...d', VT_h, V)  # [..., dim]

        return proj

    def _project_pseudoinverse(self, h: torch.Tensor, V: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        """
        Pseudoinverse projection: P_V(h) = V(V^T V)^{-1}V^T h

        Args:
            h: Input [..., dim]
            V: Arbitrary basis [k, dim] (not necessarily orthogonal)
            eps: Regularization for numerical stability

        Returns:
            Projection [..., dim]
        """
        # Gram matrix G = V V^T
        G = V @ V.T  # [k, k]

        # Regularize and invert
        G_reg = G + eps * torch.eye(G.shape[0], device=G.device, dtype=G.dtype)
        G_inv = torch.linalg.inv(G_reg)  # [k, k]

        # Compute coefficients: (V^T V)^{-1} V^T h
        VT_h = torch.einsum('kd,...d->...k', V, h)  # [..., k]
        coeffs = torch.einsum('...k,kj->...j', VT_h, G_inv)  # [..., k]

        # Project back: V · coeffs
        proj = torch.einsum('...k,kd->...d', coeffs, V)  # [..., dim]

        return proj

    def forward(self, x, *args, **kwargs):
        """
        Forward with rank-k affine ACE transformation.

        Computes:
            If use_baseline:
                h' = h - P_V(h - v⁻) + steering
            Else:
                h' = h - P_V(h) + steering

        Where P_V is orthogonal or pseudoinverse projection.
        """
        # Base layer forward
        result = self.base_layer(x, *args, **kwargs)

        # Handle tuple outputs
        if isinstance(result, tuple):
            activations = result[0]
            extra_outputs = result[1:]
        else:
            activations = result
            extra_outputs = None

        # Get basis (orthogonal or not)
        if self.force_orthogonal:
            V = self.get_orthonormal_basis()  # [k, dim]
        else:
            V = self.subspace_vectors  # [k, dim] (raw, possibly non-orthogonal)

        # Compute what to project
        if self.use_baseline:
            if not self._baseline_fitted:
                # Warning: baseline not fitted, using zero
                # (Will be properly fitted via fit_baseline() method)
                pass
            delta = activations - self.v_minus  # [..., dim]
        else:
            delta = activations

        # Project using appropriate method
        if self.force_orthogonal:
            proj_delta = self._project_orthogonal(delta, V)
        else:
            proj_delta = self._project_pseudoinverse(delta, V)

        # Remove projection
        modified = activations - self.projection_alpha * proj_delta

        # Add steering term
        if self.per_direction_steering:
            # V·α where α ∈ R^k
            steering = torch.einsum('k,kd->d', self.alpha, V)  # [dim]
        else:
            # α·mean(V) where α is scalar
            v_mean = V.mean(dim=0)  # [dim]
            steering = self.alpha * v_mean  # [dim]

        modified = modified + self.addition_alpha * steering

        # Reconstruct output
        if extra_outputs is not None:
            return (modified,) + extra_outputs
        return modified

    def fit_baseline(self, harmless_activations: torch.Tensor):
        """
        Fit ACE baseline v⁻ from harmless data.

        Args:
            harmless_activations: Activations on harmless prompts [n_samples, dim]
        """
        with torch.no_grad():
            self.v_minus.copy_(harmless_activations.mean(dim=0))
            self._baseline_fitted = True

    def initialize_from_mean_diff(
        self,
        harmful_activations: torch.Tensor,
        harmless_activations: torch.Tensor,
        noise_std: float = 0.01
    ):
        """
        Initialize subspace vectors from mean difference + noise.

        Creates k vectors by adding small random noise to the mean difference
        direction. This provides a good starting point for discovering the
        refusal subspace.

        Args:
            harmful_activations: Activations on harmful prompts [n_samples, dim]
            harmless_activations: Activations on harmless prompts [n_samples, dim]
            noise_std: Standard deviation of noise to add (default: 0.01)
        """
        with torch.no_grad():
            # Compute mean difference (primary refusal direction)
            mean_diff = harmful_activations.mean(dim=0) - harmless_activations.mean(dim=0)
            mean_diff = mean_diff / (mean_diff.norm() + 1e-8)  # Normalize

            # Create k vectors with small perturbations
            for i in range(self.rank_k):
                # Base direction + small random noise
                noise = torch.randn_like(mean_diff) * noise_std
                perturbed = mean_diff + noise

                # Normalize
                perturbed = perturbed / (perturbed.norm() + 1e-8)

                self.subspace_vectors.data[i] = perturbed

            # Also fit baseline if enabled
            if self.use_baseline:
                self.fit_baseline(harmless_activations)

    def get_projection_matrix(self) -> torch.Tensor:
        """
        Get the rank-k projection matrix.

        Returns:
            Projection matrix [dim, dim]
        """
        if self.force_orthogonal:
            # P = I - β·VV^T (orthogonal case)
            V = self.get_orthonormal_basis()  # [k, dim]

            dim = V.shape[1]
            I = torch.eye(dim, device=V.device, dtype=V.dtype)

            # VV^T = sum of outer products
            projection_sum = sum(torch.outer(v, v) for v in V)
            P = I - self.projection_alpha * projection_sum

        else:
            # P = I - β·V(V^T V)^{-1}V^T (pseudoinverse case)
            V = self.subspace_vectors  # [k, dim]

            dim = V.shape[1]
            I = torch.eye(dim, device=V.device, dtype=V.dtype)

            # Compute pseudoinverse projection
            G = V @ V.T  # [k, k]
            G_inv = torch.linalg.inv(G + 1e-6 * torch.eye(G.shape[0], device=G.device))

            # V(V^T V)^{-1}V^T
            pseudoinv_proj = V.T @ G_inv @ V  # [dim, dim]
            P = I - self.projection_alpha * pseudoinv_proj

        return P

    def get_condition_number(self) -> float:
        """
        Get condition number of Gram matrix V^T V.

        Useful for monitoring numerical stability in non-orthogonal mode.
        High condition number (> 100) indicates near-linear dependence.

        Returns:
            Condition number (1.0 for orthogonal, higher for non-orthogonal)
        """
        V = self.subspace_vectors  # [k, dim]
        G = V @ V.T  # [k, k]

        # Condition number = largest_eigenvalue / smallest_eigenvalue
        eigenvalues = torch.linalg.eigvalsh(G)
        cond = eigenvalues.max() / (eigenvalues.min() + 1e-10)

        return cond.item()


class UnifiedRDOModel(nn.Module):
    """
    Model wrapper with unified affine RDO adapters.

    Key difference from standard RDO:
    - Single affine transformation per layer (not separate modes)
    - Projection and addition happen simultaneously
    - Can use rank-k for complex geometries
    """

    def __init__(self, model, config: UnifiedRDOConfig):
        super().__init__()

        self.base_model = model
        self.config = config

        # Freeze base model
        for param in model.parameters():
            param.requires_grad = False

        # Add unified RDO layers
        self._inject_adapters()

    def _inject_adapters(self):
        """Inject unified affine adapters into target modules."""

        if not self.config.target_modules:
            raise ValueError("Must specify target_modules in config")

        # Get layers
        if hasattr(self.base_model, 'model'):
            layers = self.base_model.model.layers
        elif hasattr(self.base_model, 'transformer'):
            layers = self.base_model.transformer.h
        else:
            raise ValueError("Unsupported model architecture")

        # Get dimension
        if hasattr(self.base_model.config, 'hidden_size'):
            dim = self.base_model.config.hidden_size
        elif hasattr(self.base_model.config, 'n_embd'):
            dim = self.base_model.config.n_embd
        else:
            raise ValueError("Cannot determine hidden dimension")

        # Choose layer class
        layer_class = RankKUnifiedLayer if self.config.enable_rank_k else UnifiedRDOLayer

        # Wrap each layer
        for idx, layer in enumerate(layers):
            if self.config.enable_rank_k:
                wrapped = RankKUnifiedLayer(
                    base_layer=layer,
                    dim=dim,
                    rank_k=self.config.rank_k,
                    projection_alpha=self.config.projection_alpha,
                    addition_alpha=self.config.addition_alpha,
                    normalize=self.config.normalize_vectors,
                    use_baseline=self.config.use_baseline,
                    force_orthogonal=self.config.force_orthogonal,
                    per_direction_steering=self.config.per_direction_steering
                )
            else:
                wrapped = UnifiedRDOLayer(
                    base_layer=layer,
                    dim=dim,
                    projection_alpha=self.config.projection_alpha,
                    addition_alpha=self.config.addition_alpha,
                    normalize=self.config.normalize_vectors,
                    use_baseline=self.config.use_baseline,
                    train_magnitudes=self.config.train_magnitudes
                )
            layers[idx] = wrapped

        layer_type = "rank-k unified" if self.config.enable_rank_k else "unified affine"
        print(f"✓ Added {layer_type} RDO adapters to {len(layers)} layers")
        if self.config.enable_rank_k:
            print(f"  Subspace rank: {self.config.rank_k} vectors per layer")

    def forward(self, *args, **kwargs):
        """Forward pass with unified affine transformation."""
        return self.base_model(*args, **kwargs)

    def generate(self, *args, **kwargs):
        """Generation with unified affine transformation."""
        return self.base_model.generate(*args, **kwargs)

    def get_affine_parameters(self) -> List[nn.Parameter]:
        """Get trainable affine transformation parameters."""
        params = []
        for module in self.modules():
            if isinstance(module, UnifiedRDOLayer):
                params.append(module.vector)
            elif isinstance(module, RankKUnifiedLayer):
                params.append(module.subspace_vectors)
        return params

    def print_trainable_parameters(self):
        """Print trainable parameter statistics."""
        trainable_params = 0
        all_params = 0

        for _, param in self.named_parameters():
            num_params = param.numel()
            all_params += num_params
            if param.requires_grad:
                trainable_params += num_params

        print(
            f"trainable params: {trainable_params:,} || "
            f"all params: {all_params:,} || "
            f"trainable%: {100 * trainable_params / all_params:.4f}"
        )

    def fit_all_baselines(
        self,
        tokenizer,
        harmless_prompts: List[str],
        harmful_prompts: List[str],
        batch_size: int = 8
    ):
        """
        Fit baseline and steering magnitudes for all layers from data.

        This automatically computes:
        1. baseline (h0) = mean activation on harmless prompts per layer
        2. steering_magnitude = mean(h_harmful · v) - mean(h_harmless · v) per layer

        After calling this, you only need to train the direction vectors!

        Args:
            tokenizer: Tokenizer
            harmless_prompts: List of harmless prompts
            harmful_prompts: List of harmful prompts
            batch_size: Batch size for activation collection

        Usage:
            model = get_unified_rdo_model(base_model, config)
            model.fit_all_baselines(tokenizer, harmless_prompts, harmful_prompts)
            # Now train only the direction vectors!
        """
        print("=" * 70)
        print("Fitting Baselines and Magnitudes from Data")
        print("=" * 70)

        # Collect activations per layer
        from collections import defaultdict
        harmless_acts = defaultdict(list)
        harmful_acts = defaultdict(list)

        # Register hooks to capture activations
        handles = []
        layer_idx = 0

        if hasattr(self.base_model, 'model'):
            layers = self.base_model.model.layers
        elif hasattr(self.base_model, 'transformer'):
            layers = self.base_model.transformer.h
        else:
            raise ValueError("Unsupported model architecture")

        def get_hook(layer_id):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    act = output[0][:, -1, :].detach()  # Last token
                else:
                    act = output[:, -1, :].detach()
                if layer_id in harmless_acts:  # Currently collecting harmless
                    harmless_acts[layer_id].append(act.cpu())
                else:  # Currently collecting harmful
                    harmful_acts[layer_id].append(act.cpu())
            return hook

        for idx, layer in enumerate(layers):
            if isinstance(layer, UnifiedRDOLayer):
                handle = layer.base_layer.register_forward_hook(get_hook(idx))
                handles.append(handle)
                layer_idx += 1

        # Collect harmless activations
        print(f"\n1. Collecting activations on {len(harmless_prompts)} harmless prompts...")
        self.base_model.eval()
        with torch.no_grad():
            for i in range(0, len(harmless_prompts), batch_size):
                batch = harmless_prompts[i:i+batch_size]

                # Format with chat template
                formatted_batch = []
                for prompt in batch:
                    messages = [{"role": "user", "content": prompt}]
                    formatted = tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True
                    )
                    formatted_batch.append(formatted)

                inputs = tokenizer(formatted_batch, return_tensors='pt', padding=True, truncation=True)
                inputs = {k: v.to(self.base_model.device) for k, v in inputs.items()}
                _ = self.base_model(**inputs)

        # Clear for harmful
        for idx in list(harmless_acts.keys()):
            harmful_acts[idx] = []

        # Collect harmful activations
        print(f"2. Collecting activations on {len(harmful_prompts)} harmful prompts...")
        with torch.no_grad():
            for i in range(0, len(harmful_prompts), batch_size):
                batch = harmful_prompts[i:i+batch_size]

                # Format with chat template
                formatted_batch = []
                for prompt in batch:
                    messages = [{"role": "user", "content": prompt}]
                    formatted = tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True
                    )
                    formatted_batch.append(formatted)

                inputs = tokenizer(formatted_batch, return_tensors='pt', padding=True, truncation=True)
                inputs = {k: v.to(self.base_model.device) for k, v in inputs.items()}
                _ = self.base_model(**inputs)

        # Remove hooks
        for handle in handles:
            handle.remove()

        # Fit each layer
        print(f"\n3. Fitting {len(harmless_acts)} layers...")
        layer_idx = 0
        for idx, layer in enumerate(layers):
            if isinstance(layer, UnifiedRDOLayer) and idx in harmless_acts:
                print(f"\nLayer {layer_idx}:")

                # Stack activations
                harmless_tensor = torch.cat(harmless_acts[idx], dim=0)
                harmful_tensor = torch.cat(harmful_acts[idx], dim=0)

                # Fit baseline and magnitude
                layer.fit_baseline_and_magnitude(harmless_tensor, harmful_tensor)

                layer_idx += 1

        print("\n" + "=" * 70)
        print("✓ Baselines and magnitudes fitted!")
        print("=" * 70)
        print("\nNow you only need to train the direction vectors.")
        print("The magnitudes are fixed from data (unless train_magnitudes=True).")

    def initialize_rank_k_from_mean_diff(
        self,
        tokenizer,
        harmless_prompts: List[str],
        harmful_prompts: List[str],
        batch_size: int = 8,
        noise_std: float = 0.01
    ):
        """
        Initialize all rank-k layers from mean difference + noise.

        This is useful for warm-starting training with a good initialization
        based on the primary refusal direction.

        Args:
            tokenizer: Tokenizer
            harmless_prompts: List of harmless prompts
            harmful_prompts: List of harmful prompts
            batch_size: Batch size for activation collection
            noise_std: Noise level for perturbations (default: 0.01)

        Usage:
            config = UnifiedRDOConfig(
                enable_rank_k=True,
                rank_k=3,
                use_baseline=True,
                force_orthogonal=False  # Allow non-orthogonal vectors
            )
            model = get_unified_rdo_model(base_model, config)
            model.initialize_rank_k_from_mean_diff(tokenizer, harmless, harmful)
        """
        if not self.config.enable_rank_k:
            raise ValueError("This method only works with rank-k layers (enable_rank_k=True)")

        print("=" * 70)
        print("Initializing Rank-K Subspaces from Mean Difference")
        print("=" * 70)

        # Collect activations per layer (reuse logic from fit_all_baselines)
        from collections import defaultdict
        harmless_acts = defaultdict(list)
        harmful_acts = defaultdict(list)

        # Register hooks
        handles = []
        if hasattr(self.base_model, 'model'):
            layers = self.base_model.model.layers
        elif hasattr(self.base_model, 'transformer'):
            layers = self.base_model.transformer.h
        else:
            raise ValueError("Unsupported model architecture")

        def get_hook(layer_id):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    act = output[0][:, -1, :].detach()
                else:
                    act = output[:, -1, :].detach()
                if layer_id in harmless_acts:
                    harmless_acts[layer_id].append(act.cpu())
                else:
                    harmful_acts[layer_id].append(act.cpu())
            return hook

        for idx, layer in enumerate(layers):
            if isinstance(layer, RankKUnifiedLayer):
                handle = layer.base_layer.register_forward_hook(get_hook(idx))
                handles.append(handle)

        # Collect activations
        print(f"\n1. Collecting harmless activations ({len(harmless_prompts)} prompts)...")
        self.base_model.eval()
        with torch.no_grad():
            for i in range(0, len(harmless_prompts), batch_size):
                batch = harmless_prompts[i:i+batch_size]
                formatted_batch = []
                for prompt in batch:
                    messages = [{"role": "user", "content": prompt}]
                    formatted = tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                    formatted_batch.append(formatted)

                inputs = tokenizer(formatted_batch, return_tensors='pt', padding=True, truncation=True)
                inputs = {k: v.to(self.base_model.device) for k, v in inputs.items()}
                _ = self.base_model(**inputs)

        # Clear for harmful
        for idx in list(harmless_acts.keys()):
            harmful_acts[idx] = []

        print(f"2. Collecting harmful activations ({len(harmful_prompts)} prompts)...")
        with torch.no_grad():
            for i in range(0, len(harmful_prompts), batch_size):
                batch = harmful_prompts[i:i+batch_size]
                formatted_batch = []
                for prompt in batch:
                    messages = [{"role": "user", "content": prompt}]
                    formatted = tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                    formatted_batch.append(formatted)

                inputs = tokenizer(formatted_batch, return_tensors='pt', padding=True, truncation=True)
                inputs = {k: v.to(self.base_model.device) for k, v in inputs.items()}
                _ = self.base_model(**inputs)

        # Remove hooks
        for handle in handles:
            handle.remove()

        # Initialize each rank-k layer
        print(f"\n3. Initializing {len(harmless_acts)} rank-k layers...")
        layer_idx = 0
        for idx, layer in enumerate(layers):
            if isinstance(layer, RankKUnifiedLayer) and idx in harmless_acts:
                harmless_tensor = torch.cat(harmless_acts[idx], dim=0)
                harmful_tensor = torch.cat(harmful_acts[idx], dim=0)

                # Move to layer's device
                harmless_tensor = harmless_tensor.to(layer.subspace_vectors.device)
                harmful_tensor = harmful_tensor.to(layer.subspace_vectors.device)

                # Initialize from mean diff
                layer.initialize_from_mean_diff(
                    harmful_tensor, harmless_tensor, noise_std=noise_std
                )

                # Print diagnostics
                if not layer.force_orthogonal:
                    cond = layer.get_condition_number()
                    print(f"  Layer {layer_idx}: initialized (condition number: {cond:.2f})")
                else:
                    print(f"  Layer {layer_idx}: initialized (orthogonal)")

                layer_idx += 1

        print("\n" + "=" * 70)
        print("✓ Rank-k subspaces initialized!")
        print("=" * 70)
        if self.config.use_baseline:
            print("Baselines (v⁻) fitted from harmless data")
        if not self.config.force_orthogonal:
            print(f"Vectors are non-orthogonal (noise_std={noise_std})")
            print("Monitor condition numbers during training to check stability")

    def get_all_condition_numbers(self) -> List[float]:
        """
        Get condition numbers for all rank-k layers.

        Useful for monitoring numerical stability when using non-orthogonal vectors.

        Returns:
            List of condition numbers (one per layer)
        """
        if not self.config.enable_rank_k:
            raise ValueError("This method only works with rank-k layers")

        cond_numbers = []
        if hasattr(self.base_model, 'model'):
            layers = self.base_model.model.layers
        elif hasattr(self.base_model, 'transformer'):
            layers = self.base_model.transformer.h
        else:
            raise ValueError("Unsupported model architecture")

        for layer in layers:
            if isinstance(layer, RankKUnifiedLayer):
                cond = layer.get_condition_number()
                cond_numbers.append(cond)

        return cond_numbers

    def save_pretrained(self, save_directory: str):
        """Save unified RDO adapters."""
        import os
        import json

        os.makedirs(save_directory, exist_ok=True)

        # Save config
        config_dict = {
            'peft_type': 'UNIFIED_RDO',
            'target_modules': self.config.target_modules,
            'projection_alpha': self.config.projection_alpha,
            'addition_alpha': self.config.addition_alpha,
            'operation': self.config.operation,
            'enable_rank_k': self.config.enable_rank_k,
            'rank_k': self.config.rank_k,
            'normalize_vectors': self.config.normalize_vectors,
        }

        with open(os.path.join(save_directory, 'adapter_config.json'), 'w') as f:
            json.dump(config_dict, f, indent=2)

        # Save vectors/subspaces
        affine_data = []
        for module in self.modules():
            if isinstance(module, UnifiedRDOLayer):
                affine_data.append(module.vector.data.cpu())
            elif isinstance(module, RankKUnifiedLayer):
                affine_data.append(module.subspace_vectors.data.cpu())

        torch.save({
            'affine_data': affine_data,
            'enable_rank_k': self.config.enable_rank_k
        }, os.path.join(save_directory, 'adapter_model.bin'))

        print(f"✓ Saved unified RDO adapters to {save_directory}")

    @classmethod
    def from_pretrained(cls, model, adapter_path: str):
        """Load unified RDO adapters onto a model."""
        import os
        import json

        # Load config
        with open(os.path.join(adapter_path, 'adapter_config.json'), 'r') as f:
            config_dict = json.load(f)

        config = UnifiedRDOConfig(**config_dict)

        # Create unified RDO model
        unified_model = cls(model, config)

        # Load data
        state = torch.load(
            os.path.join(adapter_path, 'adapter_model.bin'),
            map_location='cpu'
        )

        affine_data = state['affine_data']

        # Apply to layers
        layer_idx = 0
        for module in unified_model.modules():
            if isinstance(module, UnifiedRDOLayer):
                module.vector.data = affine_data[layer_idx].to(module.vector.device)
                layer_idx += 1
            elif isinstance(module, RankKUnifiedLayer):
                module.subspace_vectors.data = affine_data[layer_idx].to(module.subspace_vectors.device)
                layer_idx += 1

        print(f"✓ Loaded unified RDO adapters from {adapter_path}")

        return unified_model


def get_unified_rdo_model(model, config: Optional[UnifiedRDOConfig] = None):
    """
    Add unified affine RDO adapters to a model.

    Args:
        model: Base model
        config: UnifiedRDOConfig (if None, uses defaults)

    Returns:
        Model with unified RDO adapters

    Example:
        >>> model = AutoModelForCausalLM.from_pretrained("model")
        >>> config = UnifiedRDOConfig(
        ...     target_modules=["layers"],
        ...     projection_alpha=1.0,
        ...     addition_alpha=1.0,
        ...     operation='affine'
        ... )
        >>> model = get_unified_rdo_model(model, config)
    """
    if config is None:
        config = UnifiedRDOConfig(target_modules=["layers"])

    return UnifiedRDOModel(model, config)


# Example usage
if __name__ == '__main__':
    """
    Example: Unified affine RDO vs separate operations.
    """

    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("=" * 70)
    print("Unified Affine RDO - Single Transformation")
    print("=" * 70)

    # Load model
    print("\nLoading model...")
    model = AutoModelForCausalLM.from_pretrained(
        'gpt2',
        torch_dtype=torch.float32
    )
    tokenizer = AutoTokenizer.from_pretrained('gpt2')

    print(f"  Model: GPT-2")
    print(f"  Layers: {len(model.transformer.h)}")

    # Test 1: Unified affine transformation
    print("\n" + "-" * 70)
    print("Test 1: Unified Affine Transformation")
    print("-" * 70)

    config = UnifiedRDOConfig(
        target_modules=["layers"],
        projection_alpha=1.0,  # Full ablation
        addition_alpha=1.0,    # Full addition
        operation='affine'
    )

    model_unified = get_unified_rdo_model(model, config)
    model_unified.print_trainable_parameters()

    # Test forward
    inputs = tokenizer("Test prompt", return_tensors='pt')

    with torch.no_grad():
        output = model_unified(**inputs)
    print(f"  ✓ Unified affine works, shape: {output.logits.shape}")

    # Test 2: Rank-k unified
    print("\n" + "-" * 70)
    print("Test 2: Rank-k Unified (for complex geometry)")
    print("-" * 70)

    config_rank_k = UnifiedRDOConfig(
        target_modules=["layers"],
        projection_alpha=1.0,
        addition_alpha=1.0,
        enable_rank_k=True,
        rank_k=3  # 3-dimensional subspace
    )

    model_rank_k = get_unified_rdo_model(model, config_rank_k)
    model_rank_k.print_trainable_parameters()

    with torch.no_grad():
        output = model_rank_k(**inputs)
    print(f"  ✓ Rank-k unified works, shape: {output.logits.shape}")

    # Comparison
    print("\n" + "=" * 70)
    print("Comparison: Separate vs Unified")
    print("=" * 70)

    print("\nSeparate Operations (old):")
    print("  Ablation:  h' = (I - vv^T)h")
    print("  Addition:  h' = h + α·v")
    print("  → Need to switch modes during training")
    print("  → Two forward passes for multi-objective loss")

    print("\nUnified Affine (new):")
    print("  Combined:  h' = (I - vv^T)h + α·v")
    print("  → Single transformation")
    print("  → One forward pass")
    print("  → More expressive (can ablate AND steer)")

    print("\n" + "=" * 70)
    print("✓ Unified affine RDO ready!")
    print("=" * 70)
