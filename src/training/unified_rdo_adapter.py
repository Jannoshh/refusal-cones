#!/usr/bin/env python3
"""
Unified RDO Adapter - Affine Concept Editing (ACE)

Implements the ACE formula from "Refusal in LLMs is an Affine Function"
(Marshall et al., 2024 - arXiv:2411.09003v3)

The ACE formula (Equation 5):

    h' = h - proj_r(h) + proj_r(r⁻) + α·r

Where:
    h   = input activation
    r   = r⁺ - r⁻ (mean harmful - mean harmless, unnormalized)
    r⁻  = mean activation on harmless prompts (baseline/reference point)
    r⁺  = mean activation on harmful prompts
    α   = steering parameter:
          - α = 0: behave like harmless (no refusal)
          - α = 1: behave like harmful (full refusal)
    proj_r(x) = (x · r̂) · r̂  where r̂ = r / ||r||

Key insight: The bias term proj_r(r⁻) ensures activations stay in a
sensible region of activation space after ablation.

For training, we use:
- α = 0 on harmful prompts (want compliance = no refusal)
- α = 1 on harmless prompts (preserve normal behavior)
"""

import torch
import torch.nn as nn
from typing import Optional, List, Literal
from dataclasses import dataclass, field


@dataclass
class UnifiedRDOConfig:
    """
    Configuration for unified affine RDO adapters.

    This implements ACE (Affine Concept Editing) as trainable adapters.

    Examples:
        # Single layer (paper recommends middle layer)
        config = UnifiedRDOConfig(layers=[15])

        # All layers
        config = UnifiedRDOConfig(layers=None)  # or layers='all'

        # Select layers (e.g., middle layers)
        config = UnifiedRDOConfig(layers=[10, 11, 12, 13, 14, 15])
    """

    # Which layers to apply ACE to
    # - None or 'all': all layers
    # - List[int]: specific layer indices (e.g., [15] for single layer)
    layers: Optional[List[int]] = None

    # Steering parameter α (0 = no refusal, 1 = full refusal)
    alpha: float = 0.0

    # Whether to normalize steering vector
    normalize_vectors: bool = True

    # Advanced: rank-k projection (for discovered geometry)
    enable_rank_k: bool = False
    rank_k: int = 1  # Number of vectors in subspace

    # Multi-dimensional ACE options (for rank-k layers)
    force_orthogonal: bool = True  # Force orthogonalization (faster, more stable)
    per_direction_steering: bool = False  # Separate α for each direction (k alphas vs 1)
    use_baseline: bool = True  # Use ACE baseline restoration v⁻

    # Loss weights for training
    lambda_harmful: float = 1.0    # Weight for harmful examples
    lambda_harmless: float = 0.5   # Weight for harmless examples


class UnifiedRDOLayer(nn.Module):
    """
    ACE (Affine Concept Editing) layer for training.

    Implements the paper's formula (Equation 5):

        h' = h - proj_r(h) + proj_r(r⁻) + α·r

    Where:
        h   = input activation
        r   = steering direction (trainable, unnormalized)
        r⁻  = baseline (mean harmless activations, fixed after fitting)
        α   = steering parameter (0 = no refusal, 1 = full refusal)
        proj_r(x) = (x · r̂) · r̂  where r̂ = r / ||r||

    The trainable parameter is `r` (the steering direction).
    The baseline `r_minus` is computed from data via fit_from_activations().
    """

    def __init__(
        self,
        base_layer: nn.Module,
        dim: int,
        alpha: float = 0.0,
        normalize: bool = True
    ):
        super().__init__()

        self.base_layer = base_layer
        self._alpha = alpha
        self.normalize = normalize

        # Freeze base layer
        for param in base_layer.parameters():
            param.requires_grad = False

        # Trainable steering direction r (unnormalized, following paper)
        self.r = nn.Parameter(torch.randn(dim) * 0.01)

        # Baseline r⁻ (mean harmless activations, computed from data)
        self.register_buffer('r_minus', torch.zeros(dim))

        # Optional: store r⁺ for analysis
        self.register_buffer('r_plus', torch.zeros(dim))

    @property
    def alpha(self) -> float:
        return self._alpha

    @alpha.setter
    def alpha(self, value: float):
        self._alpha = value

    def forward(self, x, *args, **kwargs):
        """
        Forward with ACE transformation (Equation 5).

        h' = h - proj_r(h) + proj_r(r⁻) + α·r
        """
        # Base layer forward
        result = self.base_layer(x, *args, **kwargs)

        # Handle tuple outputs
        if isinstance(result, tuple):
            h = result[0]
            extra_outputs = result[1:]
        else:
            h = result
            extra_outputs = None

        # Normalized direction r̂
        r_hat = self.r / (self.r.norm() + 1e-8)

        # proj_r(h) = (h · r̂) · r̂
        proj_h = torch.einsum('...d,d->...', h, r_hat)
        proj_h = torch.einsum('...,d->...d', proj_h, r_hat)

        # proj_r(r⁻) = (r⁻ · r̂) · r̂
        proj_r_minus = (self.r_minus @ r_hat) * r_hat

        # ACE formula: h' = h - proj_r(h) + proj_r(r⁻) + α·r
        h_prime = h - proj_h + proj_r_minus + self._alpha * self.r

        # Reconstruct output
        if extra_outputs is not None:
            return (h_prime,) + extra_outputs
        return h_prime

    def fit_from_activations(
        self,
        harmless_activations: torch.Tensor,
        harmful_activations: torch.Tensor
    ):
        """
        Fit r, r⁻, r⁺ from activation data (like ACEVectors.from_activations).

        Args:
            harmless_activations: Activations on harmless prompts [n_samples, dim]
            harmful_activations: Activations on harmful prompts [n_samples, dim]

        After calling this:
            - r = r⁺ - r⁻ (initialized to mean difference)
            - r_minus = mean(harmless_activations)
            - r_plus = mean(harmful_activations)
        """
        r_minus = harmless_activations.mean(dim=0)
        r_plus = harmful_activations.mean(dim=0)
        r = r_plus - r_minus

        self.r_minus.copy_(r_minus)
        self.r_plus.copy_(r_plus)
        self.r.data.copy_(r)

        print(f"  Fitted ACE vectors:")
        print(f"    ||r|| = {r.norm().item():.4f}")
        print(f"    ||r⁻|| = {r_minus.norm().item():.4f}")
        print(f"    ||r⁺|| = {r_plus.norm().item():.4f}")

    def get_ace_vectors(self):
        """
        Get the ACE vectors (r, r_minus, r_plus).

        Returns:
            dict with 'r', 'r_minus', 'r_plus' tensors
        """
        return {
            'r': self.r.detach().clone(),
            'r_minus': self.r_minus.clone(),
            'r_plus': self.r_plus.clone()
        }


class RankKUnifiedLayer(nn.Module):
    """
    Rank-k ACE layer for complex refusal geometries.

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
        alpha: Initial steering parameter (0 = no refusal, 1 = full refusal)
        normalize: Whether to normalize vectors (only used if force_orthogonal=True)
        use_baseline: Use ACE baseline restoration v⁻ (recommended)
        force_orthogonal: Enforce orthogonality via Gram-Schmidt (faster, more stable)
        per_direction_steering: Use vector α ∈ R^k instead of scalar (more expressive)
        projection_alpha: Scaling for projection term (default 1.0)
        addition_alpha: Scaling for steering term (default 1.0)
    """

    def __init__(
        self,
        base_layer: nn.Module,
        dim: int,
        rank_k: int = 3,
        alpha: float = 0.0,
        normalize: bool = True,
        use_baseline: bool = True,
        force_orthogonal: bool = True,
        per_direction_steering: bool = False,
        projection_alpha: float = 1.0,
        addition_alpha: float = 1.0
    ):
        super().__init__()

        self.base_layer = base_layer
        self.rank_k = rank_k
        self.normalize = normalize
        self.use_baseline = use_baseline
        self.force_orthogonal = force_orthogonal
        self.per_direction_steering = per_direction_steering
        self.projection_alpha = projection_alpha
        self.addition_alpha = addition_alpha
        self._init_alpha = alpha  # Store initial alpha for the property

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
        # Alias for backward compatibility
        self.register_buffer('r_minus', torch.zeros(dim))
        self._baseline_fitted = False

        # Per-direction steering coefficients (trainable)
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
        Forward with rank-k ACE transformation.

        Computes:
            If use_baseline:
                h' = h - P_V(h - v⁻) + steering
            Else:
                h' = h - P_V(h) + steering

        Where P_V is orthogonal or pseudoinverse projection.
        """
        import warnings

        # Base layer forward
        result = self.base_layer(x, *args, **kwargs)

        # Handle tuple outputs
        if isinstance(result, tuple):
            h = result[0]
            extra_outputs = result[1:]
        else:
            h = result
            extra_outputs = None

        # Get basis (orthogonal or not)
        if self.force_orthogonal:
            V = self.get_orthonormal_basis()  # [k, dim]
        else:
            V = self.subspace_vectors  # [k, dim] (raw, possibly non-orthogonal)

        # Compute what to project
        if self.use_baseline:
            if not self._baseline_fitted:
                warnings.warn(
                    "Baseline (v_minus) not fitted but use_baseline=True. "
                    "Using zero vector. Call fit_baseline() or fit_from_activations() first.",
                    UserWarning
                )
            delta = h - self.v_minus  # [..., dim]
        else:
            delta = h

        # Project using appropriate method
        if self.force_orthogonal:
            proj_delta = self._project_orthogonal(delta, V)
        else:
            proj_delta = self._project_pseudoinverse(delta, V)

        # Remove projection (with scaling)
        h_prime = h - self.projection_alpha * proj_delta

        # Add steering term
        if self.per_direction_steering:
            # V·α where α ∈ R^k
            steering = torch.einsum('k,kd->d', self.alpha, V)  # [dim]
        else:
            # α·mean(V) where α is scalar
            v_mean = V.mean(dim=0)  # [dim]
            steering = self.alpha * v_mean  # [dim]

        h_prime = h_prime + self.addition_alpha * steering

        # Reconstruct output
        if extra_outputs is not None:
            return (h_prime,) + extra_outputs
        return h_prime

    def fit_baseline(self, harmless_activations: torch.Tensor):
        """
        Fit ACE baseline v⁻ from harmless data only.

        Args:
            harmless_activations: Activations on harmless prompts [n_samples, dim]
        """
        with torch.no_grad():
            baseline = harmless_activations.mean(dim=0)
            self.v_minus.copy_(baseline)
            self.r_minus.copy_(baseline)  # Keep in sync for backward compat
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
            mean_diff_norm = mean_diff / (mean_diff.norm() + 1e-8)  # Normalize

            # Create k vectors with small perturbations
            for i in range(self.rank_k):
                noise = torch.randn_like(mean_diff_norm) * noise_std
                perturbed = mean_diff_norm + noise
                perturbed = perturbed / (perturbed.norm() + 1e-8)  # Normalize
                self.subspace_vectors.data[i] = perturbed

            # Also fit baseline if enabled
            if self.use_baseline:
                self.fit_baseline(harmless_activations)

    def fit_from_activations(
        self,
        harmless_activations: torch.Tensor,
        harmful_activations: torch.Tensor,
        noise_std: float = 0.01
    ):
        """
        Fit baseline and initialize subspace from activation data.

        Convenience method that calls both fit_baseline and initialize_from_mean_diff.

        Args:
            harmless_activations: [n_samples, dim]
            harmful_activations: [n_samples, dim]
            noise_std: Standard deviation of noise to add (default: 0.01)
        """
        # Fit baseline
        self.fit_baseline(harmless_activations)

        # Initialize vectors from mean diff
        with torch.no_grad():
            mean_diff = harmful_activations.mean(dim=0) - harmless_activations.mean(dim=0)
            mean_diff_norm = mean_diff / (mean_diff.norm() + 1e-8)

            for i in range(self.rank_k):
                noise = torch.randn_like(mean_diff_norm) * noise_std
                perturbed = mean_diff_norm + noise
                perturbed = perturbed / (perturbed.norm() + 1e-8)
                self.subspace_vectors.data[i] = perturbed

        print(f"  Fitted rank-{self.rank_k} ACE:")
        print(f"    ||mean_diff|| = {mean_diff.norm().item():.4f}")
        print(f"    ||v⁻|| = {self.v_minus.norm().item():.4f}")
        if not self.force_orthogonal:
            cond = self.get_condition_number()
            print(f"    Condition number: {cond:.2f}")

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
    Model wrapper with ACE (Affine Concept Editing) adapters.

    Implements the paper's formula at selected layers:
        h' = h - proj_r(h) + proj_r(r⁻) + α·r

    Key features:
    - Trainable steering direction r per layer
    - Baseline r⁻ fitted from harmless activations
    - α controls steering (0 = no refusal, 1 = full refusal)
    - Flexible layer selection (single, all, or subset)
    - Optional rank-k for complex geometries

    Examples:
        # Single layer (paper recommends middle layer ~15)
        config = UnifiedRDOConfig(layers=[15])
        model = UnifiedRDOModel(base_model, config)

        # All layers
        config = UnifiedRDOConfig(layers=None)
        model = UnifiedRDOModel(base_model, config)

        # Middle layers only
        config = UnifiedRDOConfig(layers=list(range(10, 20)))
        model = UnifiedRDOModel(base_model, config)
    """

    def __init__(self, model, config: UnifiedRDOConfig):
        super().__init__()

        self.base_model = model
        self.config = config
        self.ace_layer_indices = []  # Track which layers have ACE adapters

        # Freeze base model
        for param in model.parameters():
            param.requires_grad = False

        # Add ACE layers
        self._inject_adapters()

    def _inject_adapters(self):
        """Inject ACE adapters into selected layers."""

        # Get layers
        if hasattr(self.base_model, 'model'):
            layers = self.base_model.model.layers
        elif hasattr(self.base_model, 'transformer'):
            layers = self.base_model.transformer.h
        else:
            raise ValueError("Unsupported model architecture")

        n_layers = len(layers)

        # Get dimension
        if hasattr(self.base_model.config, 'hidden_size'):
            dim = self.base_model.config.hidden_size
        elif hasattr(self.base_model.config, 'n_embd'):
            dim = self.base_model.config.n_embd
        else:
            raise ValueError("Cannot determine hidden dimension")

        # Determine which layers to wrap
        if self.config.layers is None or self.config.layers == 'all':
            # All layers
            target_layers = list(range(n_layers))
        else:
            # Validate layer indices
            target_layers = []
            for idx in self.config.layers:
                if idx < 0:
                    idx = n_layers + idx  # Support negative indexing
                if 0 <= idx < n_layers:
                    target_layers.append(idx)
                else:
                    print(f"Warning: Layer {idx} out of range (0-{n_layers-1}), skipping")

        self.ace_layer_indices = target_layers

        # Wrap selected layers with ACE adapter
        for idx in target_layers:
            layer = layers[idx]
            if self.config.enable_rank_k:
                wrapped = RankKUnifiedLayer(
                    base_layer=layer,
                    dim=dim,
                    rank_k=self.config.rank_k,
                    alpha=self.config.alpha,
                    normalize=self.config.normalize_vectors,
                    use_baseline=self.config.use_baseline,
                    force_orthogonal=self.config.force_orthogonal,
                    per_direction_steering=self.config.per_direction_steering
                )
            else:
                wrapped = UnifiedRDOLayer(
                    base_layer=layer,
                    dim=dim,
                    alpha=self.config.alpha,
                    normalize=self.config.normalize_vectors
                )
            layers[idx] = wrapped

        layer_type = "rank-k ACE" if self.config.enable_rank_k else "ACE"
        if len(target_layers) == 1:
            print(f"✓ Added {layer_type} adapter to layer {target_layers[0]}")
        elif len(target_layers) == n_layers:
            print(f"✓ Added {layer_type} adapters to all {n_layers} layers")
        else:
            print(f"✓ Added {layer_type} adapters to {len(target_layers)} layers: {target_layers}")
        if self.config.enable_rank_k:
            print(f"  Subspace rank: {self.config.rank_k} vectors per layer")

    def forward(self, *args, **kwargs):
        """Forward pass with unified affine transformation."""
        return self.base_model(*args, **kwargs)

    def generate(self, *args, **kwargs):
        """Generation with unified affine transformation."""
        return self.base_model.generate(*args, **kwargs)

    def get_steering_parameters(self) -> List[nn.Parameter]:
        """Get trainable steering direction parameters."""
        params = []
        for module in self.modules():
            if isinstance(module, UnifiedRDOLayer):
                params.append(module.r)
            elif isinstance(module, RankKUnifiedLayer):
                params.append(module.subspace_vectors)
        return params

    def set_alpha(self, alpha: float):
        """Set steering parameter α for all layers."""
        for module in self.modules():
            if isinstance(module, (UnifiedRDOLayer, RankKUnifiedLayer)):
                module.alpha = alpha

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
        Fit ACE vectors (r, r⁻, r⁺) for ACE layers from data.

        This computes per ACE layer:
            r⁻ = mean(harmless_activations)  [baseline]
            r⁺ = mean(harmful_activations)
            r = r⁺ - r⁻  [steering direction, trainable]

        After calling this, the steering directions are initialized to the
        mean difference, and baselines are set. Training will optimize r.

        Args:
            tokenizer: Tokenizer
            harmless_prompts: List of harmless prompts
            harmful_prompts: List of harmful prompts
            batch_size: Batch size for activation collection
        """
        print("=" * 70)
        print("Fitting ACE Vectors from Data")
        if self.ace_layer_indices:
            print(f"Target layers: {self.ace_layer_indices}")
        print("=" * 70)

        # Collect activations per layer
        from collections import defaultdict
        harmless_acts = defaultdict(list)
        harmful_acts = defaultdict(list)

        # Get layers
        if hasattr(self.base_model, 'model'):
            layers = self.base_model.model.layers
        elif hasattr(self.base_model, 'transformer'):
            layers = self.base_model.transformer.h
        else:
            raise ValueError("Unsupported model architecture")

        # Register hooks to capture activations
        handles = []
        collecting_harmless = [True]  # Mutable flag

        def get_hook(layer_id):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    act = output[0][:, -1, :].detach()  # Last token
                else:
                    act = output[:, -1, :].detach()
                if collecting_harmless[0]:
                    harmless_acts[layer_id].append(act.cpu())
                else:
                    harmful_acts[layer_id].append(act.cpu())
            return hook

        for idx, layer in enumerate(layers):
            if isinstance(layer, (UnifiedRDOLayer, RankKUnifiedLayer)):
                handle = layer.base_layer.register_forward_hook(get_hook(idx))
                handles.append(handle)

        # Collect harmless activations
        print(f"\n1. Collecting activations on {len(harmless_prompts)} harmless prompts...")
        self.base_model.eval()
        with torch.no_grad():
            for i in range(0, len(harmless_prompts), batch_size):
                batch = harmless_prompts[i:i+batch_size]

                # Format with chat template
                formatted_batch = []
                for prompt in batch:
                    if hasattr(tokenizer, 'apply_chat_template'):
                        messages = [{"role": "user", "content": prompt}]
                        formatted = tokenizer.apply_chat_template(
                            messages,
                            tokenize=False,
                            add_generation_prompt=True
                        )
                    else:
                        formatted = prompt
                    formatted_batch.append(formatted)

                inputs = tokenizer(formatted_batch, return_tensors='pt', padding=True, truncation=True)
                inputs = {k: v.to(self.base_model.device) for k, v in inputs.items()}
                _ = self.base_model(**inputs)

        # Switch to harmful
        collecting_harmless[0] = False

        # Collect harmful activations
        print(f"2. Collecting activations on {len(harmful_prompts)} harmful prompts...")
        with torch.no_grad():
            for i in range(0, len(harmful_prompts), batch_size):
                batch = harmful_prompts[i:i+batch_size]

                # Format with chat template
                formatted_batch = []
                for prompt in batch:
                    if hasattr(tokenizer, 'apply_chat_template'):
                        messages = [{"role": "user", "content": prompt}]
                        formatted = tokenizer.apply_chat_template(
                            messages,
                            tokenize=False,
                            add_generation_prompt=True
                        )
                    else:
                        formatted = prompt
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
            if isinstance(layer, (UnifiedRDOLayer, RankKUnifiedLayer)) and idx in harmless_acts:
                print(f"\nLayer {layer_idx}:")

                # Stack activations
                harmless_tensor = torch.cat(harmless_acts[idx], dim=0)
                harmful_tensor = torch.cat(harmful_acts[idx], dim=0)

                # Fit ACE vectors
                layer.fit_from_activations(harmless_tensor, harmful_tensor)

                layer_idx += 1

        print("\n" + "=" * 70)
        print("✓ ACE vectors fitted!")
        print("=" * 70)
        print("\nSteering directions (r) initialized to mean difference.")
        print("Baselines (r⁻) set from harmless activations.")
        print("Ready for training!")

    def get_all_condition_numbers(self) -> List[float]:
        """
        Get condition numbers for all rank-k layers.

        Useful for monitoring numerical stability when using non-orthogonal vectors.
        High condition number (> 100) indicates near-linear dependence.

        Returns:
            List of condition numbers (one per rank-k layer)
        """
        if not self.config.enable_rank_k:
            raise ValueError("This method only works with rank-k layers (enable_rank_k=True)")

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
        """Save ACE adapters (steering directions and baselines)."""
        import os
        import json

        os.makedirs(save_directory, exist_ok=True)

        # Save config
        config_dict = {
            'peft_type': 'ACE',
            'layers': self.ace_layer_indices,  # Which layers have adapters
            'alpha': self.config.alpha,
            'enable_rank_k': self.config.enable_rank_k,
            'rank_k': self.config.rank_k,
            'normalize_vectors': self.config.normalize_vectors,
        }

        with open(os.path.join(save_directory, 'adapter_config.json'), 'w') as f:
            json.dump(config_dict, f, indent=2)

        # Save steering directions and baselines
        ace_data = []
        for module in self.modules():
            if isinstance(module, UnifiedRDOLayer):
                ace_data.append({
                    'r': module.r.data.cpu(),
                    'r_minus': module.r_minus.cpu(),
                    'r_plus': module.r_plus.cpu()
                })
            elif isinstance(module, RankKUnifiedLayer):
                ace_data.append({
                    'subspace_vectors': module.subspace_vectors.data.cpu(),
                    'r_minus': module.r_minus.cpu()
                })

        torch.save({
            'ace_data': ace_data,
            'ace_layer_indices': self.ace_layer_indices,
            'enable_rank_k': self.config.enable_rank_k
        }, os.path.join(save_directory, 'adapter_model.bin'))

        if len(self.ace_layer_indices) == 1:
            print(f"✓ Saved ACE adapter (layer {self.ace_layer_indices[0]}) to {save_directory}")
        else:
            print(f"✓ Saved ACE adapters ({len(self.ace_layer_indices)} layers) to {save_directory}")

    @classmethod
    def from_pretrained(cls, model, adapter_path: str):
        """Load ACE adapters onto a model."""
        import os
        import json

        # Load config
        with open(os.path.join(adapter_path, 'adapter_config.json'), 'r') as f:
            config_dict = json.load(f)

        config = UnifiedRDOConfig(**config_dict)

        # Create ACE model
        ace_model = cls(model, config)

        # Load data
        state = torch.load(
            os.path.join(adapter_path, 'adapter_model.bin'),
            map_location='cpu'
        )

        ace_data = state['ace_data']

        # Apply to layers
        layer_idx = 0
        for module in ace_model.modules():
            if isinstance(module, UnifiedRDOLayer):
                data = ace_data[layer_idx]
                module.r.data = data['r'].to(module.r.device)
                module.r_minus.copy_(data['r_minus'].to(module.r_minus.device))
                module.r_plus.copy_(data['r_plus'].to(module.r_plus.device))
                layer_idx += 1
            elif isinstance(module, RankKUnifiedLayer):
                data = ace_data[layer_idx]
                module.subspace_vectors.data = data['subspace_vectors'].to(module.subspace_vectors.device)
                module.r_minus.copy_(data['r_minus'].to(module.r_minus.device))
                layer_idx += 1

        print(f"✓ Loaded ACE adapters from {adapter_path}")

        return ace_model


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
    Example: ACE (Affine Concept Editing) adapters for training.
    """

    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("=" * 70)
    print("ACE Adapters - Paper Formula Implementation")
    print("=" * 70)

    print("\nACE Formula (Equation 5):")
    print("  h' = h - proj_r(h) + proj_r(r⁻) + α·r")
    print("\nWhere:")
    print("  r   = steering direction (trainable)")
    print("  r⁻  = baseline (mean harmless activations)")
    print("  α   = 0: no refusal, 1: full refusal")

    # Load model
    print("\n" + "-" * 70)
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        'gpt2',
        torch_dtype=torch.float32
    )
    tokenizer = AutoTokenizer.from_pretrained('gpt2')

    print(f"  Model: GPT-2")
    print(f"  Layers: {len(model.transformer.h)}")

    # Test 1: Standard ACE
    print("\n" + "-" * 70)
    print("Test 1: ACE Adapters (rank-1)")
    print("-" * 70)

    config = UnifiedRDOConfig(
        target_modules=["layers"],
        alpha=0.0,  # Start with no refusal (ablation mode)
    )

    model_ace = get_unified_rdo_model(model, config)
    model_ace.print_trainable_parameters()

    # Test forward
    inputs = tokenizer("Test prompt", return_tensors='pt')

    with torch.no_grad():
        output = model_ace(**inputs)
    print(f"  ✓ ACE forward works, shape: {output.logits.shape}")

    # Test alpha switching
    print("\n  Testing α switching:")
    model_ace.set_alpha(0.0)
    print(f"    α=0.0 (ablation mode)")
    model_ace.set_alpha(1.0)
    print(f"    α=1.0 (full refusal mode)")
    model_ace.set_alpha(0.5)
    print(f"    α=0.5 (halfway)")

    # Test 2: Rank-k ACE
    print("\n" + "-" * 70)
    print("Test 2: Rank-k ACE (for complex geometry)")
    print("-" * 70)

    config_rank_k = UnifiedRDOConfig(
        target_modules=["layers"],
        alpha=0.0,
        enable_rank_k=True,
        rank_k=3  # 3-dimensional subspace
    )

    model_rank_k = get_unified_rdo_model(model, config_rank_k)
    model_rank_k.print_trainable_parameters()

    with torch.no_grad():
        output = model_rank_k(**inputs)
    print(f"  ✓ Rank-k ACE works, shape: {output.logits.shape}")

    # Summary
    print("\n" + "=" * 70)
    print("Summary: ACE Training Workflow")
    print("=" * 70)

    print("\n1. Create ACE model:")
    print("   model = get_unified_rdo_model(base_model, config)")

    print("\n2. Fit baselines from data:")
    print("   model.fit_all_baselines(tokenizer, harmless_prompts, harmful_prompts)")

    print("\n3. Train with different α for harmful/harmless:")
    print("   - α=0 on harmful prompts (want compliance)")
    print("   - α=1 on harmless prompts (preserve behavior)")

    print("\n4. Save/load adapters:")
    print("   model.save_pretrained('ace_adapters/')")
    print("   model = UnifiedRDOModel.from_pretrained(base_model, 'ace_adapters/')")

    print("\n" + "=" * 70)
    print("✓ ACE adapters ready for training!")
    print("=" * 70)
