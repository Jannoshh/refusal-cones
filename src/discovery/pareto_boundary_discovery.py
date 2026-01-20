#!/usr/bin/env python3
"""
Multi-Objective Pareto Boundary Discovery for Refusal Geometry

Instead of finding boundary of a single R(v), finds the Pareto frontier
in the (refusal_score, kl_score) space - vectors that optimally trade off
refusal ablation vs behavior preservation.

Key insight: We want vectors that:
1. Ablate refusal effectively (low refusal_score)
2. Preserve normal behavior (low kl_score)
3. Can induce refusal on harmless prompts under intervention (high induce_score)

The Pareto frontier contains all "optimal" trade-offs.

## Theoretical Background: Refusal as Affine Function

Refusal behavior is modeled as an affine function of layer activations:

    R(h) = w · h + b

where w is the "refusal direction" and b is a bias term. When we ablate
by projecting out a direction v from activations:

    h' = h - α(h · v̂)v̂    where v̂ = v/||v||, α = ||v|| (scale)

The refusal score becomes:

    R(h') = w · h - α(w · v̂)(h · v̂) + b

Key observations:
1. **Direction matters**: (w · v̂) measures alignment with refusal direction
2. **Scale matters**: α controls how much we project out
3. **Per-layer norms encode importance**: Larger ||v_layer|| = stronger ablation at that layer

## Why Per-Layer Norms Matter

The mean-difference method computes:

    v_layer = mean(h_harmful) - mean(h_harmless)

This gives natural per-layer norms:
- Layers where harmful ≠ harmless get larger ||v|| → more ablation
- Layers where harmful ≈ harmless get smaller ||v|| → less ablation

When sampling/optimizing, we should NOT normalize per-layer (which forces
||v_layer|| = 1 for all layers), as this destroys the layer importance signal.
Instead, use global normalization (total ||v|| = 1) to preserve relative
layer importance while keeping vectors comparable.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Callable, List, Dict, Optional, Tuple, Union
from dataclasses import dataclass, field

from .adaptive_geometry_discovery import SimpleGP, AdaptiveSparseGP, StructuredLayerGP


class SmoothLayerParameterization:
    """
    Parameterize layer vectors v as smooth functions of layer index.

    Instead of optimizing v ∈ R^{n_layers × hidden_dim} directly, we optimize
    basis coefficients z ∈ R^{n_basis × hidden_dim} where:

        v(layer_i) = Σ_k z_k * φ_k(layer_i)

    and φ_k are RBF basis functions centered at different layers.

    This enforces smoothness by construction:
    - Adjacent layers automatically have similar directions
    - Reduces effective dimensionality from n_layers to n_basis
    - Matches the smoothness prior in StructuredLayerGP

    Theoretical motivation:
    - If the GP assumes layer smoothness (K_layer[i,j] = exp(-(i-j)²/2σ²)),
      then optimizing in the span of smooth basis functions ensures we stay
      in regions where the GP's predictions are reliable.
    """

    def __init__(
        self,
        n_layers: int,
        hidden_dim: int,
        n_basis: int = 8,
        lengthscale: float = 3.0,
        device: str = 'cpu'
    ):
        """
        Initialize smooth parameterization.

        Args:
            n_layers: Number of model layers
            hidden_dim: Hidden dimension per layer
            n_basis: Number of basis functions (controls expressiveness vs smoothness)
                     Typical: 6-12 for 28-layer models
            lengthscale: RBF lengthscale (should match StructuredLayerGP.layer_lengthscale)
            device: Torch device
        """
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.n_basis = n_basis
        self.lengthscale = lengthscale
        self.device = device

        # RBF basis centers spread across layers
        self.centers = torch.linspace(0, n_layers - 1, n_basis, device=device)

        # Precompute basis matrix [n_layers, n_basis]
        # basis[i, k] = φ_k(layer_i) = exp(-(i - c_k)² / 2σ²)
        idx = torch.arange(n_layers, dtype=torch.float32, device=device)
        self.basis = torch.exp(
            -(idx.unsqueeze(1) - self.centers.unsqueeze(0)) ** 2
            / (2 * lengthscale ** 2)
        )

        # Normalize columns so each basis function has unit L2 norm
        self.basis = self.basis / (self.basis.norm(dim=0, keepdim=True) + 1e-8)

        # Precompute pseudoinverse for v_to_z projection
        # basis_pinv @ v gives least-squares z
        self.basis_pinv = torch.linalg.pinv(self.basis)

    def z_to_v(self, z: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        """
        Convert basis coefficients to layer vectors.

        Args:
            z: Basis coefficients [n_basis, hidden_dim]
            normalize: Whether to normalize result to unit norm

        Returns:
            v: Layer vectors [n_layers, hidden_dim]
        """
        # v = basis @ z: [n_layers, n_basis] @ [n_basis, hidden_dim] -> [n_layers, hidden_dim]
        # Match dtype of basis to input z
        z_device = z.to(self.device)
        v = self.basis.to(z_device.dtype) @ z_device

        if normalize:
            v = v / (v.norm() + 1e-8)

        return v

    def v_to_z(self, v: torch.Tensor) -> torch.Tensor:
        """
        Project layer vectors to basis coefficients (least squares).

        This finds z that minimizes ||basis @ z - v||²

        Args:
            v: Layer vectors [n_layers, hidden_dim]

        Returns:
            z: Basis coefficients [n_basis, hidden_dim]
        """
        # z = basis_pinv @ v: [n_basis, n_layers] @ [n_layers, hidden_dim] -> [n_basis, hidden_dim]
        # Match dtype of basis to input v
        v_device = v.to(self.device)
        return self.basis_pinv.to(v_device.dtype) @ v_device

    def random_z(self, scale: float = 1.0) -> torch.Tensor:
        """
        Sample random basis coefficients.

        Args:
            scale: Scale of random initialization

        Returns:
            z: Random coefficients [n_basis, hidden_dim]
        """
        z = torch.randn(self.n_basis, self.hidden_dim, device=self.device) * scale
        return z

    def gradient_z_to_v(
        self,
        grad_v: torch.Tensor,
        z: Optional[torch.Tensor] = None,
        v: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Convert gradient w.r.t. v to gradient w.r.t. z via chain rule.

        Since z_to_v normalizes: v = (basis @ z) / ||basis @ z||,
        the gradient must include the normalization Jacobian.

        Let u = basis @ z (unnormalized), v = u / ||u|| (normalized).
        The Jacobian of normalization is: ∂v/∂u = (I - vv^T) / ||u||

        So: ∂L/∂u = (∂L/∂v - (∂L/∂v · v) * v) / ||u||
        And: ∂L/∂z = basis.T @ ∂L/∂u

        Args:
            grad_v: Gradient w.r.t. v [n_layers, hidden_dim]
            z: Current basis coefficients [n_basis, hidden_dim] (needed for ||u||)
            v: Current normalized vector [n_layers, hidden_dim] (for tangent projection)
               If not provided, computed from z

        Returns:
            grad_z: Gradient w.r.t. z [n_basis, hidden_dim]
        """
        grad_device = grad_v.to(self.device)
        basis = self.basis.to(grad_device.dtype)

        # If z and v not provided, fall back to simple chain rule (no normalization)
        # This maintains backward compatibility but is less accurate
        if z is None:
            return basis.T @ grad_device

        z_device = z.to(self.device).to(grad_device.dtype)

        # Compute unnormalized vector and its norm
        u = basis @ z_device  # [n_layers, hidden_dim]
        u_norm = u.norm() + 1e-8

        # Get normalized v (compute if not provided)
        if v is None:
            v = u / u_norm
        else:
            v = v.to(self.device).to(grad_device.dtype)

        # Project gradient to tangent space of sphere at v
        # This is the Jacobian of normalization: (I - vv^T) @ grad_v / ||u||
        grad_dot_v = (grad_device * v).sum()
        grad_tangent = (grad_device - grad_dot_v * v) / u_norm

        # Chain rule through basis transformation
        grad_z = basis.T @ grad_tangent

        return grad_z

    def smoothness_of_v(self, v: torch.Tensor) -> float:
        """
        Measure how well v fits the smooth basis (reconstruction error).

        Lower = more smooth (better fit to basis).

        Args:
            v: Layer vectors [n_layers, hidden_dim]

        Returns:
            error: Relative reconstruction error
        """
        z = self.v_to_z(v)
        v_reconstructed = self.z_to_v(z, normalize=False)

        # Relative error
        error = (v - v_reconstructed).norm() / (v.norm() + 1e-8)
        return error.item()


@dataclass
class ParetoDiscoveryConfig:
    """Configuration for multi-objective Pareto discovery."""

    # Objectives (all measured, some used for Pareto)
    primary_objectives: List[str] = field(
        default_factory=lambda: ['refusal_score', 'kl_score']
    )
    secondary_objectives: List[str] = field(
        default_factory=lambda: ['induce_score']
    )

    # Sampling
    n_init_samples: int = 30
    n_pareto_iterations: int = 70
    n_candidates: int = 500
    max_measurements: int = 200

    # Acquisition
    acquisition_type: str = 'ehvi'  # Expected Hypervolume Improvement
    beta: float = 0.5  # For UCB-based acquisition (lower = more exploitation)
    # Reference point: worst-case values for each objective
    # refusal_score: 0 is worst (no ablation effect), -10 is good
    # kl_score: high is worst (e.g., 10), 0 is good
    reference_point: List[float] = field(
        default_factory=lambda: [0.0, 10.0]  # [refusal_worst, kl_worst]
    )

    # Convergence
    convergence_threshold: float = 0.01  # Hypervolume improvement threshold
    min_pareto_points: int = 10

    # GP settings (one per objective)
    gp_type: str = 'simple'  # 'simple', 'sparse', or 'structured'
    kernel_type: str = 'rbf'
    kernel_lengthscale: float = 0.3
    use_sparse_gp: bool = False  # Deprecated, use gp_type='sparse'
    num_inducing: int = 64
    sparse_train_steps: int = 10
    sparse_lr: float = 0.05

    # Structured GP settings (gp_type='structured')
    layer_lengthscale: float = 3.0  # Smoothness across ~3 adjacent layers
    learn_layer_weights: bool = True  # ARD: learn per-layer importance

    # Normalization
    normalize_per_layer: bool = False  # If False, let optimizer learn layer norms

    # Gradient-based candidate generation (more efficient in high dimensions)
    use_gradient_candidates: bool = True  # Use gradient descent to generate candidates
    gradient_lr: float = 0.1  # Learning rate for gradient descent
    gradient_steps: int = 5  # Number of gradient steps per candidate
    n_gradient_candidates: int = 20  # Number of gradient-based candidates per iteration
    gradient_weight_diversity: int = 5  # Number of different scalarization weights to try

    # Smooth layer parameterization (enforces GP-consistent smoothness)
    use_smooth_parameterization: bool = True  # Optimize in smooth basis space
    n_basis: int = 8  # Number of RBF basis functions (6-12 typical for 28 layers)
    # Note: uses layer_lengthscale from Structured GP settings above


@dataclass
class AffineRefusalVector:
    """
    A complete affine refusal vector following ACE (Marshall et al., 2024).

    The ACE intervention is: h' = h - proj_v(h) + proj_v(v⁻) + α*v

    Attributes:
        v: [n_layers, hidden_dim] - The refusal direction (ablation direction)
        v_minus: [n_layers, hidden_dim] - Reference point (mean harmless activations)
        v_plus: [n_layers, hidden_dim] - Mean harmful activations (optional, for analysis)
    """
    v: torch.Tensor  # The direction to ablate
    v_minus: torch.Tensor  # Reference point for ACE bias term
    v_plus: Optional[torch.Tensor] = None  # Mean harmful (for analysis)

    def to_dict(self) -> Dict[str, torch.Tensor]:
        """Convert to dictionary for serialization."""
        d = {'v': self.v, 'v_minus': self.v_minus}
        if self.v_plus is not None:
            d['v_plus'] = self.v_plus
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, torch.Tensor]) -> 'AffineRefusalVector':
        """Create from dictionary."""
        return cls(
            v=d['v'],
            v_minus=d['v_minus'],
            v_plus=d.get('v_plus')
        )


@dataclass
class ParetoDiscoveryResults:
    """Results from multi-objective Pareto discovery."""

    # Pareto frontier
    pareto_vectors: torch.Tensor  # [n_pareto, n_layers, hidden_dim]
    pareto_scores: Dict[str, torch.Tensor]  # {objective: [n_pareto]}

    # All observations
    V_observed: torch.Tensor
    scores_observed: Dict[str, torch.Tensor]

    # Dominated points (inside Pareto frontier)
    dominated_vectors: torch.Tensor
    dominated_scores: Dict[str, torch.Tensor]

    # Hypervolume of Pareto frontier
    hypervolume: float

    # GPs for each objective
    gps: Dict[str, Union[SimpleGP, AdaptiveSparseGP]]

    n_measurements: int

    # ACE reference points (shared across all vectors in this run)
    v_minus: Optional[torch.Tensor] = None  # [n_layers, hidden_dim] mean harmless
    v_plus: Optional[torch.Tensor] = None   # [n_layers, hidden_dim] mean harmful

    def get_pareto_affine_vectors(self) -> List[AffineRefusalVector]:
        """Get Pareto vectors as full AffineRefusalVector objects."""
        if self.v_minus is None:
            raise ValueError("v_minus not available - was discovery run with ACE?")

        return [
            AffineRefusalVector(
                v=self.pareto_vectors[i],
                v_minus=self.v_minus,
                v_plus=self.v_plus
            )
            for i in range(len(self.pareto_vectors))
        ]


class MultiObjectiveScorer:
    """
    Compute multiple objective scores efficiently using forward passes only.

    NO GENERATION NEEDED - just:
    1. Forward pass on harmful prompts -> refusal token logits OR
       Forward pass on harmful prompt+completion -> completion loss
    2. Forward pass on harmless prompts -> KL from baseline

    Time per measurement: ~0.2-0.5s (vs ~5-10s with generation)
    """

    def __init__(
        self,
        model,
        tokenizer,
        harmful_prompts: List[str],
        harmless_prompts: List[str],
        refusal_toks: torch.Tensor,
        device: str = 'cuda',
        harmful_completions: Optional[List[str]] = None,
        n_score_tokens: int = 1,
        harmless_completions: Optional[List[str]] = None,
        n_kl_tokens: int = 1,
        generate_completions: bool = False,
        max_new_tokens: int = 30,
        generation_batch_size: int = 32,
    ):
        """
        Initialize the multi-objective scorer.

        Args:
            model: HuggingFace model
            tokenizer: Tokenizer
            harmful_prompts: Prompts for refusal measurement
            harmless_prompts: Prompts for retain measurement
            refusal_toks: Token IDs indicating refusal
            device: Device to use
            harmful_completions: Optional harmful completions to compute loss on.
                If provided, loss is computed on completion tokens instead of
                just the next-token refusal logits.
            n_score_tokens: Number of completion tokens to compute loss on.
                Only used when harmful_completions is provided. Set to -1 to
                use all completion tokens.
            harmless_completions: Optional precomputed completions for KL loss.
                If not provided and generate_completions=True, will auto-generate.
            n_kl_tokens: Number of completion tokens to compute KL over (default 1).
                Set to -1 for all completion tokens. Only used when
                harmless_completions is provided.
            generate_completions: If True and harmless_completions not provided,
                auto-generate completions by running baseline model.
            max_new_tokens: Number of tokens to generate for completions (default 30).
            generation_batch_size: Batch size for completion generation (default 32).
        """
        self.model = model
        self.tokenizer = tokenizer
        self.harmful_prompts = harmful_prompts
        self.harmless_prompts = harmless_prompts
        self.refusal_toks = refusal_toks.to(device)
        self.device = device
        self.harmful_completions = harmful_completions
        self.n_score_tokens = n_score_tokens
        self.n_kl_tokens = n_kl_tokens
        self.max_new_tokens = max_new_tokens
        self.generation_batch_size = generation_batch_size

        # Handle harmless completions
        if harmless_completions is not None:
            self.harmless_completions = harmless_completions
        elif generate_completions:
            print(f"Generating {len(harmless_prompts)} harmless completions "
                  f"(max_new_tokens={max_new_tokens}, batch_size={generation_batch_size})...")
            self.harmless_completions = self._generate_completions(
                harmless_prompts, max_new_tokens, batch_size=generation_batch_size
            )
            print(f"Generated completions for KL scoring")
        else:
            self.harmless_completions = None

        # Pre-tokenize prompts for speed
        self._prepare_inputs()

        # Cache baseline logits for KL computation
        self._cache_baselines()

        # ACE reference point (v⁻) - computed lazily
        self._v_minus: Optional[torch.Tensor] = None
        self._v_plus: Optional[torch.Tensor] = None
        self._v_minus_computed: bool = False

    def _ensure_v_minus_cached(self) -> torch.Tensor:
        """
        Lazily compute and cache the ACE reference point v⁻.

        v⁻ is the mean harmless activations, used as the reference point
        in the ACE formula: h' = h - proj_v(h) + proj_v(v⁻) + α*v
        """
        if not self._v_minus_computed:
            # Use compute_mean_diff_vector which returns (v, v_minus, v_plus)
            _, v_minus, v_plus = self.compute_mean_diff_vector()
            self._v_minus = v_minus
            self._v_plus = v_plus
            self._v_minus_computed = True
        return self._v_minus

    def _generate_completions(
        self,
        prompts: List[str],
        max_new_tokens: int,
        batch_size: int = 32
    ) -> List[str]:
        """
        Generate completions for prompts using the baseline model.

        Used to precompute retain targets for KL loss, following the RDO
        approach where KL is computed on model's own completions.

        Args:
            prompts: List of prompts to complete
            max_new_tokens: Maximum tokens to generate per prompt
            batch_size: Batch size for generation (default 32, increase for faster generation)

        Returns:
            List of completion strings (without the original prompt)
        """
        completions = []

        self.model.eval()
        with torch.no_grad():
            for i in range(0, len(prompts), batch_size):
                batch_prompts = prompts[i:i + batch_size]
                inputs = self.tokenizer(
                    batch_prompts,
                    return_tensors='pt',
                    padding=True,
                    truncation=True,
                    max_length=256
                ).to(self.device)

                # Generate completions
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,  # Greedy for reproducibility
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )

                # Extract just the completion (remove prompt)
                for j, output in enumerate(outputs):
                    prompt_len = inputs['attention_mask'][j].sum().item()
                    completion_tokens = output[prompt_len:]
                    completion = self.tokenizer.decode(completion_tokens, skip_special_tokens=True)
                    completions.append(completion)

        return completions

    def _format_prompts(self, prompts: List[str]) -> List[str]:
        """Apply chat template if available, otherwise return prompts as-is."""
        if self.tokenizer.chat_template is not None:
            return [
                self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": p}],
                    tokenize=False,
                    add_generation_prompt=True
                )
                for p in prompts
            ]
        return prompts

    def _prepare_inputs(self):
        """Pre-tokenize prompts for faster scoring."""
        # Format prompts with chat template for instruction-tuned models
        formatted_harmful = self._format_prompts(self.harmful_prompts)
        formatted_harmless = self._format_prompts(self.harmless_prompts)

        self.harmful_inputs = self.tokenizer(
            formatted_harmful,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.device)

        self.harmless_inputs = self.tokenizer(
            formatted_harmless,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.device)

        # If harmful completions provided, tokenize prompt+completion
        # and track where the completion starts for each example
        if self.harmful_completions is not None:
            self._prepare_completion_inputs()

        # If harmless completions provided, tokenize prompt+completion for KL
        if self.harmless_completions is not None:
            self._prepare_harmless_completion_inputs()

    def _prepare_completion_inputs(self):
        """Prepare inputs for multi-token completion scoring."""
        # Format prompts with chat template
        formatted_harmful = self._format_prompts(self.harmful_prompts)

        # Tokenize prompts alone to get prompt lengths (without tensor conversion)
        prompt_encodings = self.tokenizer(
            formatted_harmful,
            padding=False,
            truncation=True,
            max_length=256,
            add_special_tokens=False  # Keep prompt lengths aligned with full-text inputs
        )
        self.prompt_lengths = [len(ids) for ids in prompt_encodings['input_ids']]

        # Tokenize prompt + completion together
        full_texts = [
            p + c for p, c in zip(formatted_harmful, self.harmful_completions)
        ]
        self.harmful_full_inputs = self.tokenizer(
            full_texts,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=512,
            add_special_tokens=False
        ).to(self.device)

        # Create labels for completion tokens only (mask out prompt tokens)
        self.completion_labels = self.harmful_full_inputs['input_ids'].clone()
        for i, prompt_len in enumerate(self.prompt_lengths):
            # Mask prompt tokens with -100 (ignored in loss)
            self.completion_labels[i, :prompt_len] = -100

        # Determine how many completion tokens to score per example
        completion_lengths = [
            (self.completion_labels[i] != -100).sum().item()
            for i in range(len(self.harmful_prompts))
        ]
        self.completion_lengths = completion_lengths

        # If n_score_tokens is set, limit the tokens we score
        if self.n_score_tokens > 0:
            for i in range(len(self.harmful_prompts)):
                prompt_len = self.prompt_lengths[i]
                # Keep only first n_score_tokens of completion
                mask_start = prompt_len + self.n_score_tokens
                if mask_start < self.completion_labels.size(1):
                    self.completion_labels[i, mask_start:] = -100

    def _prepare_harmless_completion_inputs(self):
        """Prepare inputs for multi-token KL scoring on harmless completions."""
        # Format prompts with chat template
        formatted_harmless = self._format_prompts(self.harmless_prompts)

        # Tokenize prompts alone to get prompt lengths (without tensor conversion)
        prompt_encodings = self.tokenizer(
            formatted_harmless,
            padding=False,
            truncation=True,
            max_length=256,
            add_special_tokens=False
        )
        self.harmless_prompt_lengths = [len(ids) for ids in prompt_encodings['input_ids']]

        # Tokenize prompt + completion together
        full_texts = [
            p + c for p, c in zip(formatted_harmless, self.harmless_completions)
        ]
        self.harmless_full_inputs = self.tokenizer(
            full_texts,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=512,
            add_special_tokens=False
        ).to(self.device)

        # Track which positions are completion tokens (for KL computation)
        # We'll compute KL on these positions
        self.harmless_completion_mask = torch.zeros_like(
            self.harmless_full_inputs['input_ids'], dtype=torch.bool
        )
        for i, prompt_len in enumerate(self.harmless_prompt_lengths):
            seq_len = self.harmless_full_inputs['attention_mask'][i].sum().item()
            # Mark completion positions (after prompt, before padding)
            if self.n_kl_tokens > 0:
                # Only mark first n_kl_tokens of completion
                end_pos = min(prompt_len + self.n_kl_tokens, seq_len)
            else:
                # Mark all completion tokens
                end_pos = seq_len
            self.harmless_completion_mask[i, prompt_len:end_pos] = True

        n_kl_positions = self.harmless_completion_mask.sum().item()
        print(f"KL will be computed over {n_kl_positions} total positions "
              f"(~{n_kl_positions / len(self.harmless_prompts):.1f} tokens/prompt)")

    def _last_nonpad_logits(self, outputs, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Select logits at each sequence's last non-padding token."""
        logits = outputs.logits
        attention_mask = inputs.get('attention_mask')
        if attention_mask is None:
            return logits[:, -1, :]

        mask = attention_mask.to(device=logits.device, dtype=torch.long)
        seq_len = mask.size(1)
        positions = torch.arange(seq_len, device=logits.device).unsqueeze(0)
        last_idx = (mask * positions).max(dim=1).values
        batch_idx = torch.arange(logits.size(0), device=logits.device)
        return logits[batch_idx, last_idx, :]

    def _cache_baselines(self):
        """Cache baseline (no intervention) logits for KL computation."""
        with torch.no_grad():
            # Baseline harmful logits (for reference)
            outputs = self.model(**self.harmful_inputs)
            self.baseline_harmful_logits = self._last_nonpad_logits(outputs, self.harmful_inputs).detach()

            # Baseline harmless logits (for KL)
            if self.harmless_completions is not None:
                # Multi-token KL: cache all logits for completion positions
                outputs = self.model(**self.harmless_full_inputs)
                # Store full logits tensor for masked KL computation
                self.baseline_harmless_full_logits = outputs.logits.detach()
                # Also cache single-token for backward compatibility
                self.baseline_harmless_logits = self._last_nonpad_logits(outputs, self.harmless_full_inputs).detach()
            else:
                # Single-token KL: just cache last position
                outputs = self.model(**self.harmless_inputs)
                self.baseline_harmless_logits = self._last_nonpad_logits(outputs, self.harmless_inputs).detach()
                self.baseline_harmless_full_logits = None

    def compute_baseline_refusal_scores(self) -> Tuple[float, float]:
        """
        Compute baseline refusal scores (without any ablation).

        This is used to set dynamic thresholds - the boundary should be where
        ablation starts to have a meaningful effect, relative to the gap between
        refusal and non-refusal baselines.

        Returns:
            Tuple of (harmful_baseline, harmless_baseline):
            - harmful_baseline: Mean refusal score on harmful prompts (model refuses, positive)
            - harmless_baseline: Mean refusal score on harmless prompts (model doesn't refuse, negative)
        """
        from ..measurement.scoring import refusal_score_fn

        harmful_scores = refusal_score_fn(
            self.baseline_harmful_logits,
            self.refusal_toks
        )
        harmless_scores = refusal_score_fn(
            self.baseline_harmless_logits,
            self.refusal_toks
        )
        return harmful_scores.mean().item(), harmless_scores.mean().item()

    def compute_mean_diff_vector(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute refusal direction and reference points via mean-difference method.

        Following "Refusal in LLMs is an Affine Function" (Marshall et al., 2024):
        - r = v⁺ - v⁻ (refusal direction with natural norm)
        - v⁻ = mean harmless activations (reference point for ACE)
        - v⁺ = mean harmful activations

        The ACE intervention for ablating with direction v is:
            h' = h - proj_v(h) + proj_v(v⁻) + α*v

        Where proj_v(v⁻) is the bias term that keeps activations in a sensible region.

        Returns:
            r: [n_layers, hidden_dim] mean-diff refusal direction (v⁺ - v⁻) with natural norms
            v_minus: [n_layers, hidden_dim] mean harmless activations (reference point)
            v_plus: [n_layers, hidden_dim] mean harmful activations
        """
        # Get layers
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            layers = self.model.model.layers
        elif hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'h'):
            layers = self.model.transformer.h
        else:
            raise ValueError("Unknown model architecture")

        n_layers = len(layers)
        hidden_dim = self._infer_hidden_dim(layers)

        # Collect activations at last token position
        harmful_acts = {i: [] for i in range(n_layers)}
        harmless_acts = {i: [] for i in range(n_layers)}

        def make_collector(storage, layer_idx):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    h = output[0]
                else:
                    h = output
                # Get last token activation (simplified - assumes no padding issues)
                storage[layer_idx].append(h[:, -1, :].detach())
            return hook

        # Collect harmful activations
        handles = []
        for i, layer in enumerate(layers):
            handles.append(layer.register_forward_hook(make_collector(harmful_acts, i)))

        with torch.no_grad():
            self.model(**self.harmful_inputs)

        for h in handles:
            h.remove()

        # Collect harmless activations
        handles = []
        for i, layer in enumerate(layers):
            handles.append(layer.register_forward_hook(make_collector(harmless_acts, i)))

        with torch.no_grad():
            self.model(**self.harmless_inputs)

        for h in handles:
            h.remove()

        # Compute means per layer
        v_plus_layers = []  # Mean harmful (refusal)
        v_minus_layers = []  # Mean harmless (non-refusal) - reference point
        r_layers = []  # Difference (refusal direction)

        for i in range(n_layers):
            h_harmful = torch.cat(harmful_acts[i], dim=0).mean(dim=0)  # [hidden_dim]
            h_harmless = torch.cat(harmless_acts[i], dim=0).mean(dim=0)
            v_plus_layers.append(h_harmful)
            v_minus_layers.append(h_harmless)
            r_layers.append(h_harmful - h_harmless)

        r = torch.stack(r_layers)  # [n_layers, hidden_dim] - mean-diff refusal direction
        v_minus = torch.stack(v_minus_layers)  # [n_layers, hidden_dim] - reference point
        v_plus = torch.stack(v_plus_layers)  # [n_layers, hidden_dim]

        # Log per-layer norms (these encode layer importance!)
        norms = r.norm(dim=1)
        print(f"Mean-diff vector (r = v⁺ - v⁻) computed. Per-layer norms:")
        print(f"  Min: {norms.min():.4f}, Max: {norms.max():.4f}, Mean: {norms.mean():.4f}")
        print(f"  Top 5 layers by norm: {norms.argsort(descending=True)[:5].tolist()}")

        return r, v_minus, v_plus

    def _infer_hidden_dim(self, layers) -> Optional[int]:
        """Infer hidden size from model config or layer norm weights."""
        if hasattr(self.model, 'config'):
            for attr in ('hidden_size', 'n_embd', 'd_model'):
                value = getattr(self.model.config, attr, None)
                if value is not None:
                    return int(value)

        if layers:
            for attr in ('input_layernorm', 'ln_1', 'layernorm', 'norm'):
                layer_norm = getattr(layers[0], attr, None)
                if layer_norm is not None and hasattr(layer_norm, 'weight'):
                    return int(layer_norm.weight.shape[0])

        return None

    def _apply_ablation_hooks(
        self,
        v: torch.Tensor,
        v_minus: Optional[torch.Tensor] = None,
        alpha: float = 0.0
    ) -> List:
        """
        Register ACE (Affine Concept Editing) hooks for vector v.

        Implements the ACE formula from Marshall et al. (2024):
            h' = h - proj_v(h) + proj_v(v⁻) + α*v

        Where:
            - v = the ablation direction
            - v⁻ = reference point (mean harmless activations)
            - α = steering parameter (0 = ablate refusal, 1 = induce refusal)
            - proj_v(x) = (x·v̂)v̂ where v̂ = v/||v||

        Args:
            v: Ablation direction [n_layers, hidden_dim]
            v_minus: Reference point [n_layers, hidden_dim] (mean harmless activations)
                     If None, falls back to simple directional ablation (no bias term).
            alpha: Steering parameter. 0 = ablate refusal, 1 = induce refusal.
        """
        handles = []

        # Determine number of layers
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            layers = self.model.model.layers
        elif hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'h'):
            layers = self.model.transformer.h
        else:
            raise ValueError("Unknown model architecture")

        n_layers = len(layers)
        model_hidden_dim = self._infer_hidden_dim(layers)

        # Reshape or broadcast v to [n_layers, hidden_dim]
        if v.dim() == 1:
            if model_hidden_dim is not None:
                if v.numel() == model_hidden_dim:
                    v_per_layer = v.unsqueeze(0).repeat(n_layers, 1)
                elif v.numel() == n_layers * model_hidden_dim:
                    v_per_layer = v.reshape(n_layers, model_hidden_dim)
                else:
                    raise ValueError(
                        f"1D v must be length {model_hidden_dim} (shared) or "
                        f"{n_layers * model_hidden_dim} (flattened), got {v.numel()}."
                    )
            else:
                if v.numel() % n_layers != 0:
                    raise ValueError(
                        f"1D v length {v.numel()} is not divisible by n_layers={n_layers}."
                    )
                hidden_dim = v.numel() // n_layers
                v_per_layer = v.reshape(n_layers, hidden_dim)
        else:
            if v.shape[0] != n_layers:
                raise ValueError(
                    f"Expected v with {n_layers} layers, got shape {tuple(v.shape)}."
                )
            if model_hidden_dim is not None and v.shape[1] != model_hidden_dim:
                raise ValueError(
                    f"Expected hidden_dim {model_hidden_dim}, got {v.shape[1]}."
                )
            v_per_layer = v

        # Infer model dtype from model parameters (robust to parameterless layers like Identity)
        try:
            model_dtype = next(self.model.parameters()).dtype
        except StopIteration:
            model_dtype = torch.float32  # Default fallback

        for layer_idx, layer in enumerate(layers):
            # Get this layer's refusal direction r
            r_layer = v_per_layer[layer_idx].to(device=self.device, dtype=model_dtype)

            # Skip layers with zero (or near-zero) direction - no ablation needed
            r_norm = r_layer.norm()
            if r_norm < 1e-6:
                continue

            # Compute unit vector r̂ = r / ||r||
            r_unit = r_layer / (r_norm + 1e-8)

            # Get reference point v⁻ for this layer (if provided)
            if v_minus is not None:
                v_minus_layer = v_minus[layer_idx].to(device=self.device, dtype=model_dtype)
                # Compute proj_v(v⁻) = (v⁻·v̂)v̂
                bias = torch.dot(v_minus_layer, r_unit) * r_unit
            else:
                bias = torch.zeros_like(r_unit)

            def make_hook(r_u, r_l, b, a):
                def hook(module, input, output):
                    if isinstance(output, tuple):
                        h = output[0]
                    else:
                        h = output

                    # ACE: h' = h - proj_r(h) + proj_r(r⁻) + α*r
                    # = h - (h·r̂)r̂ + bias + α*r
                    proj_h = torch.einsum('...d,d->...', h, r_u)  # h·r̂
                    h_ace = h - torch.einsum('...,d->...d', proj_h, r_u)  # h - proj_r(h)
                    h_ace = h_ace + b  # + proj_r(r⁻) (bias term)
                    h_ace = h_ace + a * r_l  # + α*r (steering term)

                    if isinstance(output, tuple):
                        return (h_ace,) + output[1:]
                    return h_ace
                return hook

            handles.append(layer.register_forward_hook(make_hook(r_unit, r_layer, bias, alpha)))

        return handles

    def _subsample_inputs(self, inputs: dict, n_samples: int) -> dict:
        """Subsample a batch of tokenized inputs to n_samples."""
        return {k: v[:n_samples] for k, v in inputs.items()}

    def score(
        self,
        v: torch.Tensor,
        return_grad: bool = False,
        fidelity: float = 1.0
    ) -> Union[Dict[str, float], Tuple[Dict[str, float], torch.Tensor]]:
        """
        Compute all objective scores for vector v.

        Just 2 forward passes - no generation needed!

        Args:
            v: Direction vector [n_layers, hidden_dim] or [hidden_dim]
            return_grad: Whether to return gradient w.r.t. v
            fidelity: Fraction of prompts to use (0-1). Lower = faster but noisier.
                      Use fidelity < 1 for cheap screening, fidelity = 1 for final evaluation.

        Returns:
            scores: Dict with 'refusal_score', 'induce_score', 'kl_score'
                - refusal_score: logit(P_refusal) = log(P_r) - log(1-P_r)
                  Negative when ablation works (e.g., -6 is good)
                - kl_score: KL divergence on harmless (lower = better)
                - induce_score: refusal logit on harmless prompts under ablation
            grad: (optional) Gradient of combined loss w.r.t. v
        """
        from ..measurement.scoring import refusal_score_fn

        scores = {}

        # Flatten if needed
        v_flat = v.reshape(-1) if v.dim() > 1 else v
        v_flat = v_flat.to(self.device)

        if return_grad:
            v_flat = v_flat.clone().detach().requires_grad_(True)

        # Subsample inputs for low-fidelity evaluation
        if fidelity < 1.0:
            n_harmful = max(1, int(len(self.harmful_prompts) * fidelity))
            n_harmless = max(1, int(len(self.harmless_prompts) * fidelity))
            harmful_inputs = self._subsample_inputs(self.harmful_inputs, n_harmful)
            harmless_inputs = self._subsample_inputs(self.harmless_inputs, n_harmless)
            # Also subsample baseline logits for KL
            baseline_harmless_logits = self.baseline_harmless_logits[:n_harmless]
        else:
            harmful_inputs = self.harmful_inputs
            harmless_inputs = self.harmless_inputs
            baseline_harmless_logits = self.baseline_harmless_logits

        # Get ACE reference point (cached)
        v_minus = self._ensure_v_minus_cached()

        # Register ACE ablation hooks (alpha=0 for ablation)
        handles = self._apply_ablation_hooks(v_flat, v_minus=v_minus, alpha=0.0)

        try:
            # === Forward pass 1: Harmful prompts (refusal score) ===
            # Always compute refusal_score using logit-based method (first token)
            # This gives consistent scores that can be negative when ablation works
            if return_grad:
                outputs = self.model(**harmful_inputs)
            else:
                with torch.no_grad():
                    outputs = self.model(**harmful_inputs)

            ablated_harmful_logits = self._last_nonpad_logits(outputs, harmful_inputs)

            # Refusal score: logit(P_refusal) = log(P_r) - log(1 - P_r)
            # Negative when ablation works well (e.g., -6 to -14)
            # Positive when model still refuses (e.g., +5)
            refusal_logits = refusal_score_fn(
                ablated_harmful_logits,
                self.refusal_toks
            )
            scores['refusal_score'] = refusal_logits.mean().item()

            # === Optional: Multi-token compliance loss (if completions provided) ===
            if self.harmful_completions is not None:
                # Additional forward pass for multi-token scoring
                if return_grad:
                    outputs_full = self.model(**self.harmful_full_inputs)
                else:
                    with torch.no_grad():
                        outputs_full = self.model(**self.harmful_full_inputs)

                # Compute cross-entropy loss on completion tokens
                logits = outputs_full.logits[:, :-1, :].contiguous()
                labels = self.completion_labels[:, 1:].contiguous()

                loss_fct = nn.CrossEntropyLoss(reduction='none', ignore_index=-100)
                per_token_loss = loss_fct(
                    logits.view(-1, logits.size(-1)),
                    labels.view(-1)
                ).view(logits.size(0), logits.size(1))

                valid_mask = (labels != -100).float()
                per_example_loss = (per_token_loss * valid_mask).sum(dim=1) / (valid_mask.sum(dim=1) + 1e-8)

                # ablation_loss (RDO paper): CE loss on harmful completion under ablation
                # Lower = model generates harmful content more easily = ablation working
                scores['ablation_loss'] = per_example_loss.mean().item()

            # === Forward pass 2: Harmless prompts (KL score + retain loss) ===
            # kl_score: Always single-token at last position (used for GP optimization)
            # retain_loss: Multi-token over completions (for evaluation, if completions provided)

            if return_grad:
                outputs = self.model(**harmless_inputs)
            else:
                with torch.no_grad():
                    outputs = self.model(**harmless_inputs)

            ablated_harmless_logits = self._last_nonpad_logits(outputs, harmless_inputs)

            # kl_score: Single-token KL at last position (for GP)
            log_p_baseline = torch.log_softmax(baseline_harmless_logits.float(), dim=-1)
            log_p_ablated = torch.log_softmax(ablated_harmless_logits.float(), dim=-1)
            p_baseline = log_p_baseline.exp()
            kl = (p_baseline * (log_p_baseline - log_p_ablated)).sum(dim=-1).mean()
            kl = torch.clamp(kl, min=0.0, max=100.0)
            scores['kl_score'] = kl.item()

            # retain_loss: Multi-token KL over completions (if provided)
            if self.harmless_completions is not None:
                if return_grad:
                    outputs_full = self.model(**self.harmless_full_inputs)
                else:
                    with torch.no_grad():
                        outputs_full = self.model(**self.harmless_full_inputs)

                ablated_logits = outputs_full.logits  # [batch, seq, vocab]

                # Compute KL only at completion positions (masked)
                log_p_baseline_full = torch.log_softmax(self.baseline_harmless_full_logits.float(), dim=-1)
                log_p_ablated_full = torch.log_softmax(ablated_logits.float(), dim=-1)
                p_baseline_full = log_p_baseline_full.exp()

                # Per-position KL
                kl_per_pos = (p_baseline_full * (log_p_baseline_full - log_p_ablated_full)).sum(dim=-1)

                # Average only over completion positions
                mask = self.harmless_completion_mask.to(kl_per_pos.device)
                retain_loss = (kl_per_pos * mask).sum() / mask.sum().clamp(min=1)
                retain_loss = torch.clamp(retain_loss, min=0.0, max=100.0)
                scores['retain_loss'] = retain_loss.item()

            # Induce score: refusal logit on harmless prompts under ablation
            induce_logits = refusal_score_fn(
                ablated_harmless_logits,
                self.refusal_toks
            )
            scores['induce_score'] = induce_logits.mean().item()

            # Compute gradient if requested
            if return_grad:
                # For gradient computation, use CE loss (smooth, differentiable)
                # NOT the logit-based refusal_score (not good for gradients)
                if self.harmful_completions is not None:
                    # Use ablation_loss (CE on completions) for gradient - smooth & differentiable
                    loss = per_example_loss.mean() + kl
                else:
                    # Fallback: use refusal logits (less ideal but works)
                    loss = refusal_logits.mean() + kl
                loss.backward()
                grad = v_flat.grad.clone()
                return scores, grad.reshape(v.shape)

        finally:
            # Clean up hooks
            for handle in handles:
                handle.remove()

        return scores

    def score_batch(self, vs: torch.Tensor) -> List[Dict[str, float]]:
        """Score multiple vectors (sequential, hooks don't batch well)."""
        return [self.score(v) for v in vs]


class ParetoGeometryDiscovery:
    """
    Multi-objective Pareto boundary discovery.

    Finds vectors on the Pareto frontier of (refusal_score, kl_score),
    representing optimal trade-offs between refusal ablation and
    behavior preservation.
    """

    def __init__(
        self,
        scorer: MultiObjectiveScorer,
        n_layers: int,
        hidden_dim: int,
        config: Optional[ParetoDiscoveryConfig] = None,
        v_init: Optional[torch.Tensor] = None,
        use_mean_diff_init: bool = True
    ):
        self.scorer = scorer
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.config = config or ParetoDiscoveryConfig()

        # Use mean-diff initialization if no v_init provided
        if v_init is None and use_mean_diff_init:
            print("Computing mean-diff initialization (natural per-layer norms)...")
            # compute_mean_diff_vector returns (r, v_minus, v_plus)
            # r = v_plus - v_minus is the mean-diff refusal direction
            r, v_minus, v_plus = scorer.compute_mean_diff_vector()
            self.v_init = r  # Use refusal direction as initial vector
            # Cache v_minus/v_plus in the scorer for ACE
            scorer._v_minus = v_minus
            scorer._v_plus = v_plus
            scorer._v_minus_computed = True
        else:
            self.v_init = v_init

        # Observations
        self.V_observed: List[torch.Tensor] = []
        self.scores_observed: Dict[str, List[float]] = {
            'refusal_score': [],
            'induce_score': [],
            'kl_score': [],
            'ablation_loss': [],
            'retain_loss': [],
        }

        # One GP per objective
        self.gps: Dict[str, Union[SimpleGP, AdaptiveSparseGP, StructuredLayerGP]] = {}

        # Resolve GP type (handle legacy use_sparse_gp flag)
        gp_type = self.config.gp_type
        if self.config.use_sparse_gp and gp_type == 'simple':
            gp_type = 'sparse'

        for obj in self.config.primary_objectives + self.config.secondary_objectives:
            if gp_type == 'structured':
                self.gps[obj] = StructuredLayerGP(
                    n_layers=n_layers,
                    hidden_dim=hidden_dim,
                    feature_lengthscale=self.config.kernel_lengthscale,
                    layer_lengthscale=self.config.layer_lengthscale,
                    learn_layer_weights=self.config.learn_layer_weights
                )
            elif gp_type == 'sparse':
                self.gps[obj] = AdaptiveSparseGP(
                    kernel_type=self.config.kernel_type,
                    lengthscale=self.config.kernel_lengthscale,
                    num_inducing=self.config.num_inducing,
                    train_steps=self.config.sparse_train_steps,
                    lr=self.config.sparse_lr
                )
            else:
                self.gps[obj] = SimpleGP(
                    kernel_type=self.config.kernel_type,
                    lengthscale=self.config.kernel_lengthscale
                )

        self.gp_type = gp_type

        # Initialize smooth layer parameterization if enabled
        # Derive device from v_init to avoid device mismatches with scorer
        if self.config.use_smooth_parameterization:
            device = self.v_init.device if self.v_init is not None else 'cpu'
            self.smooth_param = SmoothLayerParameterization(
                n_layers=n_layers,
                hidden_dim=hidden_dim,
                n_basis=self.config.n_basis,
                lengthscale=self.config.layer_lengthscale,
                device=device
            )
        else:
            self.smooth_param = None

    def discover(self) -> ParetoDiscoveryResults:
        """Run multi-objective Pareto discovery."""

        print("=" * 70)
        print("Multi-Objective Pareto Boundary Discovery")
        print("=" * 70)
        print(f"\nObjectives: {self.config.primary_objectives}")
        print(f"Secondary: {self.config.secondary_objectives}")
        print(f"GP type: {self.gp_type}")
        if self.config.use_gradient_candidates:
            print(f"Candidate generation: GRADIENT-BASED (efficient in {self.n_layers * self.hidden_dim} dims)")
            print(f"  - Gradient steps: {self.config.gradient_steps}, lr: {self.config.gradient_lr}")
            print(f"  - Candidates per iter: {self.config.n_gradient_candidates} gradient + random exploration")
            if self.config.use_smooth_parameterization:
                effective_dim = self.config.n_basis * self.hidden_dim
                full_dim = self.n_layers * self.hidden_dim
                print(f"  - Smooth parameterization: {self.config.n_basis} basis functions "
                      f"(effective dim: {effective_dim}, reduction: {full_dim/effective_dim:.1f}x)")
        else:
            print(f"Candidate generation: Random sampling ({self.config.n_candidates} candidates)")

        # Phase 1: Initial sampling
        print("\n" + "=" * 70)
        print(f"Phase 1: Initial Exploration ({self.config.n_init_samples} samples)")
        print("=" * 70)
        self._initial_exploration()

        # Phase 2: Pareto frontier search
        print("\n" + "=" * 70)
        print(f"Phase 2: Pareto Frontier Search ({self.config.n_pareto_iterations} iterations)")
        print("=" * 70)
        self._pareto_search()

        # Phase 3: Extract Pareto frontier
        print("\n" + "=" * 70)
        print("Phase 3: Extract Pareto Frontier")
        print("=" * 70)
        results = self._extract_pareto()

        return results

    def _normalize_vector(self, v: torch.Tensor) -> torch.Tensor:
        """Normalize vector according to config."""
        if self.config.normalize_per_layer:
            # Unit norm per layer
            return v / (v.norm(dim=-1, keepdim=True) + 1e-8)
        else:
            # Single global unit norm (preserves relative layer importance)
            return v / (v.norm() + 1e-8)

    def _riemannian_gradient_step(
        self,
        v: torch.Tensor,
        grad: torch.Tensor,
        lr: float
    ) -> torch.Tensor:
        """
        Take a Riemannian gradient step on the unit sphere.

        For optimization on the hypersphere S^{d-1}, we:
        1. Project gradient to tangent space: grad_t = grad - (grad·v)v
        2. Take step in tangent direction: v' = v - lr * grad_t
        3. Retract back to sphere: v'' = v' / ||v'||

        Args:
            v: Current vector on sphere [n_layers, hidden_dim]
            grad: Euclidean gradient of objective [n_layers, hidden_dim]
            lr: Learning rate

        Returns:
            Updated vector on sphere
        """
        v_flat = v.reshape(-1)
        grad_flat = grad.reshape(-1)

        # Project gradient to tangent space (remove component along v)
        grad_tangent = grad_flat - torch.dot(grad_flat, v_flat) * v_flat

        # Take step (we're minimizing, so subtract)
        v_new = v_flat - lr * grad_tangent

        # Retract to sphere
        v_new = v_new / (v_new.norm() + 1e-8)

        return v_new.reshape(v.shape)

    def _generate_gradient_candidates(self) -> List[torch.Tensor]:
        """
        Generate candidates using gradient descent on scalarized objectives.

        This is much more efficient than random sampling in high dimensions
        (e.g., 28 layers × 1024 = 28,672 dims).

        Strategy:
        1. Start from current Pareto points (or mean-diff init)
        2. Use different scalarization weights for diversity
        3. Run gradient descent for a few steps
        4. Return the final points as candidates

        If use_smooth_parameterization is enabled:
        - Optimize in z-space (basis coefficients) instead of v-space
        - This enforces layer smoothness consistent with GP prior
        - Reduces effective dimensionality from n_layers to n_basis

        Returns:
            List of candidate vectors
        """
        candidates = []
        pareto_mask = self._get_pareto_mask()
        pareto_indices = torch.where(pareto_mask)[0].tolist() if pareto_mask.any() else []

        # If no Pareto points yet, use mean-diff init
        if len(pareto_indices) == 0:
            if self.v_init is not None:
                starting_points = [self.v_init.clone()]
            else:
                # Random starting points
                starting_points = [
                    self._normalize_vector(torch.randn(self.n_layers, self.hidden_dim))
                    for _ in range(min(3, self.config.n_gradient_candidates))
                ]
        else:
            # Use Pareto points as starting points
            starting_points = [self.V_observed[idx].clone() for idx in pareto_indices]

        # Generate diverse scalarization weights
        # We want to explore different trade-offs between refusal and KL
        n_weights = self.config.gradient_weight_diversity
        weights = []
        for i in range(n_weights):
            # Linear spacing from refusal-focused to KL-focused
            w_refusal = i / (n_weights - 1) if n_weights > 1 else 0.5
            weights.append((w_refusal, 1 - w_refusal))

        # Generate candidates from each starting point with each weight
        n_per_start = max(1, self.config.n_gradient_candidates // len(starting_points))

        # Use smooth parameterization if enabled
        use_smooth = self.config.use_smooth_parameterization and self.smooth_param is not None

        oom_occurred = False
        for v_start in starting_points[:self.config.n_gradient_candidates]:
            if oom_occurred:
                # After OOM, just use perturbations instead of gradients
                v = v_start + 0.2 * torch.randn_like(v_start)
                v = self._normalize_vector(v)
                candidates.append(v.detach().cpu())
                continue

            for w_refusal, w_kl in weights[:n_per_start]:
                if use_smooth:
                    # === Smooth parameterization: optimize in z-space ===
                    # Convert v to basis coefficients z
                    z = self.smooth_param.v_to_z(v_start)

                    # Run gradient descent in z-space
                    for step in range(self.config.gradient_steps):
                        try:
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()

                            # Convert z to v for scoring
                            v = self.smooth_param.z_to_v(z, normalize=True)

                            # Get gradient w.r.t. v
                            scores, grad_v = self.scorer.score(v, return_grad=True)

                            # Convert gradient to z-space via chain rule
                            # Includes normalization Jacobian: ∂L/∂z = basis.T @ (I - vv^T) @ ∂L/∂v / ||u||
                            grad_z = self.smooth_param.gradient_z_to_v(grad_v, z=z, v=v)

                            # Add noise for diversity based on weight
                            if w_refusal < 0.3:
                                grad_z = grad_z + 0.1 * torch.randn_like(grad_z)
                            elif w_refusal > 0.7:
                                grad_z = grad_z + 0.05 * torch.randn_like(grad_z)

                            # Simple gradient descent in z-space (no Riemannian needed)
                            # The smoothness constraint is built into the parameterization
                            z = z - self.config.gradient_lr * grad_z

                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                        except RuntimeError as e:
                            if "out of memory" in str(e).lower():
                                if torch.cuda.is_available():
                                    torch.cuda.empty_cache()
                                oom_occurred = True
                                break
                            raise

                    # Convert final z back to v
                    v_final = self.smooth_param.z_to_v(z, normalize=True)
                    candidates.append(v_final.detach().cpu())
                else:
                    # === Original: optimize directly in v-space ===
                    v = v_start.clone()

                    # Run gradient descent
                    for step in range(self.config.gradient_steps):
                        try:
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()

                            # Get gradient of scalarized objective
                            scores, grad = self.scorer.score(v, return_grad=True)

                            # Add noise for diversity based on weight
                            if w_refusal < 0.3:
                                grad = grad + 0.1 * torch.randn_like(grad)
                            elif w_refusal > 0.7:
                                grad = grad + 0.05 * torch.randn_like(grad)

                            # Riemannian gradient step
                            v = self._riemannian_gradient_step(v, grad, self.config.gradient_lr)

                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                        except RuntimeError as e:
                            if "out of memory" in str(e).lower():
                                if torch.cuda.is_available():
                                    torch.cuda.empty_cache()
                                oom_occurred = True
                                break
                            raise

                    candidates.append(v.detach().cpu())

                # Early exit if we have enough candidates
                if len(candidates) >= self.config.n_gradient_candidates:
                    break

            if len(candidates) >= self.config.n_gradient_candidates:
                break

        return candidates

    def _initial_exploration(self):
        """
        Phase 1: Initial exploration.

        If gradient_candidates is enabled, use gradient descent from mean-diff
        vector to find a good initial Pareto region. Otherwise, use stratified
        random sampling.
        """
        samples_collected = 0

        # If gradient-based, first do gradient descent from v_init
        if self.config.use_gradient_candidates and self.v_init is not None:
            print("  Using gradient descent from mean-diff initialization...")

            # Do gradient descent with different scalarization weights
            # to find initial Pareto points
            v = self.v_init.clone()
            v = self._normalize_vector(v)

            # First, score the initial point
            scores = self.scorer.score(v)
            self.V_observed.append(v.detach().cpu())  # Store on CPU for consistency
            for obj, val in scores.items():
                self.scores_observed[obj].append(val)
            samples_collected += 1
            print(f"  Init (mean-diff): refusal={scores['refusal_score']:.4f}, "
                  f"kl={scores['kl_score']:.4f}")

            # Now do gradient descent with varying weights to explore Pareto front
            # Use fewer trajectories (5) to reduce memory pressure
            n_gradient_init = min(5, self.config.n_init_samples // 5)
            weights = [(i / (n_gradient_init - 1), 1 - i / (n_gradient_init - 1))
                       for i in range(n_gradient_init)] if n_gradient_init > 1 else [(0.5, 0.5)]

            # Use smooth parameterization if enabled
            use_smooth = self.config.use_smooth_parameterization and self.smooth_param is not None
            if use_smooth:
                print(f"  Using smooth parameterization ({self.config.n_basis} basis functions)")

            oom_count = 0
            for w_idx, (w_r, w_k) in enumerate(weights):
                if oom_count >= 2:
                    # Too many OOM errors, skip remaining gradient trajectories
                    print("  Skipping remaining gradient trajectories due to OOM...")
                    break

                if use_smooth:
                    # === Smooth parameterization: optimize in z-space ===
                    z = self.smooth_param.v_to_z(self.v_init)

                    for step in range(self.config.gradient_steps):
                        try:
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()

                            # Convert z to v for scoring
                            v = self.smooth_param.z_to_v(z, normalize=True)

                            # Get gradient w.r.t. v
                            scores, grad_v = self.scorer.score(v, return_grad=True)

                            # Convert gradient to z-space (includes normalization Jacobian)
                            grad_z = self.smooth_param.gradient_z_to_v(grad_v, z=z, v=v)

                            # Simple gradient descent in z-space
                            z = z - self.config.gradient_lr * grad_z

                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                        except RuntimeError as e:
                            if "out of memory" in str(e).lower():
                                if torch.cuda.is_available():
                                    torch.cuda.empty_cache()
                                print(f"  Warning: OOM during gradient step {step}, stopping trajectory...")
                                oom_count += 1
                                break
                            raise

                    # Convert final z to v
                    v = self.smooth_param.z_to_v(z, normalize=True)
                else:
                    # === Original: optimize directly in v-space ===
                    v = self.v_init.clone()
                    v = self._normalize_vector(v)

                    # Gradient steps for initial exploration
                    for step in range(self.config.gradient_steps):
                        try:
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()

                            scores, grad = self.scorer.score(v, return_grad=True)
                            v = self._riemannian_gradient_step(v, grad, self.config.gradient_lr)

                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                        except RuntimeError as e:
                            if "out of memory" in str(e).lower():
                                if torch.cuda.is_available():
                                    torch.cuda.empty_cache()
                                print(f"  Warning: OOM during gradient step {step}, stopping trajectory...")
                                oom_count += 1
                                break
                            raise

                # Store final point (without gradients)
                with torch.no_grad():
                    scores = self.scorer.score(v)
                self.V_observed.append(v.detach().cpu())  # Store on CPU
                for obj, val in scores.items():
                    self.scores_observed[obj].append(val)
                samples_collected += 1

                if w_idx % 2 == 0:
                    print(f"  Gradient init {w_idx+1}: refusal={scores['refusal_score']:.4f}, "
                          f"kl={scores['kl_score']:.4f}")

        # Fill remaining samples with stratified random
        remaining = self.config.n_init_samples - samples_collected
        for i in range(remaining):
            # Sample direction
            if self.v_init is not None and i < remaining // 3:
                v = self.v_init + 0.5 * torch.randn_like(self.v_init)
            else:
                v = torch.randn(self.n_layers, self.hidden_dim)

            v = self._normalize_vector(v)

            # Score
            scores = self.scorer.score(v)

            # Store on CPU for consistency
            self.V_observed.append(v.detach().cpu())
            for obj, val in scores.items():
                self.scores_observed[obj].append(val)

            if i % 10 == 0:
                print(f"  Sample {samples_collected + i + 1}: refusal={scores['refusal_score']:.4f}, "
                      f"kl={scores['kl_score']:.4f}")

        # Fit GPs
        self._update_gps()

    def _pareto_search(self):
        """Phase 2: Search for Pareto frontier using EHVI acquisition."""

        for iteration in range(self.config.n_pareto_iterations):
            if len(self.V_observed) >= self.config.max_measurements:
                print(f"\n  Reached max measurements ({self.config.max_measurements})")
                break

            # Generate candidates
            candidates = self._generate_candidates()

            # Compute acquisition (Expected Hypervolume Improvement)
            acquisition = self._compute_ehvi_acquisition(candidates)

            # Select best
            best_idx = acquisition.argmax()
            v_next = candidates[best_idx]

            # Score
            scores = self.scorer.score(v_next)

            # Store on CPU for consistency
            self.V_observed.append(v_next.detach().cpu())
            for obj, val in scores.items():
                self.scores_observed[obj].append(val)

            # Update GPs
            self._update_gps()

            if iteration % 10 == 0:
                hv = self._compute_hypervolume()
                print(f"  Iter {iteration}: refusal={scores['refusal_score']:.4f}, "
                      f"kl={scores['kl_score']:.4f}, HV={hv:.4f}")

    def _generate_candidates(self) -> torch.Tensor:
        """
        Generate candidate directions using a mix of strategies.

        When gradient_candidates is enabled (default), uses efficient gradient
        descent in high-dimensional space. Also includes random exploration.
        """
        candidates = []

        # Strategy 1: Gradient-based candidates (efficient in high dimensions)
        if self.config.use_gradient_candidates:
            gradient_candidates = self._generate_gradient_candidates()
            candidates.extend(gradient_candidates)

            # Reduce random candidates when using gradients
            n_random = max(50, self.config.n_candidates // 10)
        else:
            # Original behavior: mostly random
            n_random = self.config.n_candidates // 2

        # Strategy 2: Random uniform on sphere (exploration)
        n_uniform = n_random // 2
        for _ in range(n_uniform):
            v = torch.randn(self.n_layers, self.hidden_dim)
            v = self._normalize_vector(v)
            candidates.append(v)

        # Strategy 3: Near current Pareto points (local refinement)
        n_near_pareto = n_random - n_uniform
        pareto_mask = self._get_pareto_mask()
        pareto_indices = torch.where(pareto_mask)[0]

        if len(pareto_indices) > 0:
            for _ in range(n_near_pareto):
                idx = pareto_indices[torch.randint(len(pareto_indices), (1,))].item()
                v_base = self.V_observed[idx]
                v = v_base + 0.2 * torch.randn_like(v_base)
                v = self._normalize_vector(v)
                candidates.append(v)
        else:
            for _ in range(n_near_pareto):
                v = torch.randn(self.n_layers, self.hidden_dim)
                v = self._normalize_vector(v)
                candidates.append(v)

        return torch.stack(candidates)

    def _compute_ehvi_acquisition(self, candidates: torch.Tensor) -> torch.Tensor:
        """
        Compute Expected Hypervolume Improvement acquisition.

        Simplified version: Use scalarized UCB with varying weights.
        Full EHVI is expensive; this approximation works well in practice.
        """
        V_flat = candidates.reshape(len(candidates), -1)

        # Get predictions for each objective
        mus = {}
        sigmas = {}
        for obj in self.config.primary_objectives:
            mu, sigma = self.gps[obj].predict(V_flat)
            mus[obj] = mu
            sigmas[obj] = sigma

        # Scalarized acquisition with Pareto-aware weighting
        # Lower is better for both objectives, so we want to minimize
        # Use negative UCB: -mu + beta * sigma (explore low regions)

        # Random scalarization for diversity
        n_candidates = len(candidates)
        weights = torch.rand(n_candidates, len(self.config.primary_objectives))
        weights = weights / weights.sum(dim=1, keepdim=True)

        acquisition = torch.zeros(n_candidates)
        for i, obj in enumerate(self.config.primary_objectives):
            # Lower is better: use negative of mu, plus exploration
            obj_acq = -mus[obj] + self.config.beta * sigmas[obj]
            acquisition += weights[:, i] * obj_acq

        return acquisition

    def _get_pareto_mask(self) -> torch.Tensor:
        """Get mask of points on Pareto frontier."""
        n = len(self.V_observed)
        if n == 0:
            return torch.tensor([])

        # Get scores for primary objectives
        scores = torch.stack([
            torch.tensor(self.scores_observed[obj])
            for obj in self.config.primary_objectives
        ], dim=1)  # [n, n_objectives]

        # Point i is Pareto if no other point dominates it
        # Point j dominates i if j is <= i in all objectives and < in at least one
        pareto_mask = torch.ones(n, dtype=torch.bool)

        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                # Check if j dominates i
                if (scores[j] <= scores[i]).all() and (scores[j] < scores[i]).any():
                    pareto_mask[i] = False
                    break

        return pareto_mask

    def _compute_hypervolume(self) -> float:
        """
        Compute hypervolume of current Pareto frontier.

        Both objectives are minimized:
        - refusal_score: lower (more negative) is better
        - kl_score: lower is better

        Hypervolume = area dominated by Pareto points relative to reference.
        """
        pareto_mask = self._get_pareto_mask()

        if not pareto_mask.any():
            return 0.0

        # Get Pareto scores
        pareto_scores = torch.stack([
            torch.tensor(self.scores_observed[obj])[pareto_mask]
            for obj in self.config.primary_objectives
        ], dim=1)

        # Reference point: worst-case values
        # refusal_score=0 (no ablation), kl_score=10 (high divergence)
        ref = torch.tensor(self.config.reference_point)

        # For minimization: hypervolume is area between Pareto front and reference
        # Sort by first objective (ascending - lower refusal is better)
        sorted_idx = pareto_scores[:, 0].argsort()
        sorted_scores = pareto_scores[sorted_idx]

        # Compute area using sweeping
        hv = 0.0
        prev_y = ref[1]  # Start from worst KL

        for point in sorted_scores:
            # Width: from this point's refusal to reference refusal
            width = ref[0] - point[0]
            # Height: from previous y to this point's KL
            height = prev_y - point[1]

            if width > 0 and height > 0:
                hv += width * height

            prev_y = min(prev_y, point[1])

        return hv

    def _update_gps(self):
        """Update all GPs with current observations."""
        if len(self.V_observed) == 0:
            return

        V_flat = torch.stack(self.V_observed).reshape(len(self.V_observed), -1)

        for obj in self.config.primary_objectives + self.config.secondary_objectives:
            scores = torch.tensor(self.scores_observed[obj])
            self.gps[obj].fit(V_flat, scores)

    def _extract_pareto(self) -> ParetoDiscoveryResults:
        """Extract final Pareto frontier."""

        V_tensor = torch.stack(self.V_observed)
        pareto_mask = self._get_pareto_mask()
        dominated_mask = ~pareto_mask
        n_observed = len(self.V_observed)

        def _aligned_scores(mask: torch.Tensor) -> Dict[str, torch.Tensor]:
            """Return scores for objectives aligned to V_observed length."""
            aligned: Dict[str, torch.Tensor] = {}
            for obj, vals in self.scores_observed.items():
                if len(vals) != n_observed:
                    # Skip objectives that were never recorded (e.g., no completions)
                    continue
                aligned[obj] = torch.tensor(vals)[mask]
            return aligned

        # Pareto vectors and scores
        pareto_vectors = V_tensor[pareto_mask]
        pareto_scores = _aligned_scores(pareto_mask)

        # Dominated vectors and scores
        dominated_vectors = V_tensor[dominated_mask]
        dominated_scores = _aligned_scores(dominated_mask)

        # All scores
        all_scores = {
            obj: torch.tensor(vals)
            for obj, vals in self.scores_observed.items()
            if len(vals) == n_observed
        }

        hv = self._compute_hypervolume()

        print(f"\n  Pareto frontier: {len(pareto_vectors)} vectors")
        print(f"  Dominated: {len(dominated_vectors)} vectors")
        print(f"  Hypervolume: {hv:.4f}")

        # Print Pareto points
        print("\n  Pareto vectors (refusal, kl):")
        for i in range(min(5, len(pareto_vectors))):
            r = pareto_scores['refusal_score'][i].item()
            k = pareto_scores['kl_score'][i].item()
            print(f"    {i+1}. refusal={r:.4f}, kl={k:.4f}")

        # Get ACE reference points from scorer (if available)
        v_minus = getattr(self.scorer, '_v_minus', None)
        v_plus = getattr(self.scorer, '_v_plus', None)

        return ParetoDiscoveryResults(
            pareto_vectors=pareto_vectors,
            pareto_scores=pareto_scores,
            V_observed=V_tensor,
            scores_observed=all_scores,
            dominated_vectors=dominated_vectors,
            dominated_scores=dominated_scores,
            hypervolume=hv,
            gps=self.gps,
            n_measurements=len(self.V_observed),
            v_minus=v_minus,
            v_plus=v_plus
        )

    def get_vector_for_tradeoff(self, refusal_weight: float = 0.5) -> torch.Tensor:
        """
        Get a vector from Pareto frontier for desired trade-off.

        Args:
            refusal_weight: Weight for refusal (0=prioritize retain, 1=prioritize ablation)

        Returns:
            v: Best vector for the specified trade-off
        """
        pareto_mask = self._get_pareto_mask()
        pareto_indices = torch.where(pareto_mask)[0]

        if len(pareto_indices) == 0:
            # Return best by refusal
            refusal_scores = torch.tensor(self.scores_observed['refusal_score'])
            return self.V_observed[refusal_scores.argmin()]

        # Scalarize objectives
        refusal_scores = torch.tensor(self.scores_observed['refusal_score'])[pareto_mask]
        kl_scores = torch.tensor(self.scores_observed['kl_score'])[pareto_mask]

        # Normalize to [0, 1]
        r_norm = (refusal_scores - refusal_scores.min()) / (refusal_scores.max() - refusal_scores.min() + 1e-8)
        k_norm = (kl_scores - kl_scores.min()) / (kl_scores.max() - kl_scores.min() + 1e-8)

        # Weighted sum (lower is better)
        combined = refusal_weight * r_norm + (1 - refusal_weight) * k_norm
        best_pareto_idx = combined.argmin()

        return torch.stack(self.V_observed)[pareto_mask][best_pareto_idx]


# Convenience function for quick usage
def discover_pareto_boundary(
    model,
    tokenizer,
    harmful_prompts: List[str],
    harmless_prompts: List[str],
    refusal_toks: torch.Tensor,
    n_layers: int,
    hidden_dim: int,
    v_init: Optional[torch.Tensor] = None,
    config: Optional[ParetoDiscoveryConfig] = None,
    harmful_completions: Optional[List[str]] = None,
    n_score_tokens: int = 1
) -> ParetoDiscoveryResults:
    """
    Discover Pareto frontier of refusal vectors.

    Args:
        model: HuggingFace model
        tokenizer: Tokenizer
        harmful_prompts: Prompts for refusal measurement
        harmless_prompts: Prompts for retain measurement
        refusal_toks: Token IDs indicating refusal
        n_layers: Number of model layers
        hidden_dim: Hidden dimension
        v_init: Initial vector (optional)
        config: Discovery configuration
        harmful_completions: Optional harmful completions to compute loss on.
            If provided, loss is computed on completion tokens instead of
            just the next-token refusal logits.
        n_score_tokens: Number of completion tokens to compute loss on.
            Only used when harmful_completions is provided. Set to -1 to
            use all completion tokens. Default: 1

    Returns:
        ParetoDiscoveryResults with Pareto-optimal vectors
    """
    scorer = MultiObjectiveScorer(
        model=model,
        tokenizer=tokenizer,
        harmful_prompts=harmful_prompts,
        harmless_prompts=harmless_prompts,
        refusal_toks=refusal_toks,
        harmful_completions=harmful_completions,
        n_score_tokens=n_score_tokens
    )

    discovery = ParetoGeometryDiscovery(
        scorer=scorer,
        n_layers=n_layers,
        hidden_dim=hidden_dim,
        config=config,
        v_init=v_init
    )

    return discovery.discover()


@dataclass
class BoundaryThenParetoConfig:
    """Configuration for two-phase boundary + Pareto discovery."""

    # Phase 1: Boundary discovery
    # Dynamic threshold: boundary = baseline_refusal - fraction * (baseline_refusal - baseline_harmless)
    # This places the boundary at `fraction` of the way from refusal to non-refusal
    use_dynamic_threshold: bool = True  # Compute threshold from baseline
    boundary_fraction: float = 0.05  # Fraction of gap from refusal to non-refusal (5% = just starting to work)
    boundary_threshold: float = 0.0  # Fixed threshold (used if use_dynamic_threshold=False)
    n_boundary_init: int = 20  # Initial samples for boundary
    n_boundary_iterations: int = 30  # Iterations to find boundary
    boundary_beta: float = 1.96  # Exploration parameter for straddle

    # Phase 2: Pareto search inside boundary
    n_pareto_init: int = 10  # Initial samples inside boundary
    n_pareto_iterations: int = 50  # Pareto iterations
    pareto_beta: float = 0.5  # UCB exploration for Pareto

    # Shared settings
    max_measurements: int = 150
    gp_type: str = 'structured'
    layer_lengthscale: float = 3.0
    kernel_lengthscale: float = 0.3

    # Gradient settings
    use_gradient_candidates: bool = True
    gradient_steps: int = 5
    gradient_lr: float = 0.1

    # Smooth parameterization
    use_smooth_parameterization: bool = True
    n_basis: int = 8

    # Multi-fidelity optimization
    # Screen many candidates cheaply (few prompts), then evaluate best with full prompts
    # Cost comparison (8 harmful + 8 harmless prompts):
    #   Single fidelity: 1 candidate × 16 prompts = 16 prompt-passes
    #   Multi-fidelity:  8 screen × 4 prompts + 1 promote × 16 = 48 prompt-passes
    # BUT multi-fidelity evaluates 8× more candidates, so better exploration per iteration
    use_multi_fidelity: bool = False  # Disabled by default (enable for better exploration at ~3× cost)
    low_fidelity_fraction: float = 0.25  # Use 25% of prompts for screening (e.g., 2 of 8)
    n_screen_candidates: int = 8  # Number of candidates to screen with low fidelity
    n_promote: int = 1  # Number of candidates to promote to high-fidelity evaluation


@dataclass
class BoundaryThenParetoResults:
    """Results from two-phase discovery."""

    # Phase 1 results
    boundary_points: torch.Tensor  # Points near threshold
    inside_points: torch.Tensor  # Points where ablation works (refusal < threshold)
    outside_points: torch.Tensor  # Points where ablation fails

    # Phase 2 results (Pareto frontier INSIDE boundary)
    pareto_vectors: torch.Tensor
    pareto_scores: Dict[str, torch.Tensor]

    # All observations
    V_observed: torch.Tensor
    scores_observed: Dict[str, torch.Tensor]

    # Metadata
    boundary_threshold: float
    n_boundary_measurements: int
    n_pareto_measurements: int
    hypervolume: float


class BoundaryThenParetoDiscovery:
    """
    Two-phase discovery: first find boundary, then Pareto search inside.

    Phase 1 (Boundary): Uses straddle acquisition to find where refusal_score
    crosses the threshold. This identifies the "working region" where ablation
    succeeds.

    Phase 2 (Pareto): Searches for optimal trade-offs between refusal and KL
    ONLY inside the boundary. Candidates outside are rejected.

    This is more efficient than unconstrained Pareto search because:
    - Boundary discovery focuses measurements on the important transition
    - Pareto search doesn't waste measurements on non-working vectors
    """

    def __init__(
        self,
        scorer: MultiObjectiveScorer,
        n_layers: int,
        hidden_dim: int,
        config: Optional[BoundaryThenParetoConfig] = None,
        v_init: Optional[torch.Tensor] = None
    ):
        self.scorer = scorer
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.config = config or BoundaryThenParetoConfig()
        self.v_init = v_init

        # Observations (shared across phases)
        self.V_observed: List[torch.Tensor] = []
        self.scores_observed: Dict[str, List[float]] = {
            'refusal_score': [],
            'kl_score': [],
            'induce_score': [],
            'ablation_loss': [],
            'retain_loss': [],
        }

        # GP for refusal score (used in boundary phase)
        from .adaptive_geometry_discovery import SimpleGP, StructuredLayerGP
        if self.config.gp_type == 'structured':
            self.refusal_gp = StructuredLayerGP(
                n_layers=n_layers,
                hidden_dim=hidden_dim,
                feature_lengthscale=self.config.kernel_lengthscale,
                layer_lengthscale=self.config.layer_lengthscale
            )
        else:
            self.refusal_gp = SimpleGP(lengthscale=self.config.kernel_lengthscale)

    def discover(self) -> BoundaryThenParetoResults:
        """Run two-phase discovery."""

        print("=" * 70)
        print("Two-Phase Discovery: Boundary → Pareto")
        print("=" * 70)

        # Compute dynamic threshold from baseline if enabled
        if self.config.use_dynamic_threshold:
            baseline_refusal, baseline_harmless = self.scorer.compute_baseline_refusal_scores()
            gap = baseline_refusal - baseline_harmless
            self.config.boundary_threshold = baseline_refusal - self.config.boundary_fraction * gap
            print(f"Dynamic threshold: refusal={baseline_refusal:.3f}, harmless={baseline_harmless:.3f}")
            print(f"  Gap={gap:.3f}, fraction={self.config.boundary_fraction:.0%}")
            print(f"  Boundary = {baseline_refusal:.3f} - {self.config.boundary_fraction:.0%}*{gap:.3f} = {self.config.boundary_threshold:.3f}")
        else:
            print(f"Fixed boundary threshold: {self.config.boundary_threshold}")

        # Evaluate v_init first (if provided) to include in observations
        if self.v_init is not None:
            print("Evaluating initial vector (difference-in-means)...")
            v_init_normed = self.v_init / (self.v_init.norm() + 1e-8)
            scores = self.scorer.score(v_init_normed)
            self.V_observed.append(v_init_normed.detach().cpu())
            for obj, val in scores.items():
                self.scores_observed[obj].append(val)
            print(f"  v_init refusal={scores['refusal_score']:.4f}, kl={scores.get('kl_score', 0):.4f}")

        # Phase 1: Find boundary
        print("\n" + "=" * 70)
        print("Phase 1: Boundary Discovery")
        print("=" * 70)
        boundary_points, inside_points, outside_points = self._boundary_phase()
        n_boundary = len(self.V_observed)

        print(f"\n  Boundary points: {len(boundary_points)}")
        print(f"  Inside (ablation works): {len(inside_points)}")
        print(f"  Outside (ablation fails): {len(outside_points)}")

        if len(inside_points) == 0:
            print("  WARNING: No points inside boundary! Adjusting threshold...")
            # Use the best refusal score as new threshold
            best_refusal = min(self.scores_observed['refusal_score'])
            self.config.boundary_threshold = best_refusal + 1.0
            # Re-classify
            boundary_points, inside_points, outside_points = self._classify_points()

        # Phase 2: Pareto search inside boundary
        print("\n" + "=" * 70)
        print("Phase 2: Constrained Pareto Search (inside boundary)")
        print("=" * 70)
        pareto_vectors, pareto_scores, hypervolume = self._pareto_phase(inside_points)
        n_pareto = len(self.V_observed) - n_boundary

        print(f"\n  Pareto vectors found: {len(pareto_vectors)}")
        print(f"  Hypervolume: {hypervolume:.4f}")

        return BoundaryThenParetoResults(
            boundary_points=torch.stack(boundary_points) if boundary_points else torch.empty(0),
            inside_points=torch.stack(inside_points) if inside_points else torch.empty(0),
            outside_points=torch.stack(outside_points) if outside_points else torch.empty(0),
            pareto_vectors=pareto_vectors,
            pareto_scores=pareto_scores,
            V_observed=torch.stack(self.V_observed),
            scores_observed={k: torch.tensor(v) for k, v in self.scores_observed.items()},
            boundary_threshold=self.config.boundary_threshold,
            n_boundary_measurements=n_boundary,
            n_pareto_measurements=n_pareto,
            hypervolume=hypervolume
        )

    def _boundary_phase(self) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        """Phase 1: Find the boundary using straddle acquisition."""

        threshold = self.config.boundary_threshold

        # Initial exploration
        print(f"  Initial exploration ({self.config.n_boundary_init} samples)...")
        for i in range(self.config.n_boundary_init):
            if self.v_init is not None and i < self.config.n_boundary_init // 3:
                # Near v_init: small perturbation
                v = self.v_init + 0.3 * torch.randn_like(self.v_init)
            else:
                # Random direction: match v_init's per-layer norm structure
                device = self.v_init.device if self.v_init is not None else 'cpu'
                v = torch.randn(self.n_layers, self.hidden_dim, device=device)
                if self.v_init is not None:
                    # Scale each layer to match v_init's per-layer norms
                    v_init_norms = self.v_init.norm(dim=-1, keepdim=True)
                    v = v / (v.norm(dim=-1, keepdim=True) + 1e-8) * v_init_norms
            # Normalize globally (for GP, which expects unit vectors)
            v = v / (v.norm() + 1e-8)

            scores = self.scorer.score(v)
            self.V_observed.append(v.detach().cpu())
            for obj, val in scores.items():
                self.scores_observed[obj].append(val)

            if i % 10 == 0:
                print(f"    Sample {i+1}: refusal={scores['refusal_score']:.4f}")

        # Fit GP
        self._update_refusal_gp()

        # Boundary iterations using straddle acquisition
        if self.config.use_multi_fidelity:
            print(f"  Multi-fidelity straddle search ({self.config.n_boundary_iterations} iterations)...")
            print(f"    Low-fidelity: {self.config.low_fidelity_fraction:.0%} of prompts, "
                  f"screen {self.config.n_screen_candidates} → promote {self.config.n_promote}")
        else:
            print(f"  Straddle search ({self.config.n_boundary_iterations} iterations)...")

        for iteration in range(self.config.n_boundary_iterations):
            if len(self.V_observed) >= self.config.max_measurements // 2:
                break

            # Generate candidates
            n_candidates = self.config.n_screen_candidates if self.config.use_multi_fidelity else 50
            candidates = self._generate_candidates(near_boundary=True, n_candidates=n_candidates)

            # Straddle acquisition: β*σ - |μ - threshold|
            V_flat = candidates.reshape(len(candidates), -1)
            mu, sigma = self.refusal_gp.predict(V_flat)
            straddle = self.config.boundary_beta * sigma - torch.abs(mu - threshold)

            if self.config.use_multi_fidelity:
                # Multi-fidelity: screen with low fidelity, promote best to high fidelity
                # Step 1: Get top candidates by GP acquisition
                top_k = min(self.config.n_promote * 3, len(candidates))
                top_indices = straddle.argsort(descending=True)[:top_k]

                # Step 2: Score top candidates with low fidelity
                low_fi_scores = []
                for idx in top_indices:
                    v = candidates[idx]
                    s = self.scorer.score(v, fidelity=self.config.low_fidelity_fraction)
                    low_fi_scores.append(s['refusal_score'])

                # Step 3: Select best by low-fidelity straddle
                low_fi_scores = torch.tensor(low_fi_scores)
                low_fi_straddle = -torch.abs(low_fi_scores - threshold)  # Closer to threshold = better
                promote_indices = low_fi_straddle.argsort(descending=True)[:self.config.n_promote]

                # Step 4: Evaluate promoted candidates with high fidelity
                best_score = None
                best_v = None
                for pi in promote_indices:
                    orig_idx = top_indices[pi]
                    v = candidates[orig_idx]
                    scores = self.scorer.score(v, fidelity=1.0)

                    if best_score is None or abs(scores['refusal_score'] - threshold) < abs(best_score['refusal_score'] - threshold):
                        best_score = scores
                        best_v = v

                # Add best to observations
                self.V_observed.append(best_v.detach().cpu())
                for obj, val in best_score.items():
                    self.scores_observed[obj].append(val)
                scores = best_score
            else:
                # Single fidelity: just pick best by GP acquisition
                best_idx = straddle.argmax()
                v_next = candidates[best_idx]

                # Score with full fidelity
                scores = self.scorer.score(v_next)
                self.V_observed.append(v_next.detach().cpu())
                for obj, val in scores.items():
                    self.scores_observed[obj].append(val)

            # Update GP
            self._update_refusal_gp()

            if iteration % 10 == 0:
                print(f"    Iter {iteration}: refusal={scores['refusal_score']:.4f}")

        # Classify points
        return self._classify_points()

    def _classify_points(self) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        """Classify observed points as inside/outside/boundary."""
        threshold = self.config.boundary_threshold
        boundary_margin = 1.0  # Points within this margin of threshold are "boundary"

        inside = []
        outside = []
        boundary = []

        for v, r in zip(self.V_observed, self.scores_observed['refusal_score']):
            if r < threshold - boundary_margin:
                inside.append(v)
            elif r > threshold + boundary_margin:
                outside.append(v)
            else:
                boundary.append(v)

        return boundary, inside, outside

    def _pareto_phase(
        self,
        inside_points: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], float]:
        """Phase 2: Pareto search constrained to inside boundary."""

        # Use inside points as starting points
        starting_points = inside_points if inside_points else self.V_observed[-10:]

        # Create a constrained Pareto discovery
        pareto_config = ParetoDiscoveryConfig(
            n_init_samples=min(self.config.n_pareto_init, len(starting_points)),
            n_pareto_iterations=self.config.n_pareto_iterations,
            max_measurements=self.config.max_measurements - len(self.V_observed),
            beta=self.config.pareto_beta,
            gp_type=self.config.gp_type,
            layer_lengthscale=self.config.layer_lengthscale,
            kernel_lengthscale=self.config.kernel_lengthscale,
            use_gradient_candidates=self.config.use_gradient_candidates,
            gradient_steps=self.config.gradient_steps,
            gradient_lr=self.config.gradient_lr,
            use_smooth_parameterization=self.config.use_smooth_parameterization,
            n_basis=self.config.n_basis
        )

        # Initialize Pareto discovery with inside points
        pareto_discovery = ParetoGeometryDiscovery(
            scorer=self.scorer,
            n_layers=self.n_layers,
            hidden_dim=self.hidden_dim,
            config=pareto_config,
            v_init=starting_points[0] if starting_points else None,
            use_mean_diff_init=False
        )

        # Seed with existing inside points (skip re-scoring)
        index_by_id = {id(v): i for i, v in enumerate(self.V_observed)}
        for v in starting_points[:pareto_config.n_init_samples]:
            idx = index_by_id.get(id(v))
            if idx is None:
                # Fallback: try to find by value (should be rare)
                for i, v_obs in enumerate(self.V_observed):
                    if torch.allclose(v_obs, v):
                        idx = i
                        break

            if idx is None:
                # Not previously observed; score once and add to Pareto only
                scores = self.scorer.score(v)
                pareto_discovery.V_observed.append(v.detach().cpu())
                for obj, val in scores.items():
                    pareto_discovery.scores_observed[obj].append(val)
                continue

            # Reuse cached scores from boundary phase
            pareto_discovery.V_observed.append(v.detach().cpu())
            for obj, vals in self.scores_observed.items():
                if idx < len(vals):
                    pareto_discovery.scores_observed[obj].append(vals[idx])

        pareto_discovery._update_gps()

        # Run Pareto iterations with boundary constraint
        print(f"  Pareto search ({pareto_config.n_pareto_iterations} iterations)...")
        for iteration in range(pareto_config.n_pareto_iterations):
            if len(self.V_observed) >= self.config.max_measurements:
                break

            # Generate candidates
            candidates = pareto_discovery._generate_candidates()

            # Filter to inside boundary (predict with refusal GP)
            V_flat = candidates.reshape(len(candidates), -1)
            mu_refusal, _ = self.refusal_gp.predict(V_flat)
            inside_mask = mu_refusal < self.config.boundary_threshold
            valid_candidates = candidates[inside_mask]

            if len(valid_candidates) == 0:
                # All candidates outside - sample near inside points
                idx = torch.randint(len(starting_points), (1,)).item()
                v_next = starting_points[idx] + 0.1 * torch.randn_like(starting_points[idx])
                v_next = v_next / (v_next.norm() + 1e-8)
            else:
                # Compute acquisition on valid candidates
                acquisition = pareto_discovery._compute_ehvi_acquisition(valid_candidates)
                best_idx = acquisition.argmax()
                v_next = valid_candidates[best_idx]

            # Score
            scores = self.scorer.score(v_next)

            # Store in both
            pareto_discovery.V_observed.append(v_next.detach().cpu())
            self.V_observed.append(v_next.detach().cpu())
            for obj, val in scores.items():
                pareto_discovery.scores_observed[obj].append(val)
                self.scores_observed[obj].append(val)

            pareto_discovery._update_gps()

            if iteration % 10 == 0:
                hv = pareto_discovery._compute_hypervolume()
                print(f"    Iter {iteration}: refusal={scores['refusal_score']:.4f}, "
                      f"kl={scores['kl_score']:.4f}, HV={hv:.4f}")

        # Extract Pareto frontier
        results = pareto_discovery._extract_pareto()
        return results.pareto_vectors, results.pareto_scores, results.hypervolume

    def _compute_refusal_gradient(self, v: torch.Tensor) -> torch.Tensor:
        """
        Compute gradient of refusal_score with respect to v.

        Uses the scorer's gradient computation (actual loss gradient through model).
        Falls back to GP gradient if scorer doesn't support gradients.
        """
        v = v.clone().detach().requires_grad_(True)

        # Try to get gradient from scorer (actual loss gradient)
        if hasattr(self.scorer, 'score_with_grad'):
            try:
                scores, grad = self.scorer.score_with_grad(v)
                return grad['refusal_score'].detach()
            except Exception:
                pass

        # Fallback: use GP gradient (less accurate but always available)
        v_flat = v.reshape(1, -1)
        mu, _ = self.refusal_gp.predict(v_flat)

        # Compute gradient of GP mean
        mu.sum().backward()
        grad = v.grad.clone()

        return grad.detach()

    def _project_orthogonal_to_gradient(
        self,
        directions: torch.Tensor,
        gradient: torch.Tensor
    ) -> torch.Tensor:
        """
        Project directions to be orthogonal to the gradient.

        This keeps movement along the boundary (level set) rather than across it.

        Args:
            directions: [N, n_layers, hidden_dim] candidate directions
            gradient: [n_layers, hidden_dim] gradient to project out

        Returns:
            Orthogonalized directions (still on unit sphere)
        """
        g_flat = gradient.reshape(-1)
        g_norm_sq = (g_flat @ g_flat) + 1e-8

        result = []
        for d in directions:
            d_flat = d.reshape(-1)
            # Project out gradient component: d' = d - (d·g / ||g||²) * g
            proj = (d_flat @ g_flat) / g_norm_sq
            d_orthogonal = d_flat - proj * g_flat

            # Reshape and normalize back to sphere
            d_orthogonal = d_orthogonal.reshape(self.n_layers, self.hidden_dim)
            d_orthogonal = d_orthogonal / (d_orthogonal.norm() + 1e-8)
            result.append(d_orthogonal)

        return torch.stack(result)

    def _generate_candidates(self, near_boundary: bool = False, n_candidates: int = 200) -> torch.Tensor:
        """
        Generate candidate vectors.

        Args:
            near_boundary: If True and we have boundary points, use smarter exploration
            n_candidates: Number of candidates to generate

        When near_boundary=True and we have points on the boundary:
        - If use_gradient_candidates=True: uses gradient-orthogonal exploration
        - If use_gradient_candidates=False: uses simple random perturbation (faster)
        """
        candidates = []
        n_total = n_candidates

        if near_boundary and len(self.V_observed) > 5:
            threshold = self.config.boundary_threshold
            scores = torch.tensor(self.scores_observed['refusal_score'])
            distances = torch.abs(scores - threshold)
            near_indices = distances.argsort()[:10]

            # Find points that are actually ON the boundary (within margin)
            boundary_margin = 1.0
            on_boundary_mask = distances < boundary_margin
            boundary_indices = torch.where(on_boundary_mask)[0]

            if len(boundary_indices) > 0 and self.config.use_gradient_candidates:
                # === Gradient-orthogonal exploration for boundary points ===
                n_orthogonal = n_total // 3

                for _ in range(n_orthogonal):
                    # Pick a boundary point
                    idx = boundary_indices[torch.randint(len(boundary_indices), (1,))].item()
                    v_boundary = self.V_observed[idx]

                    # Compute gradient at this point
                    try:
                        grad = self._compute_refusal_gradient(v_boundary)

                        # Generate random direction and project orthogonal to gradient
                        d_random = torch.randn(self.n_layers, self.hidden_dim)
                        d_orthogonal = self._project_orthogonal_to_gradient(
                            d_random.unsqueeze(0), grad
                        )[0]

                        # Move along orthogonal direction (stay on boundary)
                        step_size = 0.2
                        v_new = v_boundary + step_size * d_orthogonal
                        v_new = v_new / (v_new.norm() + 1e-8)
                        candidates.append(v_new)
                    except Exception:
                        # Fallback to random perturbation
                        v = v_boundary + 0.2 * torch.randn_like(v_boundary)
                        v = v / (v.norm() + 1e-8)
                        candidates.append(v)

                # === Gradient descent toward boundary for off-boundary points ===
                n_toward_boundary = n_total // 3

                for _ in range(n_toward_boundary):
                    # Pick a point not on boundary
                    off_boundary = torch.where(~on_boundary_mask)[0]
                    if len(off_boundary) == 0:
                        off_boundary = near_indices
                    idx = off_boundary[torch.randint(len(off_boundary), (1,))].item()
                    v_start = self.V_observed[idx]
                    score = scores[idx].item()

                    try:
                        grad = self._compute_refusal_gradient(v_start)

                        # Move toward threshold:
                        # if score > threshold (outside), move in -grad direction
                        # if score < threshold (inside), move in +grad direction
                        if score > threshold:
                            direction = -grad  # Decrease refusal
                        else:
                            direction = grad   # Increase refusal

                        direction = direction / (direction.norm() + 1e-8)
                        step_size = 0.1
                        v_new = v_start + step_size * direction
                        v_new = v_new / (v_new.norm() + 1e-8)
                        candidates.append(v_new)
                    except Exception:
                        v = v_start + 0.2 * torch.randn_like(v_start)
                        v = v / (v.norm() + 1e-8)
                        candidates.append(v)
            elif len(boundary_indices) > 0:
                # Boundary points exist but no gradients - simple perturbation
                for _ in range(n_total // 2):
                    idx = boundary_indices[torch.randint(len(boundary_indices), (1,))].item()
                    v = self.V_observed[idx] + 0.2 * torch.randn(self.n_layers, self.hidden_dim)
                    v = v / (v.norm() + 1e-8)
                    candidates.append(v)
            else:
                # No boundary points yet - sample near points closest to threshold
                for _ in range(n_total // 2):
                    idx = near_indices[torch.randint(len(near_indices), (1,))].item()
                    v = self.V_observed[idx] + 0.2 * torch.randn(self.n_layers, self.hidden_dim)
                    v = v / (v.norm() + 1e-8)
                    candidates.append(v)

        # Random samples for exploration
        while len(candidates) < n_total:
            v = torch.randn(self.n_layers, self.hidden_dim)
            v = v / (v.norm() + 1e-8)
            candidates.append(v)

        return torch.stack(candidates)

    def _update_refusal_gp(self):
        """Update the refusal GP with current observations."""
        V_flat = torch.stack(self.V_observed).reshape(len(self.V_observed), -1)
        R = torch.tensor(self.scores_observed['refusal_score'])
        self.refusal_gp.fit(V_flat, R)
