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
        n_score_tokens: int = 1
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
        """
        self.model = model
        self.tokenizer = tokenizer
        self.harmful_prompts = harmful_prompts
        self.harmless_prompts = harmless_prompts
        self.refusal_toks = refusal_toks.to(device)
        self.device = device
        self.harmful_completions = harmful_completions
        self.n_score_tokens = n_score_tokens

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

    def _prepare_inputs(self):
        """Pre-tokenize prompts for faster scoring."""
        self.harmful_inputs = self.tokenizer(
            self.harmful_prompts,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.device)

        self.harmless_inputs = self.tokenizer(
            self.harmless_prompts,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.device)

        # If harmful completions provided, tokenize prompt+completion
        # and track where the completion starts for each example
        if self.harmful_completions is not None:
            self._prepare_completion_inputs()

    def _prepare_completion_inputs(self):
        """Prepare inputs for multi-token completion scoring."""
        # Tokenize prompts alone to get prompt lengths
        prompt_encodings = self.tokenizer(
            self.harmful_prompts,
            return_tensors='pt',
            padding=False,
            truncation=True,
            max_length=256
        )
        self.prompt_lengths = [len(ids) for ids in prompt_encodings['input_ids']]

        # Tokenize prompt + completion together
        full_texts = [
            p + c for p, c in zip(self.harmful_prompts, self.harmful_completions)
        ]
        self.harmful_full_inputs = self.tokenizer(
            full_texts,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=512
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
            outputs = self.model(**self.harmless_inputs)
            self.baseline_harmless_logits = self._last_nonpad_logits(outputs, self.harmless_inputs).detach()

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

            # Compute unit vector r̂ = r / ||r||
            r_norm = r_layer.norm() + 1e-8
            r_unit = r_layer / r_norm

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

    def score(self, v: torch.Tensor, return_grad: bool = False) -> Union[Dict[str, float], Tuple[Dict[str, float], torch.Tensor]]:
        """
        Compute all objective scores for vector v.

        Just 2 forward passes - no generation needed!

        Args:
            v: Direction vector [n_layers, hidden_dim] or [hidden_dim]
            return_grad: Whether to return gradient w.r.t. v

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

        # Get ACE reference point (cached)
        v_minus = self._ensure_v_minus_cached()

        # Register ACE ablation hooks (alpha=0 for ablation)
        handles = self._apply_ablation_hooks(v_flat, v_minus=v_minus, alpha=0.0)

        try:
            # === Forward pass 1: Harmful prompts (refusal score or completion loss) ===
            if self.harmful_completions is not None:
                # Multi-token completion scoring
                if return_grad:
                    outputs = self.model(**self.harmful_full_inputs)
                else:
                    with torch.no_grad():
                        outputs = self.model(**self.harmful_full_inputs)

                # Compute cross-entropy loss on completion tokens
                # Shift logits and labels for next-token prediction
                logits = outputs.logits[:, :-1, :].contiguous()
                labels = self.completion_labels[:, 1:].contiguous()

                # Compute per-token loss
                loss_fct = nn.CrossEntropyLoss(reduction='none', ignore_index=-100)
                per_token_loss = loss_fct(
                    logits.view(-1, logits.size(-1)),
                    labels.view(-1)
                ).view(logits.size(0), logits.size(1))

                # Average over valid completion tokens per example
                valid_mask = (labels != -100).float()
                per_example_loss = (per_token_loss * valid_mask).sum(dim=1) / (valid_mask.sum(dim=1) + 1e-8)

                # Refusal score: negative loss (lower loss = model complies = lower score)
                # We want to MINIMIZE refusal_score, so negative loss works
                # Actually: higher loss = more refusal, lower loss = more compliance
                # For consistency: refusal_score = loss (high = refusing, low = complying)
                refusal_logits = per_example_loss
                scores['refusal_score'] = per_example_loss.mean().item()
            else:
                # Original single-token refusal scoring
                if return_grad:
                    outputs = self.model(**self.harmful_inputs)
                else:
                    with torch.no_grad():
                        outputs = self.model(**self.harmful_inputs)

                ablated_harmful_logits = self._last_nonpad_logits(outputs, self.harmful_inputs)

                # Refusal score: logit(P_refusal) = log(P_r) - log(1 - P_r)
                # Negative when ablation works well (e.g., -6)
                # Uses the proper refusal_score_fn from scoring.py
                refusal_logits = refusal_score_fn(
                    ablated_harmful_logits,
                    self.refusal_toks
                )
                scores['refusal_score'] = refusal_logits.mean().item()

            # === Forward pass 2: Harmless prompts (KL score) ===
            if return_grad:
                outputs = self.model(**self.harmless_inputs)
            else:
                with torch.no_grad():
                    outputs = self.model(**self.harmless_inputs)

            ablated_harmless_logits = self._last_nonpad_logits(outputs, self.harmless_inputs)

            # KL divergence: KL(baseline || ablated)
            # Use log_softmax for numerical stability
            log_p_baseline = torch.log_softmax(self.baseline_harmless_logits.float(), dim=-1)
            log_p_ablated = torch.log_softmax(ablated_harmless_logits.float(), dim=-1)
            p_baseline = log_p_baseline.exp()

            kl = (p_baseline * (log_p_baseline - log_p_ablated)).sum(dim=-1).mean()
            # Clamp to avoid NaN/Inf
            kl = torch.clamp(kl, min=0.0, max=100.0)
            scores['kl_score'] = kl.item()

            # Induce score: refusal logit on harmless prompts under ablation
            induce_logits = refusal_score_fn(
                ablated_harmless_logits,
                self.refusal_toks
            )
            scores['induce_score'] = induce_logits.mean().item()

            # Compute gradient if requested
            if return_grad:
                # Combined loss: want refusal_score negative (minimize it)
                # and kl_score low (minimize it)
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
            'kl_score': []
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

    def discover(self) -> ParetoDiscoveryResults:
        """Run multi-objective Pareto discovery."""

        print("=" * 70)
        print("Multi-Objective Pareto Boundary Discovery")
        print("=" * 70)
        print(f"\nObjectives: {self.config.primary_objectives}")
        print(f"Secondary: {self.config.secondary_objectives}")

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

    def _initial_exploration(self):
        """Phase 1: Initial stratified sampling."""

        for i in range(self.config.n_init_samples):
            # Sample direction
            if self.v_init is not None and i < self.config.n_init_samples // 3:
                v = self.v_init + 0.5 * torch.randn_like(self.v_init)
            else:
                v = torch.randn(self.n_layers, self.hidden_dim)

            v = self._normalize_vector(v)

            # Score
            scores = self.scorer.score(v)

            # Store
            self.V_observed.append(v)
            for obj, val in scores.items():
                self.scores_observed[obj].append(val)

            if i % 10 == 0:
                print(f"  Sample {i+1}: refusal={scores['refusal_score']:.4f}, "
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

            # Store
            self.V_observed.append(v_next)
            for obj, val in scores.items():
                self.scores_observed[obj].append(val)

            # Update GPs
            self._update_gps()

            if iteration % 10 == 0:
                hv = self._compute_hypervolume()
                print(f"  Iter {iteration}: refusal={scores['refusal_score']:.4f}, "
                      f"kl={scores['kl_score']:.4f}, HV={hv:.4f}")

    def _generate_candidates(self) -> torch.Tensor:
        """Generate candidate directions."""
        candidates = []

        # Mix of strategies
        n_uniform = self.config.n_candidates // 2
        n_near_pareto = self.config.n_candidates - n_uniform

        # Uniform on sphere
        for _ in range(n_uniform):
            v = torch.randn(self.n_layers, self.hidden_dim)
            v = self._normalize_vector(v)
            candidates.append(v)

        # Near current Pareto points
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

        # Pareto vectors and scores
        pareto_vectors = V_tensor[pareto_mask]
        pareto_scores = {
            obj: torch.tensor(self.scores_observed[obj])[pareto_mask]
            for obj in self.scores_observed
        }

        # Dominated vectors and scores
        dominated_vectors = V_tensor[dominated_mask]
        dominated_scores = {
            obj: torch.tensor(self.scores_observed[obj])[dominated_mask]
            for obj in self.scores_observed
        }

        # All scores
        all_scores = {
            obj: torch.tensor(vals)
            for obj, vals in self.scores_observed.items()
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
