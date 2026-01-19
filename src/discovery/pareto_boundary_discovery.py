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
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Callable, List, Dict, Optional, Tuple, Union
from dataclasses import dataclass, field

from .adaptive_geometry_discovery import SimpleGP, AdaptiveSparseGP


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
    beta: float = 1.96  # For UCB-based acquisition
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
    kernel_type: str = 'rbf'
    kernel_lengthscale: float = 0.3
    use_sparse_gp: bool = True
    num_inducing: int = 64
    sparse_train_steps: int = 10
    sparse_lr: float = 0.05


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


class MultiObjectiveScorer:
    """
    Compute multiple objective scores efficiently using forward passes only.

    NO GENERATION NEEDED - just:
    1. Forward pass on harmful prompts -> refusal token logits
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
        device: str = 'cuda'
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.harmful_prompts = harmful_prompts
        self.harmless_prompts = harmless_prompts
        self.refusal_toks = refusal_toks.to(device)
        self.device = device

        # Pre-tokenize prompts for speed
        self._prepare_inputs()

        # Cache baseline logits for KL computation
        self._cache_baselines()

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

    def _apply_ablation_hooks(self, v: torch.Tensor) -> List:
        """Register ablation hooks for vector v."""
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

        for layer_idx, layer in enumerate(layers):
            # Get this layer's vector, move to device, and normalize
            v_layer = v_per_layer[layer_idx].to(self.device)
            v_layer = v_layer / (v_layer.norm() + 1e-8)

            def make_hook(v_l):
                def hook(module, input, output):
                    if isinstance(output, tuple):
                        h = output[0]
                    else:
                        h = output

                    # Project out: h' = h - (h·v)v
                    proj = torch.einsum('...d,d->...', h, v_l)
                    h_ablated = h - torch.einsum('...,d->...d', proj, v_l)

                    if isinstance(output, tuple):
                        return (h_ablated,) + output[1:]
                    return h_ablated
                return hook

            handles.append(layer.register_forward_hook(make_hook(v_layer)))

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

        # Register ablation hooks
        handles = self._apply_ablation_hooks(v_flat)

        try:
            # === Forward pass 1: Harmful prompts (refusal score) ===
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
            p_baseline = torch.softmax(self.baseline_harmless_logits, dim=-1)
            p_ablated = torch.softmax(ablated_harmless_logits, dim=-1)

            eps = 1e-8
            kl = (p_baseline * (torch.log(p_baseline + eps) - torch.log(p_ablated + eps))).sum(dim=-1).mean()
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
        v_init: Optional[torch.Tensor] = None
    ):
        self.scorer = scorer
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.config = config or ParetoDiscoveryConfig()
        self.v_init = v_init

        # Observations
        self.V_observed: List[torch.Tensor] = []
        self.scores_observed: Dict[str, List[float]] = {
            'refusal_score': [],
            'induce_score': [],
            'kl_score': []
        }

        # One GP per objective
        self.gps: Dict[str, Union[SimpleGP, AdaptiveSparseGP]] = {}
        for obj in self.config.primary_objectives + self.config.secondary_objectives:
            if self.config.use_sparse_gp:
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

    def _initial_exploration(self):
        """Phase 1: Initial stratified sampling."""

        for i in range(self.config.n_init_samples):
            # Sample direction
            if self.v_init is not None and i < self.config.n_init_samples // 3:
                v = self.v_init + 0.5 * torch.randn_like(self.v_init)
            else:
                v = torch.randn(self.n_layers, self.hidden_dim)

            v = v / v.norm(dim=-1, keepdim=True)

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
            v = v / v.norm(dim=-1, keepdim=True)
            candidates.append(v)

        # Near current Pareto points
        pareto_mask = self._get_pareto_mask()
        pareto_indices = torch.where(pareto_mask)[0]

        if len(pareto_indices) > 0:
            for _ in range(n_near_pareto):
                idx = pareto_indices[torch.randint(len(pareto_indices), (1,))].item()
                v_base = self.V_observed[idx]
                v = v_base + 0.2 * torch.randn_like(v_base)
                v = v / v.norm(dim=-1, keepdim=True)
                candidates.append(v)
        else:
            for _ in range(n_near_pareto):
                v = torch.randn(self.n_layers, self.hidden_dim)
                v = v / v.norm(dim=-1, keepdim=True)
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

        return ParetoDiscoveryResults(
            pareto_vectors=pareto_vectors,
            pareto_scores=pareto_scores,
            V_observed=V_tensor,
            scores_observed=all_scores,
            dominated_vectors=dominated_vectors,
            dominated_scores=dominated_scores,
            hypervolume=hv,
            gps=self.gps,
            n_measurements=len(self.V_observed)
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
    config: Optional[ParetoDiscoveryConfig] = None
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

    Returns:
        ParetoDiscoveryResults with Pareto-optimal vectors
    """
    scorer = MultiObjectiveScorer(
        model=model,
        tokenizer=tokenizer,
        harmful_prompts=harmful_prompts,
        harmless_prompts=harmless_prompts,
        refusal_toks=refusal_toks
    )

    discovery = ParetoGeometryDiscovery(
        scorer=scorer,
        n_layers=n_layers,
        hidden_dim=hidden_dim,
        config=config,
        v_init=v_init
    )

    return discovery.discover()
