"""
Single-vector discovery: find one direction v that works across all layers.

Instead of optimizing a [n_layers, hidden_dim] matrix, we find a single
vector v ∈ R^{hidden_dim} and ablate it at every layer using ACE:

    h'_i = h_i - proj_v(h_i) + proj_v(v⁻_i)

where v⁻_i is the mean harmless activation at layer i (layer-specific baseline).

This dramatically simplifies the search space while leveraging the insight that
a single refusal direction may be represented similarly across layers.
"""

import torch
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Callable
import math


@dataclass
class SingleVectorDiscoveryConfig:
    """Configuration for single-vector discovery."""
    # Search settings
    n_init_samples: int = 20
    n_iterations: int = 50

    # GP settings
    kernel_lengthscale: float = 0.3  # RBF lengthscale on unit sphere
    noise_variance: float = 0.01

    # Acquisition
    beta: float = 2.0  # UCB exploration parameter
    n_candidates: int = 100  # Candidates per iteration

    # Boundary discovery (optional)
    use_boundary_search: bool = True
    boundary_threshold: Optional[float] = None  # Auto-computed if None
    boundary_fraction: float = 0.05  # 5% of gap from refusal to non-refusal

    # Gradient-based refinement
    use_gradient_refinement: bool = True
    gradient_steps: int = 5
    gradient_lr: float = 0.1


@dataclass
class SingleVectorResults:
    """Results from single-vector discovery."""
    best_vector: torch.Tensor  # [hidden_dim]
    best_scores: Dict[str, float]

    # All observations
    V_observed: torch.Tensor  # [n_obs, hidden_dim]
    scores_observed: Dict[str, torch.Tensor]

    # Pareto frontier (if multi-objective)
    pareto_vectors: Optional[torch.Tensor] = None
    pareto_scores: Optional[Dict[str, torch.Tensor]] = None
    hypervolume: float = 0.0

    n_measurements: int = 0


class SingleVectorScorer:
    """
    Score a single vector by ablating it at all layers with ACE.

    Uses pre-computed layer-specific baselines v⁻_i.
    """

    def __init__(
        self,
        model,
        tokenizer,
        harmful_prompts: List[str],
        harmless_prompts: List[str],
        refusal_toks: torch.Tensor,
        device: str = "cuda",
        layers: Optional[List[int]] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.harmful_prompts = harmful_prompts
        self.harmless_prompts = harmless_prompts
        self.refusal_toks = refusal_toks.to(device)
        self.device = device

        # Get model structure
        if hasattr(model, 'model') and hasattr(model.model, 'layers'):
            self.layer_modules = model.model.layers
        elif hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
            self.layer_modules = model.transformer.h
        else:
            raise ValueError("Unknown model architecture")

        self.n_layers = len(self.layer_modules)
        self.hidden_dim = model.config.hidden_size

        # Which layers to ablate (default: all)
        self.layers = layers if layers is not None else list(range(self.n_layers))

        # Pre-compute baselines
        self._baselines_computed = False
        self._v_minus = None  # [n_layers, hidden_dim] - mean harmless per layer
        self._v_plus = None   # [n_layers, hidden_dim] - mean harmful per layer

    def compute_baselines(self):
        """Pre-compute layer-specific baselines from harmless activations."""
        if self._baselines_computed:
            return

        print("Computing layer-specific baselines...")

        # Collect activations
        harmful_acts = self._collect_activations(self.harmful_prompts)
        harmless_acts = self._collect_activations(self.harmless_prompts)

        # Mean per layer
        self._v_minus = torch.stack([harmless_acts[l].mean(dim=0) for l in range(self.n_layers)])
        self._v_plus = torch.stack([harmful_acts[l].mean(dim=0) for l in range(self.n_layers)])

        self._baselines_computed = True
        print(f"  Baselines computed for {self.n_layers} layers")

    def _collect_activations(self, prompts: List[str]) -> Dict[int, torch.Tensor]:
        """Collect activations at all layers for given prompts."""
        activations = {l: [] for l in range(self.n_layers)}
        hooks = []

        def make_hook(layer_idx):
            def hook(module, input, output):
                h = output[0] if isinstance(output, tuple) else output
                activations[layer_idx].append(h[:, -1, :].detach())
            return hook

        for layer_idx in range(self.n_layers):
            hooks.append(self.layer_modules[layer_idx].register_forward_hook(make_hook(layer_idx)))

        for prompt in prompts:
            if self.tokenizer.chat_template is not None:
                formatted = self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=False,
                    add_generation_prompt=True
                )
            else:
                formatted = prompt

            inputs = self.tokenizer(formatted, return_tensors="pt", truncation=True, max_length=512)
            inputs = inputs.to(self.device)

            with torch.no_grad():
                self.model(**inputs)

        for hook in hooks:
            hook.remove()

        return {l: torch.stack(activations[l]).squeeze(1) for l in range(self.n_layers)}

    def score(self, v: torch.Tensor) -> Dict[str, float]:
        """
        Score a single vector by ablating at all layers with ACE.

        Args:
            v: [hidden_dim] direction vector (will be normalized)

        Returns:
            Dict with 'refusal_score' and 'kl_score'
        """
        self.compute_baselines()

        v = v.to(self.device)
        v_norm = v / (v.norm() + 1e-8)

        hooks = []

        # Register ACE hooks at all target layers
        for layer_idx in self.layers:
            baseline = self._v_minus[layer_idx]  # Layer-specific baseline
            hooks.append(
                self.layer_modules[layer_idx].register_forward_hook(
                    self._make_ace_hook(v_norm, baseline)
                )
            )

        # Score harmful prompts (refusal score)
        refusal_scores = []
        for prompt in self.harmful_prompts:
            if self.tokenizer.chat_template is not None:
                formatted = self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=False,
                    add_generation_prompt=True
                )
            else:
                formatted = prompt

            inputs = self.tokenizer(formatted, return_tensors="pt", truncation=True, max_length=512)
            inputs = inputs.to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits[:, -1, :]
                log_probs = F.log_softmax(logits.float(), dim=-1)
                refusal_log_prob = log_probs[0, self.refusal_toks].mean().item()
                refusal_scores.append(refusal_log_prob)

        # Remove hooks before KL measurement
        for hook in hooks:
            hook.remove()

        # KL divergence on harmless (capability preservation)
        # Compare ablated vs original logits
        kl_scores = []
        for prompt in self.harmless_prompts:
            if self.tokenizer.chat_template is not None:
                formatted = self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=False,
                    add_generation_prompt=True
                )
            else:
                formatted = prompt

            inputs = self.tokenizer(formatted, return_tensors="pt", truncation=True, max_length=512)
            inputs = inputs.to(self.device)

            # Original logits
            with torch.no_grad():
                orig_outputs = self.model(**inputs)
                orig_logits = orig_outputs.logits[:, -1, :]

            # Ablated logits
            hooks = []
            for layer_idx in self.layers:
                baseline = self._v_minus[layer_idx]
                hooks.append(
                    self.layer_modules[layer_idx].register_forward_hook(
                        self._make_ace_hook(v_norm, baseline)
                    )
                )

            with torch.no_grad():
                abl_outputs = self.model(**inputs)
                abl_logits = abl_outputs.logits[:, -1, :]

            for hook in hooks:
                hook.remove()

            # KL(original || ablated)
            orig_probs = F.softmax(orig_logits.float(), dim=-1)
            abl_log_probs = F.log_softmax(abl_logits.float(), dim=-1)
            kl = F.kl_div(abl_log_probs, orig_probs, reduction='batchmean').item()
            kl_scores.append(kl)

        return {
            'refusal_score': sum(refusal_scores) / len(refusal_scores),
            'kl_score': sum(kl_scores) / len(kl_scores),
        }

    def _make_ace_hook(self, v_norm: torch.Tensor, baseline: torch.Tensor):
        """Create ACE ablation hook: h' = h - proj_v(h) + proj_v(baseline)."""
        def hook(module, input, output):
            h = output[0] if isinstance(output, tuple) else output

            # Project out v from h
            proj_h = torch.einsum('...d,d->...', h, v_norm)

            # Project baseline onto v
            proj_baseline = torch.einsum('d,d->', baseline, v_norm)

            # ACE: h' = h - proj_v(h) + proj_v(baseline)
            h_ablated = h - torch.einsum('...,d->...d', proj_h, v_norm) + proj_baseline * v_norm

            if isinstance(output, tuple):
                return (h_ablated,) + output[1:]
            return h_ablated

        return hook

    def compute_baseline_refusal(self) -> Tuple[float, float]:
        """Compute baseline refusal scores (without ablation)."""
        harmful_scores = []
        for prompt in self.harmful_prompts:
            if self.tokenizer.chat_template is not None:
                formatted = self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=False,
                    add_generation_prompt=True
                )
            else:
                formatted = prompt

            inputs = self.tokenizer(formatted, return_tensors="pt", truncation=True, max_length=512)
            inputs = inputs.to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits[:, -1, :]
                log_probs = F.log_softmax(logits.float(), dim=-1)
                refusal_log_prob = log_probs[0, self.refusal_toks].mean().item()
                harmful_scores.append(refusal_log_prob)

        harmless_scores = []
        for prompt in self.harmless_prompts:
            if self.tokenizer.chat_template is not None:
                formatted = self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=False,
                    add_generation_prompt=True
                )
            else:
                formatted = prompt

            inputs = self.tokenizer(formatted, return_tensors="pt", truncation=True, max_length=512)
            inputs = inputs.to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits[:, -1, :]
                log_probs = F.log_softmax(logits.float(), dim=-1)
                refusal_log_prob = log_probs[0, self.refusal_toks].mean().item()
                harmless_scores.append(refusal_log_prob)

        return (
            sum(harmful_scores) / len(harmful_scores),
            sum(harmless_scores) / len(harmless_scores)
        )

    def get_mean_diff_direction(self) -> torch.Tensor:
        """Get mean difference direction averaged across layers."""
        self.compute_baselines()

        # Mean diff per layer, then average
        r_per_layer = self._v_plus - self._v_minus  # [n_layers, hidden_dim]

        # Average across layers (or could weight by norm)
        r_mean = r_per_layer.mean(dim=0)

        # Normalize
        return r_mean / (r_mean.norm() + 1e-8)

    def get_all_r_directions(self) -> List[torch.Tensor]:
        """Get all r_i directions (normalized) for initialization."""
        self.compute_baselines()

        r_per_layer = self._v_plus - self._v_minus  # [n_layers, hidden_dim]

        # Normalize each
        directions = []
        for i in range(self.n_layers):
            r_i = r_per_layer[i]
            r_i_norm = r_i / (r_i.norm() + 1e-8)
            directions.append(r_i_norm)

        return directions


class SingleVectorGP:
    """Simple GP for optimization on the unit sphere in R^d."""

    def __init__(
        self,
        hidden_dim: int,
        lengthscale: float = 0.3,
        noise_variance: float = 0.01,
    ):
        self.hidden_dim = hidden_dim
        self.lengthscale = lengthscale
        self.noise_variance = noise_variance

        self.V_observed = []  # List of [hidden_dim] tensors
        self.y_observed = []  # List of scalars

    def add_observation(self, v: torch.Tensor, y: float):
        """Add an observation."""
        self.V_observed.append(v.detach().cpu())
        self.y_observed.append(y)

    def _geodesic_distance(self, v1: torch.Tensor, v2: torch.Tensor) -> float:
        """Geodesic distance on unit sphere."""
        v1 = v1.float()
        v2 = v2.float()
        cos_sim = torch.clamp(torch.dot(v1, v2), -1.0, 1.0)
        return torch.acos(cos_sim).item()

    def _kernel(self, v1: torch.Tensor, v2: torch.Tensor) -> float:
        """RBF kernel using geodesic distance."""
        dist = self._geodesic_distance(v1, v2)
        return math.exp(-dist**2 / (2 * self.lengthscale**2))

    def predict(self, v: torch.Tensor) -> Tuple[float, float]:
        """Predict mean and variance at v."""
        if len(self.V_observed) == 0:
            return 0.0, 1.0

        v = v.detach().cpu().float()
        n = len(self.V_observed)

        # Kernel matrix
        K = torch.zeros(n, n)
        for i in range(n):
            for j in range(n):
                K[i, j] = self._kernel(self.V_observed[i], self.V_observed[j])
        K += self.noise_variance * torch.eye(n)

        # Kernel vector
        k = torch.tensor([self._kernel(v, self.V_observed[i]) for i in range(n)])

        # GP prediction
        y = torch.tensor(self.y_observed, dtype=torch.float32)

        try:
            L = torch.linalg.cholesky(K)
            alpha = torch.cholesky_solve(y.unsqueeze(1), L).squeeze()
            mu = torch.dot(k, alpha).item()

            v_solve = torch.cholesky_solve(k.unsqueeze(1), L).squeeze()
            var = max(0.0, 1.0 - torch.dot(k, v_solve).item())
        except Exception:
            # Fallback if Cholesky fails
            mu = y.mean().item()
            var = 1.0

        return mu, var


class SingleVectorDiscovery:
    """
    Discover optimal single vector for refusal ablation.

    Searches over unit sphere in R^{hidden_dim} to find the vector
    that best reduces refusal when ablated at all layers with ACE.
    """

    def __init__(
        self,
        scorer: SingleVectorScorer,
        config: SingleVectorDiscoveryConfig = None,
        v_init: Optional[torch.Tensor] = None,
    ):
        self.scorer = scorer
        self.config = config or SingleVectorDiscoveryConfig()
        self.hidden_dim = scorer.hidden_dim

        # Initialize with mean-diff direction if not provided
        if v_init is None:
            v_init = scorer.get_mean_diff_direction()
        self.v_init = v_init

        # GP for refusal score (primary objective)
        self.gp = SingleVectorGP(
            hidden_dim=self.hidden_dim,
            lengthscale=self.config.kernel_lengthscale,
            noise_variance=self.config.noise_variance,
        )

        # Storage
        self.V_observed = []
        self.scores_observed = {'refusal_score': [], 'kl_score': []}

    def discover(self) -> SingleVectorResults:
        """Run discovery to find optimal single vector."""
        print(f"\n{'='*60}")
        print("SINGLE-VECTOR DISCOVERY")
        print(f"{'='*60}")
        print(f"Hidden dim: {self.hidden_dim}")
        print(f"Init samples: {self.config.n_init_samples}")
        print(f"Iterations: {self.config.n_iterations}")

        # Compute dynamic threshold if needed
        if self.config.use_boundary_search and self.config.boundary_threshold is None:
            baseline_harmful, baseline_harmless = self.scorer.compute_baseline_refusal()
            gap = baseline_harmful - baseline_harmless
            threshold = baseline_harmful - self.config.boundary_fraction * gap
            print(f"\nDynamic threshold: {threshold:.4f}")
            print(f"  (baseline_harmful={baseline_harmful:.4f}, gap={gap:.4f})")
        else:
            threshold = self.config.boundary_threshold

        # Evaluate all r_i directions as initialization
        print("\nEvaluating all r_i directions...")
        r_directions = self.scorer.get_all_r_directions()

        for i, r_i in enumerate(r_directions):
            scores = self.scorer.score(r_i)
            self._add_observation(r_i, scores)

            if (i + 1) % 5 == 0 or i == len(r_directions) - 1:
                best_idx = min(range(len(self.scores_observed['refusal_score'])),
                              key=lambda j: self.scores_observed['refusal_score'][j])
                print(f"  r_{i}: refusal={scores['refusal_score']:.4f}, "
                      f"best so far={self.scores_observed['refusal_score'][best_idx]:.4f}")

        # Also evaluate mean-diff direction
        v_init_norm = self.v_init / (self.v_init.norm() + 1e-8)
        scores = self.scorer.score(v_init_norm)
        self._add_observation(v_init_norm, scores)
        print(f"  mean-diff: refusal={scores['refusal_score']:.4f}")

        # Additional random samples if needed
        n_additional = max(0, self.config.n_init_samples - len(r_directions) - 1)
        if n_additional > 0:
            print(f"\nAdditional random samples ({n_additional})...")
            for i in range(n_additional):
                # Sample near best so far
                best_idx = min(range(len(self.scores_observed['refusal_score'])),
                              key=lambda j: self.scores_observed['refusal_score'][j])
                v = self._sample_near(self.V_observed[best_idx], concentration=10.0)
                scores = self.scorer.score(v)
                self._add_observation(v, scores)

        # GP-guided optimization
        print(f"\nGP-guided optimization ({self.config.n_iterations} iterations)...")
        for i in range(self.config.n_iterations):
            # Generate candidates
            candidates = self._generate_candidates()

            # Select best by UCB
            best_ucb = float('-inf')
            best_v = None

            for v in candidates:
                mu, var = self.gp.predict(v)
                # We want to MINIMIZE refusal score, so negate
                ucb = -mu + self.config.beta * math.sqrt(var)
                if ucb > best_ucb:
                    best_ucb = ucb
                    best_v = v

            # Evaluate
            scores = self.scorer.score(best_v)
            self._add_observation(best_v, scores)

            if (i + 1) % 10 == 0:
                best_idx = min(range(len(self.scores_observed['refusal_score'])),
                              key=lambda j: self.scores_observed['refusal_score'][j])
                print(f"  {i+1}/{self.config.n_iterations}: "
                      f"best_refusal={self.scores_observed['refusal_score'][best_idx]:.4f}, "
                      f"latest={scores['refusal_score']:.4f}")

        # Find best and Pareto frontier
        best_idx = min(range(len(self.scores_observed['refusal_score'])),
                      key=lambda i: self.scores_observed['refusal_score'][i])

        best_vector = self.V_observed[best_idx]
        best_scores = {
            'refusal_score': self.scores_observed['refusal_score'][best_idx],
            'kl_score': self.scores_observed['kl_score'][best_idx],
        }

        # Compute Pareto frontier
        pareto_indices = self._compute_pareto_frontier()
        pareto_vectors = torch.stack([self.V_observed[i] for i in pareto_indices])
        pareto_scores = {
            'refusal_score': torch.tensor([self.scores_observed['refusal_score'][i] for i in pareto_indices]),
            'kl_score': torch.tensor([self.scores_observed['kl_score'][i] for i in pareto_indices]),
        }

        # Compute hypervolume
        hv = self._compute_hypervolume(pareto_scores)

        print(f"\n{'='*60}")
        print("DISCOVERY COMPLETE")
        print(f"{'='*60}")
        print(f"Total measurements: {len(self.V_observed)}")
        print(f"Best refusal score: {best_scores['refusal_score']:.4f}")
        print(f"Best KL score: {best_scores['kl_score']:.4f}")
        print(f"Pareto vectors: {len(pareto_indices)}")
        print(f"Hypervolume: {hv:.4f}")

        return SingleVectorResults(
            best_vector=best_vector,
            best_scores=best_scores,
            V_observed=torch.stack(self.V_observed),
            scores_observed={k: torch.tensor(v) for k, v in self.scores_observed.items()},
            pareto_vectors=pareto_vectors,
            pareto_scores=pareto_scores,
            hypervolume=hv,
            n_measurements=len(self.V_observed),
        )

    def _add_observation(self, v: torch.Tensor, scores: Dict[str, float]):
        """Add observation to storage and GP."""
        v_cpu = v.detach().cpu()
        self.V_observed.append(v_cpu)
        self.scores_observed['refusal_score'].append(scores['refusal_score'])
        self.scores_observed['kl_score'].append(scores['kl_score'])
        self.gp.add_observation(v_cpu, scores['refusal_score'])

    def _sample_near(self, v: torch.Tensor, concentration: float = 10.0) -> torch.Tensor:
        """Sample from von Mises-Fisher distribution centered at v."""
        # Simple approximation: add Gaussian noise and renormalize
        noise = torch.randn_like(v) / concentration
        v_new = v + noise
        return v_new / (v_new.norm() + 1e-8)

    def _generate_candidates(self) -> List[torch.Tensor]:
        """Generate candidate vectors for acquisition."""
        candidates = []

        # Near current best
        best_idx = min(range(len(self.scores_observed['refusal_score'])),
                      key=lambda i: self.scores_observed['refusal_score'][i])
        v_best = self.V_observed[best_idx]

        for _ in range(self.config.n_candidates // 2):
            candidates.append(self._sample_near(v_best, concentration=20.0))

        # Near random observed points
        for _ in range(self.config.n_candidates // 2):
            idx = torch.randint(len(self.V_observed), (1,)).item()
            candidates.append(self._sample_near(self.V_observed[idx], concentration=10.0))

        return candidates

    def _compute_pareto_frontier(self) -> List[int]:
        """Compute Pareto frontier indices (minimizing both objectives)."""
        n = len(self.V_observed)
        pareto = []

        for i in range(n):
            is_dominated = False
            r_i = self.scores_observed['refusal_score'][i]
            k_i = self.scores_observed['kl_score'][i]

            for j in range(n):
                if i == j:
                    continue
                r_j = self.scores_observed['refusal_score'][j]
                k_j = self.scores_observed['kl_score'][j]

                # j dominates i if j is <= on both and < on at least one
                if r_j <= r_i and k_j <= k_i and (r_j < r_i or k_j < k_i):
                    is_dominated = True
                    break

            if not is_dominated:
                pareto.append(i)

        return pareto

    def _compute_hypervolume(self, pareto_scores: Dict[str, torch.Tensor]) -> float:
        """Compute hypervolume indicator."""
        if len(pareto_scores['refusal_score']) == 0:
            return 0.0

        # Reference point (worst case)
        ref_refusal = max(self.scores_observed['refusal_score']) + 1
        ref_kl = max(self.scores_observed['kl_score']) + 1

        # Sort by refusal score
        refusal = pareto_scores['refusal_score'].numpy()
        kl = pareto_scores['kl_score'].numpy()

        indices = refusal.argsort()
        refusal = refusal[indices]
        kl = kl[indices]

        # Compute area
        hv = 0.0
        prev_kl = ref_kl

        for r, k in zip(refusal, kl):
            if k < prev_kl:
                hv += (ref_refusal - r) * (prev_kl - k)
                prev_kl = k

        return hv
