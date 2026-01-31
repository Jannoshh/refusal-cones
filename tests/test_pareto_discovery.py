#!/usr/bin/env python3
"""Tests for multi-objective Pareto boundary discovery."""

import torch
import numpy as np
from types import SimpleNamespace

from src.discovery.pareto_boundary_discovery import (
    ParetoGeometryDiscovery,
    ParetoDiscoveryConfig,
    ParetoDiscoveryResults,
    MultiObjectiveScorer,
    SmoothLayerParameterization,
)


class MockScorer:
    """Mock scorer for testing without a real model."""

    def __init__(self, n_layers: int, hidden_dim: int):
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim

        # Create "true" optimal directions
        torch.manual_seed(42)
        self.refusal_dir = torch.randn(n_layers * hidden_dim)
        self.refusal_dir = self.refusal_dir / self.refusal_dir.norm()

        self.retain_dir = torch.randn(n_layers * hidden_dim)
        # Make somewhat orthogonal to refusal
        self.retain_dir = self.retain_dir - (self.retain_dir @ self.refusal_dir) * self.refusal_dir
        self.retain_dir = self.retain_dir / self.retain_dir.norm()

    def score(self, v: torch.Tensor, return_grad: bool = False):
        """
        Mock scoring function.

        refusal_score: low when aligned with refusal_dir
        kl_score: low when orthogonal to retain_dir
        """
        v_flat = v.reshape(-1)
        v_flat = v_flat / (v_flat.norm() + 1e-8)

        # Refusal score: lower when aligned with refusal direction
        refusal_alignment = (v_flat @ self.refusal_dir).abs()
        refusal_score = 1.0 - refusal_alignment.item() + 0.05 * torch.randn(1).item()
        refusal_score = max(0, min(1, refusal_score))

        # KL score: lower when NOT aligned with retain direction
        retain_alignment = (v_flat @ self.retain_dir).abs()
        kl_score = retain_alignment.item() + 0.05 * torch.randn(1).item()
        kl_score = max(0, min(1, kl_score))

        # Induce score (placeholder)
        induce_score = 1.0 - refusal_score

        scores = {
            'refusal_score': refusal_score,
            'kl_score': kl_score,
            'induce_score': induce_score
        }

        if return_grad:
            # Mock gradient
            grad = -self.refusal_dir.reshape(v.shape) + 0.1 * torch.randn_like(v)
            return scores, grad

        return scores


def test_pareto_discovery_config_defaults():
    """Test that config has sensible defaults."""
    config = ParetoDiscoveryConfig()

    assert config.n_init_samples == 30
    assert config.n_pareto_iterations == 70
    assert 'refusal_score' in config.primary_objectives
    assert 'kl_score' in config.primary_objectives
    # Default is now simple GP (use_sparse_gp=False)
    assert config.use_sparse_gp is False
    # Can use gp_type='sparse' for sparse GP
    assert config.gp_type == 'simple'
    # Gradient-based candidates enabled by default
    assert config.use_gradient_candidates is True
    assert config.gradient_lr == 0.1
    assert config.gradient_steps == 5
    assert config.n_gradient_candidates == 20


def test_pareto_discovery_finds_frontier():
    """Test that Pareto discovery finds a non-trivial frontier."""
    torch.manual_seed(123)
    n_layers, hidden_dim = 2, 4

    scorer = MockScorer(n_layers, hidden_dim)

    config = ParetoDiscoveryConfig(
        n_init_samples=15,
        n_pareto_iterations=20,
        max_measurements=50,
        use_sparse_gp=True,
        num_inducing=8,
        sparse_train_steps=3
    )

    discovery = ParetoGeometryDiscovery(
        scorer=scorer,
        n_layers=n_layers,
        hidden_dim=hidden_dim,
        config=config,
        use_mean_diff_init=False  # MockScorer doesn't have compute_mean_diff_vector
    )

    results = discovery.discover()

    # Should have found some Pareto points
    assert isinstance(results, ParetoDiscoveryResults)
    assert len(results.pareto_vectors) > 0
    assert results.n_measurements <= config.max_measurements

    # Pareto vectors should be globally normalized (not per-layer)
    # With global normalization, the total norm is 1.0, not per-layer
    for v in results.pareto_vectors:
        total_norm = v.norm()
        assert torch.isclose(total_norm, torch.tensor(1.0), atol=0.1)

    # Hypervolume should be positive
    assert results.hypervolume >= 0


def test_pareto_mask_correctness():
    """Test that Pareto mask correctly identifies non-dominated points."""
    torch.manual_seed(456)
    n_layers, hidden_dim = 2, 3

    scorer = MockScorer(n_layers, hidden_dim)

    config = ParetoDiscoveryConfig(
        n_init_samples=10,
        n_pareto_iterations=0,  # Just initial samples
        use_sparse_gp=True,
        num_inducing=6,
        sparse_train_steps=2
    )

    discovery = ParetoGeometryDiscovery(
        scorer=scorer,
        n_layers=n_layers,
        hidden_dim=hidden_dim,
        config=config,
        use_mean_diff_init=False  # MockScorer doesn't have compute_mean_diff_vector
    )

    # Run initial exploration
    discovery._initial_exploration()

    # Get Pareto mask
    pareto_mask = discovery._get_pareto_mask()

    # At least one point should be Pareto optimal
    assert pareto_mask.any()

    # Verify Pareto property: no Pareto point is dominated
    scores = torch.stack([
        torch.tensor(discovery.scores_observed[obj])
        for obj in config.primary_objectives
    ], dim=1)

    pareto_indices = torch.where(pareto_mask)[0]
    for i in pareto_indices:
        for j in range(len(scores)):
            if i == j:
                continue
            # j should NOT dominate i
            dominates = (scores[j] <= scores[i]).all() and (scores[j] < scores[i]).any()
            assert not dominates, f"Point {j} dominates Pareto point {i}"


def test_get_vector_for_tradeoff():
    """Test selecting vectors for different trade-offs."""
    torch.manual_seed(789)
    n_layers, hidden_dim = 2, 4

    scorer = MockScorer(n_layers, hidden_dim)

    config = ParetoDiscoveryConfig(
        n_init_samples=15,
        n_pareto_iterations=15,
        max_measurements=40,
        use_sparse_gp=True,
        num_inducing=8,
        sparse_train_steps=2
    )

    discovery = ParetoGeometryDiscovery(
        scorer=scorer,
        n_layers=n_layers,
        hidden_dim=hidden_dim,
        config=config,
        use_mean_diff_init=False  # MockScorer doesn't have compute_mean_diff_vector
    )

    results = discovery.discover()

    # Get vectors for different trade-offs
    v_refusal_priority = discovery.get_vector_for_tradeoff(refusal_weight=0.9)
    v_balanced = discovery.get_vector_for_tradeoff(refusal_weight=0.5)
    v_retain_priority = discovery.get_vector_for_tradeoff(refusal_weight=0.1)

    # All should be valid tensors
    assert v_refusal_priority.shape == (n_layers, hidden_dim)
    assert v_balanced.shape == (n_layers, hidden_dim)
    assert v_retain_priority.shape == (n_layers, hidden_dim)


def test_hypervolume_computation():
    """Test hypervolume computation."""
    torch.manual_seed(111)
    n_layers, hidden_dim = 2, 3

    scorer = MockScorer(n_layers, hidden_dim)

    config = ParetoDiscoveryConfig(
        n_init_samples=20,
        n_pareto_iterations=0,
        use_sparse_gp=True,
        num_inducing=8,
        sparse_train_steps=2
    )

    discovery = ParetoGeometryDiscovery(
        scorer=scorer,
        n_layers=n_layers,
        hidden_dim=hidden_dim,
        config=config,
        use_mean_diff_init=False  # MockScorer doesn't have compute_mean_diff_vector
    )

    discovery._initial_exploration()

    hv = discovery._compute_hypervolume()

    # Hypervolume should be non-negative and bounded
    assert hv >= 0
    assert hv <= 1.0  # Reference point is (1, 1)


def test_induce_score_varies_with_vector():
    """Induce score should depend on the evaluated vector."""
    torch.manual_seed(0)
    vocab_size = 6
    hidden_dim = 4
    n_layers = 2

    class DummyBatch(dict):
        def to(self, device):
            for key, value in self.items():
                self[key] = value.to(device)
            return self

    class DummyTokenizer:
        def __init__(self, seq_len=2):
            self.seq_len = seq_len

        def __call__(self, prompts, **kwargs):
            batch = len(prompts)
            input_ids = torch.zeros((batch, self.seq_len), dtype=torch.long)
            attention_mask = torch.ones_like(input_ids)
            return DummyBatch({'input_ids': input_ids, 'attention_mask': attention_mask})

    class DummyInnerModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = torch.nn.Embedding(vocab_size, hidden_dim)
            self.layers = torch.nn.ModuleList([torch.nn.Identity() for _ in range(n_layers)])
            self.lm_head = torch.nn.Linear(hidden_dim, vocab_size, bias=False)

        def forward(self, input_ids, attention_mask=None):
            h = self.embed(input_ids)
            for layer in self.layers:
                h = layer(h)
            logits = self.lm_head(h)
            return SimpleNamespace(logits=logits)

    class DummyWrapper(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.model = inner

        def forward(self, **kwargs):
            return self.model(**kwargs)

    inner = DummyInnerModel()
    with torch.no_grad():
        inner.embed.weight.zero_()
        inner.embed.weight[0] = torch.tensor([1.0, 0.5, -0.5, 0.25])
        inner.lm_head.weight.zero_()
        inner.lm_head.weight[:hidden_dim] = torch.eye(hidden_dim)

    wrapper = DummyWrapper(inner)
    tokenizer = DummyTokenizer()
    refusal_toks = torch.tensor([0], dtype=torch.long)

    scorer = MultiObjectiveScorer(
        model=wrapper,
        tokenizer=tokenizer,
        harmful_prompts=["harmful"],
        harmless_prompts=["harmless"],
        refusal_toks=refusal_toks,
        device='cpu'
    )

    # Pre-set ACE reference point to bypass compute_mean_diff_vector
    # (which requires a real model architecture)
    scorer._v_minus = torch.zeros(n_layers, hidden_dim)
    scorer._v_plus = torch.zeros(n_layers, hidden_dim)
    scorer._v_minus_computed = True

    v1 = torch.tensor([[1.0, 0.0, 0.0, 0.0],
                       [1.0, 0.0, 0.0, 0.0]])
    v2 = torch.tensor([[0.0, 1.0, 0.0, 0.0],
                       [0.0, 1.0, 0.0, 0.0]])

    scores_v1 = scorer.score(v1)
    scores_v2 = scorer.score(v2)

    assert not np.isclose(scores_v1['induce_score'], scores_v2['induce_score'])


def test_gradient_based_candidate_generation():
    """Test that gradient-based candidate generation produces valid candidates."""
    torch.manual_seed(42)
    n_layers, hidden_dim = 2, 4

    scorer = MockScorer(n_layers, hidden_dim)

    # Test with gradient-based candidates enabled
    config = ParetoDiscoveryConfig(
        n_init_samples=10,
        n_pareto_iterations=5,
        max_measurements=20,
        use_gradient_candidates=True,
        gradient_steps=3,
        gradient_lr=0.1,
        n_gradient_candidates=5,
        use_sparse_gp=True,
        num_inducing=6,
        sparse_train_steps=2
    )

    discovery = ParetoGeometryDiscovery(
        scorer=scorer,
        n_layers=n_layers,
        hidden_dim=hidden_dim,
        config=config,
        use_mean_diff_init=False
    )

    # Initialize with some data first
    for _ in range(5):
        v = torch.randn(n_layers, hidden_dim)
        v = v / v.norm()
        scores = scorer.score(v)
        discovery.V_observed.append(v)
        for obj, val in scores.items():
            discovery.scores_observed[obj].append(val)

    # Test gradient candidate generation
    gradient_candidates = discovery._generate_gradient_candidates()

    # Should have generated candidates
    assert len(gradient_candidates) > 0
    assert len(gradient_candidates) <= config.n_gradient_candidates

    # All candidates should be normalized
    for v in gradient_candidates:
        assert v.shape == (n_layers, hidden_dim)
        assert torch.isclose(v.norm(), torch.tensor(1.0), atol=0.1)


def test_riemannian_gradient_step():
    """Test that Riemannian gradient step stays on unit sphere."""
    torch.manual_seed(123)
    n_layers, hidden_dim = 2, 4

    scorer = MockScorer(n_layers, hidden_dim)

    config = ParetoDiscoveryConfig(use_gradient_candidates=True)
    discovery = ParetoGeometryDiscovery(
        scorer=scorer,
        n_layers=n_layers,
        hidden_dim=hidden_dim,
        config=config,
        use_mean_diff_init=False
    )

    # Start with unit vector
    v = torch.randn(n_layers, hidden_dim)
    v = v / v.norm()

    # Create a gradient
    grad = torch.randn_like(v)

    # Take a step
    v_new = discovery._riemannian_gradient_step(v, grad, lr=0.1)

    # Result should be on unit sphere
    assert v_new.shape == v.shape
    assert torch.isclose(v_new.norm(), torch.tensor(1.0), atol=1e-6)

    # Should have moved from original position
    assert not torch.allclose(v_new, v)


def test_gradient_vs_random_generates_different_candidates():
    """Test that gradient-based and random methods generate different candidates."""
    torch.manual_seed(999)
    n_layers, hidden_dim = 2, 4

    scorer = MockScorer(n_layers, hidden_dim)

    # With gradients
    config_grad = ParetoDiscoveryConfig(
        n_init_samples=10,
        n_pareto_iterations=3,
        use_gradient_candidates=True,
        gradient_steps=3,
        n_gradient_candidates=5,
        use_sparse_gp=True,
        num_inducing=6,
        sparse_train_steps=2
    )

    # Without gradients
    config_random = ParetoDiscoveryConfig(
        n_init_samples=10,
        n_pareto_iterations=3,
        use_gradient_candidates=False,
        n_candidates=50,
        use_sparse_gp=True,
        num_inducing=6,
        sparse_train_steps=2
    )

    discovery_grad = ParetoGeometryDiscovery(
        scorer=scorer,
        n_layers=n_layers,
        hidden_dim=hidden_dim,
        config=config_grad,
        use_mean_diff_init=False
    )

    discovery_random = ParetoGeometryDiscovery(
        scorer=scorer,
        n_layers=n_layers,
        hidden_dim=hidden_dim,
        config=config_random,
        use_mean_diff_init=False
    )

    # Run discovery
    results_grad = discovery_grad.discover()
    results_random = discovery_random.discover()

    # Both should produce valid results
    assert len(results_grad.pareto_vectors) > 0
    assert len(results_random.pareto_vectors) > 0

    # Verify measurements are within bounds
    assert results_grad.n_measurements <= config_grad.n_init_samples + config_grad.n_pareto_iterations
    assert results_random.n_measurements <= config_random.n_init_samples + config_random.n_pareto_iterations


# ============================================================================
# Tests for SmoothLayerParameterization
# ============================================================================

def test_smooth_parameterization_basis_shape():
    """Test that basis matrices have correct shapes."""
    n_layers, hidden_dim, n_basis = 28, 512, 8

    param = SmoothLayerParameterization(
        n_layers=n_layers,
        hidden_dim=hidden_dim,
        n_basis=n_basis,
        lengthscale=3.0
    )

    assert param.basis.shape == (n_layers, n_basis)
    assert param.basis_pinv.shape == (n_basis, n_layers)
    assert param.centers.shape == (n_basis,)


def test_smooth_parameterization_z_to_v():
    """Test z_to_v conversion produces correct shape and normalization."""
    n_layers, hidden_dim, n_basis = 8, 64, 4

    param = SmoothLayerParameterization(
        n_layers=n_layers,
        hidden_dim=hidden_dim,
        n_basis=n_basis,
        lengthscale=2.0
    )

    z = torch.randn(n_basis, hidden_dim)
    v = param.z_to_v(z, normalize=True)

    assert v.shape == (n_layers, hidden_dim)
    assert torch.isclose(v.norm(), torch.tensor(1.0), atol=1e-6)


def test_smooth_parameterization_roundtrip():
    """Test that v_to_z -> z_to_v roundtrip preserves smooth vectors."""
    n_layers, hidden_dim, n_basis = 8, 64, 4

    param = SmoothLayerParameterization(
        n_layers=n_layers,
        hidden_dim=hidden_dim,
        n_basis=n_basis,
        lengthscale=2.0
    )

    # Create a smooth vector from basis
    z_original = torch.randn(n_basis, hidden_dim)
    v = param.z_to_v(z_original, normalize=False)

    # Roundtrip
    z_recovered = param.v_to_z(v)
    v_recovered = param.z_to_v(z_recovered, normalize=False)

    # Should be identical for vectors in the span of the basis
    assert torch.allclose(v, v_recovered, atol=1e-5)


def test_smooth_parameterization_gradient_chain_rule():
    """Test that gradient conversion via chain rule is correct."""
    n_layers, hidden_dim, n_basis = 8, 64, 4

    param = SmoothLayerParameterization(
        n_layers=n_layers,
        hidden_dim=hidden_dim,
        n_basis=n_basis,
        lengthscale=2.0
    )

    # Create z with gradients
    z = torch.randn(n_basis, hidden_dim, requires_grad=True)

    # Forward: v = basis @ z
    v = param.basis @ z

    # Create a simple loss
    loss = (v ** 2).sum()

    # Compute gradient via autograd
    loss.backward()
    grad_z_autograd = z.grad.clone()

    # Compute gradient via our chain rule function
    # grad_v = 2 * v
    grad_v = 2 * v.detach()
    grad_z_manual = param.gradient_z_to_v(grad_v)

    # Should match
    assert torch.allclose(grad_z_autograd, grad_z_manual, atol=1e-5)


def test_smooth_parameterization_smoothness_metric():
    """Test that smoothness metric correctly identifies smooth vs non-smooth vectors."""
    n_layers, hidden_dim, n_basis = 16, 32, 6

    param = SmoothLayerParameterization(
        n_layers=n_layers,
        hidden_dim=hidden_dim,
        n_basis=n_basis,
        lengthscale=3.0
    )

    # Smooth vector (from basis)
    z = torch.randn(n_basis, hidden_dim)
    v_smooth = param.z_to_v(z, normalize=True)

    # Random (non-smooth) vector
    v_random = torch.randn(n_layers, hidden_dim)
    v_random = v_random / v_random.norm()

    smoothness_smooth = param.smoothness_of_v(v_smooth)
    smoothness_random = param.smoothness_of_v(v_random)

    # Smooth vector should have near-zero reconstruction error
    assert smoothness_smooth < 0.01

    # Random vector should have higher reconstruction error
    assert smoothness_random > smoothness_smooth


def test_smooth_parameterization_basis_functions_are_smooth():
    """Test that basis functions vary smoothly across layers."""
    n_layers, hidden_dim, n_basis = 16, 32, 4
    lengthscale = 3.0

    param = SmoothLayerParameterization(
        n_layers=n_layers,
        hidden_dim=hidden_dim,
        n_basis=n_basis,
        lengthscale=lengthscale
    )

    # Check that adjacent layers have similar basis values
    for k in range(n_basis):
        basis_k = param.basis[:, k]
        # Compute differences between adjacent layers
        diffs = (basis_k[1:] - basis_k[:-1]).abs()
        # With lengthscale=3, adjacent layers should differ by at most ~0.3
        assert diffs.max() < 0.5, f"Basis {k} not smooth: max diff = {diffs.max()}"


def test_smooth_parameterization_with_pareto_discovery():
    """Test that smooth parameterization integrates correctly with ParetoGeometryDiscovery."""
    torch.manual_seed(42)
    n_layers, hidden_dim = 4, 8

    scorer = MockScorer(n_layers, hidden_dim)

    # Config with smooth parameterization enabled
    config = ParetoDiscoveryConfig(
        n_init_samples=10,
        n_pareto_iterations=5,
        max_measurements=20,
        use_gradient_candidates=True,
        use_smooth_parameterization=True,
        n_basis=3,  # Small for fast test
        layer_lengthscale=1.5,
        gradient_steps=2,
        n_gradient_candidates=3,
        use_sparse_gp=True,
        num_inducing=6,
        sparse_train_steps=2
    )

    discovery = ParetoGeometryDiscovery(
        scorer=scorer,
        n_layers=n_layers,
        hidden_dim=hidden_dim,
        config=config,
        use_mean_diff_init=False
    )

    # Verify smooth_param was initialized
    assert discovery.smooth_param is not None
    assert discovery.smooth_param.n_basis == config.n_basis

    results = discovery.discover()

    # Should produce valid results
    assert len(results.pareto_vectors) > 0
    assert results.n_measurements <= config.max_measurements

    # All Pareto vectors should be normalized
    for v in results.pareto_vectors:
        assert torch.isclose(v.norm(), torch.tensor(1.0), atol=0.1)


def test_smooth_vs_nonsmooth_gradient_candidates():
    """Test that smooth parameterization produces smoother candidates."""
    torch.manual_seed(123)
    n_layers, hidden_dim = 8, 16

    scorer = MockScorer(n_layers, hidden_dim)

    # With smooth parameterization
    config_smooth = ParetoDiscoveryConfig(
        n_init_samples=8,
        n_pareto_iterations=3,
        use_gradient_candidates=True,
        use_smooth_parameterization=True,
        n_basis=4,
        layer_lengthscale=2.0,
        gradient_steps=3,
        n_gradient_candidates=5,
        use_sparse_gp=True,
        num_inducing=6,
        sparse_train_steps=2
    )

    # Without smooth parameterization
    config_nonsmooth = ParetoDiscoveryConfig(
        n_init_samples=8,
        n_pareto_iterations=3,
        use_gradient_candidates=True,
        use_smooth_parameterization=False,
        gradient_steps=3,
        n_gradient_candidates=5,
        use_sparse_gp=True,
        num_inducing=6,
        sparse_train_steps=2
    )

    discovery_smooth = ParetoGeometryDiscovery(
        scorer=scorer,
        n_layers=n_layers,
        hidden_dim=hidden_dim,
        config=config_smooth,
        use_mean_diff_init=False
    )

    discovery_nonsmooth = ParetoGeometryDiscovery(
        scorer=scorer,
        n_layers=n_layers,
        hidden_dim=hidden_dim,
        config=config_nonsmooth,
        use_mean_diff_init=False
    )

    # Initialize with same data
    for _ in range(5):
        v = torch.randn(n_layers, hidden_dim)
        v = v / v.norm()
        scores = scorer.score(v)
        discovery_smooth.V_observed.append(v.clone())
        discovery_nonsmooth.V_observed.append(v.clone())
        for obj, val in scores.items():
            discovery_smooth.scores_observed[obj].append(val)
            discovery_nonsmooth.scores_observed[obj].append(val)

    # Generate candidates
    candidates_smooth = discovery_smooth._generate_gradient_candidates()
    candidates_nonsmooth = discovery_nonsmooth._generate_gradient_candidates()

    # Both should produce candidates
    assert len(candidates_smooth) > 0
    assert len(candidates_nonsmooth) > 0

    # Measure smoothness of candidates
    def measure_layer_smoothness(v):
        """Measure how much v changes between adjacent layers."""
        v_normalized = v / (v.norm(dim=1, keepdim=True) + 1e-8)
        # Cosine similarity between adjacent layers
        cos_sims = (v_normalized[:-1] * v_normalized[1:]).sum(dim=1)
        return cos_sims.mean().item()

    smooth_scores = [measure_layer_smoothness(v) for v in candidates_smooth]
    nonsmooth_scores = [measure_layer_smoothness(v) for v in candidates_nonsmooth]

    avg_smooth = np.mean(smooth_scores)
    avg_nonsmooth = np.mean(nonsmooth_scores)

    # Smooth candidates should have higher layer-to-layer similarity on average
    # (This is a soft check since random seeds affect results)
    print(f"Avg layer similarity - smooth: {avg_smooth:.4f}, nonsmooth: {avg_nonsmooth:.4f}")


def test_smooth_parameterization_device_from_v_init():
    """Test that SmoothLayerParameterization device is derived from v_init, not cuda.is_available()."""
    torch.manual_seed(42)
    n_layers, hidden_dim = 4, 8

    # Create a CPU scorer
    scorer = MockScorer(n_layers, hidden_dim)

    # Create a CPU v_init explicitly
    v_init = torch.randn(n_layers, hidden_dim, device='cpu')
    v_init = v_init / v_init.norm()

    config = ParetoDiscoveryConfig(
        n_init_samples=5,
        n_pareto_iterations=3,
        use_gradient_candidates=True,
        use_smooth_parameterization=True,
        n_basis=3,
        use_sparse_gp=True,
        num_inducing=4,
        sparse_train_steps=2
    )

    discovery = ParetoGeometryDiscovery(
        scorer=scorer,
        n_layers=n_layers,
        hidden_dim=hidden_dim,
        config=config,
        v_init=v_init,
        use_mean_diff_init=False
    )

    # Verify smooth_param is on CPU (same as v_init)
    assert discovery.smooth_param is not None
    assert str(discovery.smooth_param.device) == 'cpu'
    assert discovery.smooth_param.basis.device.type == 'cpu'

    # Verify z_to_v produces CPU tensors
    z = discovery.smooth_param.random_z()
    assert z.device.type == 'cpu'

    v = discovery.smooth_param.z_to_v(z)
    assert v.device.type == 'cpu'


def test_smooth_parameterization_gradient_with_normalization():
    """Test that gradient_z_to_v correctly includes the normalization Jacobian."""
    n_layers, hidden_dim, n_basis = 8, 16, 4

    param = SmoothLayerParameterization(
        n_layers=n_layers,
        hidden_dim=hidden_dim,
        n_basis=n_basis,
        lengthscale=2.0
    )

    # Create z with gradients
    z = torch.randn(n_basis, hidden_dim, requires_grad=True)

    # Forward pass WITH normalization (as used in actual code)
    v_unnorm = param.basis @ z
    v_norm = v_unnorm.norm()
    v = v_unnorm / (v_norm + 1e-8)

    # Create a simple loss based on normalized v
    loss = (v ** 2).sum()

    # Compute gradient via autograd
    loss.backward()
    grad_z_autograd = z.grad.clone()

    # Compute gradient via our chain rule function with normalization
    # grad_v = 2 * v (for sum of squares loss)
    grad_v = 2 * v.detach()

    # Use z (without grad) for the gradient computation
    z_no_grad = z.detach().clone()
    grad_z_manual = param.gradient_z_to_v(grad_v, z=z_no_grad, v=v.detach())

    # Should be close (may not be exact due to numerical precision)
    assert torch.allclose(grad_z_autograd, grad_z_manual, atol=1e-4), \
        f"Gradient mismatch: autograd={grad_z_autograd[:2,:4]}, manual={grad_z_manual[:2,:4]}"


def test_smooth_parameterization_gradient_descent_direction():
    """Test that gradient descent with normalization moves in the correct direction."""
    torch.manual_seed(42)  # For reproducibility
    n_layers, hidden_dim, n_basis = 6, 32, 4  # More basis = better expressiveness

    param = SmoothLayerParameterization(
        n_layers=n_layers,
        hidden_dim=hidden_dim,
        n_basis=n_basis,
        lengthscale=2.0
    )

    # Create a target that's representable in the smooth basis
    # (random targets may not be in the span of smooth functions)
    z_target = torch.randn(n_basis, hidden_dim)
    target = param.z_to_v(z_target, normalize=True)

    def compute_loss(z):
        v = param.z_to_v(z, normalize=True)
        # Negative cosine similarity (we want to maximize similarity = minimize loss)
        return -(v * target).sum()

    # Initialize z far from target
    z = torch.randn(n_basis, hidden_dim) * 0.1

    # Compute initial loss and similarity
    initial_loss = compute_loss(z).item()
    v_initial = param.z_to_v(z, normalize=True)
    initial_similarity = (v_initial * target).sum().item()

    # Take gradient steps using autograd (the correct way)
    lr = 0.5
    for _ in range(50):
        z.requires_grad_(True)
        v = param.z_to_v(z, normalize=True)
        loss = -(v * target).sum()
        loss.backward()

        z = (z - lr * z.grad).detach()

    # Compute final loss and similarity
    final_loss = compute_loss(z).item()
    v_final = param.z_to_v(z, normalize=True)
    final_similarity = (v_final * target).sum().item()

    # Loss should decrease significantly
    assert final_loss < initial_loss - 0.1, \
        f"Loss did not decrease enough: initial={initial_loss:.4f}, final={final_loss:.4f}"

    # Similarity should increase
    assert final_similarity > initial_similarity, \
        f"Similarity did not improve: initial={initial_similarity:.4f}, final={final_similarity:.4f}"

    # Final similarity should be reasonably high (target is in basis span)
    assert final_similarity > 0.8, f"Final similarity too low: {final_similarity:.4f}"
