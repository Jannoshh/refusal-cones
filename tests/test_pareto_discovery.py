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
