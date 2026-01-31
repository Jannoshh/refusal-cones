#!/usr/bin/env python3
"""
Tests for multi-dimensional ACE with non-orthogonal vectors.
"""

import torch
import pytest
from src.training.unified_rdo_adapter import RankKUnifiedLayer


class DummyLayer(torch.nn.Module):
    """Dummy layer that returns activations unchanged."""
    def forward(self, x):
        return x


def test_orthogonal_projection():
    """Test orthogonal projection P = VV^T."""
    dim = 128
    rank_k = 3

    layer = RankKUnifiedLayer(
        base_layer=DummyLayer(),
        dim=dim,
        rank_k=rank_k,
        force_orthogonal=True,
        use_baseline=False
    )

    # Initialize with known orthogonal vectors
    V = torch.eye(dim)[:rank_k, :]  # First k standard basis vectors
    layer.subspace_vectors.data = V

    # Test projection
    h = torch.randn(2, dim)
    V_ortho = layer.get_orthonormal_basis()

    # Manual projection
    proj_manual = torch.einsum('kd,bd->bk', V_ortho, h)  # [2, k]
    proj_manual = torch.einsum('bk,kd->bd', proj_manual, V_ortho)  # [2, dim]

    # Method projection
    proj_method = layer._project_orthogonal(h, V_ortho)

    # Should be identical
    assert torch.allclose(proj_manual, proj_method, atol=1e-5)

    # Projection should zero out components in V
    h_projected = h - proj_method
    for v in V_ortho:
        overlap = (h_projected @ v).abs()
        assert torch.allclose(overlap, torch.zeros_like(overlap), atol=1e-5)


def test_pseudoinverse_projection():
    """Test pseudoinverse projection P = V(V^T V)^{-1}V^T."""
    dim = 128
    rank_k = 3

    layer = RankKUnifiedLayer(
        base_layer=DummyLayer(),
        dim=dim,
        rank_k=rank_k,
        force_orthogonal=False,
        use_baseline=False
    )

    # Create non-orthogonal vectors
    V = torch.randn(rank_k, dim)
    layer.subspace_vectors.data = V

    # Test projection
    h = torch.randn(2, dim)
    proj = layer._project_pseudoinverse(h, V)

    # Verify projection properties:
    # 1. P^2 = P (idempotent)
    proj_twice = layer._project_pseudoinverse(proj, V)
    assert torch.allclose(proj, proj_twice, atol=1e-4)

    # 2. Projection is in span(V)
    # Compute coefficients
    G = V @ V.T
    G_inv = torch.linalg.inv(G + 1e-6 * torch.eye(rank_k))
    coeffs = torch.einsum('bd,kd->bk', proj, V) @ G_inv

    # Reconstruct - should match
    reconstructed = torch.einsum('bk,kd->bd', coeffs, V)
    assert torch.allclose(proj, reconstructed, atol=1e-4)


def test_baseline_restoration():
    """Test ACE baseline restoration."""
    dim = 128
    rank_k = 3
    batch_size = 10

    layer = RankKUnifiedLayer(
        base_layer=DummyLayer(),
        dim=dim,
        rank_k=rank_k,
        force_orthogonal=True,
        use_baseline=True
    )

    # Fit baseline
    harmless_acts = torch.randn(batch_size, dim)
    layer.fit_baseline(harmless_acts)

    assert layer._baseline_fitted
    assert torch.allclose(layer.v_minus, harmless_acts.mean(dim=0))


def test_initialize_from_mean_diff():
    """Test initialization from mean difference + noise."""
    dim = 128
    rank_k = 3
    batch_size = 10

    layer = RankKUnifiedLayer(
        base_layer=DummyLayer(),
        dim=dim,
        rank_k=rank_k,
        use_baseline=True,
        force_orthogonal=False
    )

    # Create data
    harmful = torch.randn(batch_size, dim) + 1.0  # Shifted
    harmless = torch.randn(batch_size, dim)

    # Initialize
    layer.initialize_from_mean_diff(harmful, harmless, noise_std=0.01)

    # Check baseline fitted
    assert layer._baseline_fitted
    assert torch.allclose(layer.v_minus, harmless.mean(dim=0))

    # Check vectors are normalized and close to mean diff
    mean_diff = (harmful.mean(dim=0) - harmless.mean(dim=0))
    mean_diff = mean_diff / mean_diff.norm()

    for v in layer.subspace_vectors:
        # Should be normalized
        assert torch.allclose(v.norm(), torch.tensor(1.0), atol=1e-4)

        # Should be close to mean diff (within noise)
        similarity = (v @ mean_diff).abs()
        assert similarity > 0.95  # At least 95% overlap


def test_condition_number():
    """Test condition number computation."""
    dim = 128
    rank_k = 3

    # Orthogonal case
    layer_ortho = RankKUnifiedLayer(
        base_layer=DummyLayer(),
        dim=dim,
        rank_k=rank_k,
        force_orthogonal=True
    )

    V_ortho = torch.eye(dim)[:rank_k, :]
    layer_ortho.subspace_vectors.data = V_ortho

    cond_ortho = layer_ortho.get_condition_number()
    assert abs(cond_ortho - 1.0) < 0.1  # Should be ~1.0

    # Non-orthogonal case
    layer_non_ortho = RankKUnifiedLayer(
        base_layer=DummyLayer(),
        dim=dim,
        rank_k=rank_k,
        force_orthogonal=False
    )

    # Create near-collinear vectors (high condition number)
    V_bad = torch.randn(rank_k, dim)
    V_bad[1] = V_bad[0] + 0.01 * torch.randn(dim)  # Nearly collinear
    layer_non_ortho.subspace_vectors.data = V_bad

    cond_bad = layer_non_ortho.get_condition_number()
    assert cond_bad > 10  # Should be high


def test_forward_orthogonal_vs_nonorthogonal():
    """Test that both modes produce valid outputs."""
    dim = 128
    rank_k = 3

    # Create dummy input
    x = torch.randn(2, 10, dim)  # [batch, seq, dim]

    # Orthogonal mode
    layer_ortho = RankKUnifiedLayer(
        base_layer=DummyLayer(),
        dim=dim,
        rank_k=rank_k,
        force_orthogonal=True,
        use_baseline=False,
        projection_alpha=1.0,
        addition_alpha=0.0  # No steering for simplicity
    )

    output_ortho = layer_ortho(x)
    assert output_ortho.shape == x.shape

    # Non-orthogonal mode
    layer_non_ortho = RankKUnifiedLayer(
        base_layer=DummyLayer(),
        dim=dim,
        rank_k=rank_k,
        force_orthogonal=False,
        use_baseline=False,
        projection_alpha=1.0,
        addition_alpha=0.0
    )

    # Use same vectors (but non-orthogonal processing)
    layer_non_ortho.subspace_vectors.data = layer_ortho.subspace_vectors.data.clone()

    output_non_ortho = layer_non_ortho(x)
    assert output_non_ortho.shape == x.shape


def test_per_direction_steering():
    """Test per-direction steering (vector α)."""
    dim = 128
    rank_k = 3

    # Scalar α
    layer_scalar = RankKUnifiedLayer(
        base_layer=DummyLayer(),
        dim=dim,
        rank_k=rank_k,
        per_direction_steering=False,
        addition_alpha=1.0
    )

    assert layer_scalar.alpha.numel() == 1  # Scalar

    # Vector α
    layer_vector = RankKUnifiedLayer(
        base_layer=DummyLayer(),
        dim=dim,
        rank_k=rank_k,
        per_direction_steering=True,
        addition_alpha=1.0
    )

    assert layer_vector.alpha.numel() == rank_k  # Vector

    # Test forward
    x = torch.randn(2, dim)
    output_scalar = layer_scalar(x)
    output_vector = layer_vector(x)

    assert output_scalar.shape == x.shape
    assert output_vector.shape == x.shape


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
