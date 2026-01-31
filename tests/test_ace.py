"""
Tests for ACE (Affine Concept Editing) implementation.

Verifies the implementation matches the paper's formula (Equation 5):
    h' = h - proj_r(h) + proj_r(r⁻) + α·r
"""

import torch
import pytest
from src.steering.ace import (
    ace_transform,
    ACEVectors,
    ACEConfig,
)


class TestACETransform:
    """Test the core ACE transformation function."""

    def test_alpha_zero_projects_to_baseline(self):
        """
        With α=0, the r-parallel component of h' should equal proj_r(r⁻).

        This is the "standardization" property from the paper:
        α=0 means "act like harmless" regardless of input.
        """
        torch.manual_seed(42)
        dim = 64

        # Create test vectors
        r_minus = torch.randn(dim)  # Baseline (mean harmless)
        r_plus = torch.randn(dim)   # Mean harmful
        r = r_plus - r_minus         # Steering direction
        r_hat = r / r.norm()

        # Test with various input activations
        for _ in range(5):
            h = torch.randn(dim)

            # Apply ACE with α=0
            h_prime = ace_transform(h, r, r_minus, alpha=0.0)

            # The r-parallel component of h' should equal proj_r(r⁻)
            proj_h_prime = (h_prime @ r_hat) * r_hat
            proj_r_minus = (r_minus @ r_hat) * r_hat

            assert torch.allclose(proj_h_prime, proj_r_minus, atol=1e-5), \
                "α=0 should set r-component to baseline's r-component"

    def test_alpha_one_projects_to_harmful(self):
        """
        With α=1, the r-parallel component of h' should equal proj_r(r⁺).

        α=1 means "act like harmful" regardless of input.
        """
        torch.manual_seed(42)
        dim = 64

        r_minus = torch.randn(dim)
        r_plus = torch.randn(dim)
        r = r_plus - r_minus
        r_hat = r / r.norm()

        for _ in range(5):
            h = torch.randn(dim)

            # Apply ACE with α=1
            h_prime = ace_transform(h, r, r_minus, alpha=1.0)

            # r-parallel component should equal proj_r(r⁺)
            proj_h_prime = (h_prime @ r_hat) * r_hat
            proj_r_plus = (r_plus @ r_hat) * r_hat

            assert torch.allclose(proj_h_prime, proj_r_plus, atol=1e-5), \
                "α=1 should set r-component to harmful's r-component"

    def test_orthogonal_component_preserved(self):
        """
        The component orthogonal to r should be unchanged.

        h'_perp = h_perp (ACE only affects the r-direction)
        """
        torch.manual_seed(42)
        dim = 64

        r_minus = torch.randn(dim)
        r_plus = torch.randn(dim)
        r = r_plus - r_minus
        r_hat = r / r.norm()

        h = torch.randn(dim)

        # Compute orthogonal components
        h_parallel = (h @ r_hat) * r_hat
        h_perp = h - h_parallel

        # Apply ACE
        h_prime = ace_transform(h, r, r_minus, alpha=0.5)

        # Orthogonal component of h'
        h_prime_parallel = (h_prime @ r_hat) * r_hat
        h_prime_perp = h_prime - h_prime_parallel

        assert torch.allclose(h_perp, h_prime_perp, atol=1e-5), \
            "Orthogonal component should be unchanged"

    def test_formula_matches_paper(self):
        """
        Verify our implementation matches Equation 5 exactly:
        h' = h - proj_r(h) + proj_r(r⁻) + α·r
        """
        torch.manual_seed(42)
        dim = 64

        r_minus = torch.randn(dim)
        r_plus = torch.randn(dim)
        r = r_plus - r_minus
        r_hat = r / r.norm()
        h = torch.randn(dim)
        alpha = 0.7

        # Our implementation
        h_prime = ace_transform(h, r, r_minus, alpha)

        # Manual computation of paper's formula
        proj_h = (h @ r_hat) * r_hat
        proj_r_minus = (r_minus @ r_hat) * r_hat
        h_prime_manual = h - proj_h + proj_r_minus + alpha * r

        assert torch.allclose(h_prime, h_prime_manual, atol=1e-5), \
            "Implementation should match paper's Equation 5"

    def test_batched_input(self):
        """Test that ACE works with batched inputs."""
        torch.manual_seed(42)
        dim = 64
        batch_size = 8
        seq_len = 16

        r_minus = torch.randn(dim)
        r_plus = torch.randn(dim)
        r = r_plus - r_minus

        # Batched input [batch, seq, dim]
        h = torch.randn(batch_size, seq_len, dim)

        # Apply ACE
        h_prime = ace_transform(h, r, r_minus, alpha=0.0)

        # Check shape preserved
        assert h_prime.shape == h.shape, "Output shape should match input"

        # Check each position individually matches
        for b in range(batch_size):
            for s in range(seq_len):
                h_single = ace_transform(h[b, s], r, r_minus, alpha=0.0)
                assert torch.allclose(h_prime[b, s], h_single, atol=1e-5)

    def test_alpha_interpolation(self):
        """
        Test that α linearly interpolates the r-component.

        At α=0.5, the r-component should be halfway between r⁻ and r⁺.
        """
        torch.manual_seed(42)
        dim = 64

        r_minus = torch.randn(dim)
        r_plus = torch.randn(dim)
        r = r_plus - r_minus
        r_hat = r / r.norm()
        h = torch.randn(dim)

        # Get r-components at different α values
        h_0 = ace_transform(h, r, r_minus, alpha=0.0)
        h_half = ace_transform(h, r, r_minus, alpha=0.5)
        h_1 = ace_transform(h, r, r_minus, alpha=1.0)

        proj_0 = h_0 @ r_hat
        proj_half = h_half @ r_hat
        proj_1 = h_1 @ r_hat

        # Check linear interpolation
        expected_half = (proj_0 + proj_1) / 2
        assert torch.allclose(proj_half, expected_half, atol=1e-5), \
            "α=0.5 should give halfway r-component"


class TestACEVectors:
    """Test the ACEVectors dataclass."""

    def test_from_activations(self):
        """Test computing ACE vectors from activation data."""
        torch.manual_seed(42)
        dim = 64
        n_samples = 20

        # Simulate activations with different means
        harmless_acts = torch.randn(n_samples, dim) + torch.randn(dim) * 0.5
        harmful_acts = torch.randn(n_samples, dim) + torch.randn(dim) * 0.5 + 2.0

        vectors = ACEVectors.from_activations(harmless_acts, harmful_acts, layer=15)

        # Check r = r⁺ - r⁻
        expected_r = harmful_acts.mean(0) - harmless_acts.mean(0)
        assert torch.allclose(vectors.r, expected_r, atol=1e-5)

        # Check r_minus = mean(harmless)
        assert torch.allclose(vectors.r_minus, harmless_acts.mean(0), atol=1e-5)

        # Check r_plus = mean(harmful)
        assert torch.allclose(vectors.r_plus, harmful_acts.mean(0), atol=1e-5)

    def test_r_hat_normalized(self):
        """Test that r_hat is properly normalized."""
        torch.manual_seed(42)
        dim = 64

        r = torch.randn(dim) * 5  # Arbitrary magnitude
        r_minus = torch.randn(dim)
        r_plus = r_minus + r

        vectors = ACEVectors(r=r, r_minus=r_minus, r_plus=r_plus, layer=15)

        assert abs(vectors.r_hat.norm().item() - 1.0) < 1e-5, \
            "r_hat should be unit norm"

    def test_save_load_roundtrip(self, tmp_path):
        """Test saving and loading ACE vectors."""
        torch.manual_seed(42)
        dim = 64

        original = ACEVectors(
            r=torch.randn(dim),
            r_minus=torch.randn(dim),
            r_plus=torch.randn(dim),
            layer=15
        )

        path = str(tmp_path / "ace_vectors.pt")
        original.save(path)
        loaded = ACEVectors.load(path)

        assert torch.allclose(original.r, loaded.r)
        assert torch.allclose(original.r_minus, loaded.r_minus)
        assert torch.allclose(original.r_plus, loaded.r_plus)
        assert original.layer == loaded.layer


class TestACEConfig:
    """Test ACE configuration."""

    def test_default_values(self):
        """Test default configuration values."""
        config = ACEConfig()

        assert config.layer == 15
        assert config.alpha == 0.0
        assert config.position_start is None
        assert config.position_end is None

    def test_custom_values(self):
        """Test custom configuration."""
        config = ACEConfig(
            layer=20,
            alpha=0.5,
            position_start=10,
            position_end=50
        )

        assert config.layer == 20
        assert config.alpha == 0.5
        assert config.position_start == 10
        assert config.position_end == 50


class TestStandardizationProperty:
    """
    Test the key "standardization" property from the paper.

    A standardized steering method should give the same r-component
    output regardless of input, when α is fixed.
    """

    def test_standardization_alpha_zero(self):
        """
        Different inputs should produce same r-component when α=0.

        This is the key insight: ACE "standardizes" the behavior
        so the steering parameter fully controls output.
        """
        torch.manual_seed(42)
        dim = 64

        r_minus = torch.randn(dim)
        r_plus = torch.randn(dim)
        r = r_plus - r_minus
        r_hat = r / r.norm()

        # Different inputs (simulating harmful vs harmless prompts)
        h_harmful = r_plus + torch.randn(dim) * 0.1  # Near harmful mean
        h_harmless = r_minus + torch.randn(dim) * 0.1  # Near harmless mean
        h_random = torch.randn(dim) * 3  # Random activation

        # Apply ACE with α=0
        out_harmful = ace_transform(h_harmful, r, r_minus, alpha=0.0)
        out_harmless = ace_transform(h_harmless, r, r_minus, alpha=0.0)
        out_random = ace_transform(h_random, r, r_minus, alpha=0.0)

        # All should have the same r-component (= proj_r(r⁻))
        proj_harmful = out_harmful @ r_hat
        proj_harmless = out_harmless @ r_hat
        proj_random = out_random @ r_hat

        assert torch.allclose(proj_harmful, proj_harmless, atol=1e-5)
        assert torch.allclose(proj_harmless, proj_random, atol=1e-5)

    def test_standardization_alpha_one(self):
        """Different inputs should produce same r-component when α=1."""
        torch.manual_seed(42)
        dim = 64

        r_minus = torch.randn(dim)
        r_plus = torch.randn(dim)
        r = r_plus - r_minus
        r_hat = r / r.norm()

        h1 = torch.randn(dim)
        h2 = torch.randn(dim) * 5
        h3 = torch.randn(dim) - 10

        out1 = ace_transform(h1, r, r_minus, alpha=1.0)
        out2 = ace_transform(h2, r, r_minus, alpha=1.0)
        out3 = ace_transform(h3, r, r_minus, alpha=1.0)

        proj1 = out1 @ r_hat
        proj2 = out2 @ r_hat
        proj3 = out3 @ r_hat

        assert torch.allclose(proj1, proj2, atol=1e-5)
        assert torch.allclose(proj2, proj3, atol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
