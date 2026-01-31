"""
Unit tests for REINFORCE optimizer weighting logic.
"""

import torch
import pytest


def calc_reinforce_weights(rewards: torch.Tensor, normalize_weights: bool = True) -> torch.Tensor:
    """
    Compute variance-reduced REINFORCE weights.

    This is a standalone version of the function from REINFORCEVectorOptimizer
    for testing purposes.

    Formula from Geisler et al. (2025):
        w_i = (n * R_i - sum(R)) / (n-1) / n

    This simplifies to: w_i = R_i - mean(R)
    """
    # Mean baseline
    mean_reward = rewards.mean()

    # Variance-reduced weights (baseline subtraction)
    weights = rewards - mean_reward

    # Optional: Normalize to sum to 1
    if normalize_weights and weights.abs().sum() > 0:
        weights = weights / weights.abs().sum()

    return weights


class TestReinforceWeights:
    """Tests for REINFORCE variance-reduced weight calculation."""

    def test_weights_sum_to_approximately_zero_before_normalization(self):
        """Weights before normalization should sum to ~0 (baseline subtraction property)."""
        rewards = torch.tensor([0.8, 0.2, 0.5, 0.5])

        # Compute weights without normalization
        mean_reward = rewards.mean()
        weights = rewards - mean_reward

        # Should sum to approximately zero
        assert abs(weights.sum().item()) < 1e-6

    def test_weights_normalized_correctly(self):
        """Normalized weights should have abs values sum to 1."""
        rewards = torch.tensor([0.8, 0.2, 0.5, 0.5])
        weights = calc_reinforce_weights(rewards, normalize_weights=True)

        # Abs sum should be 1.0
        assert abs(weights.abs().sum().item() - 1.0) < 1e-6

    def test_best_reward_gets_positive_weight(self):
        """The best performing sample should get a positive weight."""
        rewards = torch.tensor([0.1, 0.9, 0.5, 0.3])
        weights = calc_reinforce_weights(rewards, normalize_weights=True)

        # Index 1 has highest reward, should have positive weight
        assert weights[1].item() > 0

    def test_worst_reward_gets_negative_weight(self):
        """The worst performing sample should get a negative weight."""
        rewards = torch.tensor([0.1, 0.9, 0.5, 0.3])
        weights = calc_reinforce_weights(rewards, normalize_weights=True)

        # Index 0 has lowest reward, should have negative weight
        assert weights[0].item() < 0

    def test_equal_rewards_give_zero_weights(self):
        """When all rewards are equal, all weights should be zero."""
        rewards = torch.tensor([0.5, 0.5, 0.5, 0.5])

        # Don't normalize (would divide by zero)
        weights = calc_reinforce_weights(rewards, normalize_weights=False)

        # All weights should be zero
        assert torch.allclose(weights, torch.zeros(4))

    def test_weight_ordering_matches_reward_ordering(self):
        """Weights should preserve the ordering of rewards."""
        rewards = torch.tensor([0.2, 0.8, 0.4, 0.6])
        weights = calc_reinforce_weights(rewards, normalize_weights=False)

        # Sort order should match
        reward_order = torch.argsort(rewards)
        weight_order = torch.argsort(weights)

        assert torch.equal(reward_order, weight_order)

    def test_formula_equivalence(self):
        """Test that simplified formula equals the original Geisler formula."""
        rewards = torch.tensor([0.3, 0.7, 0.5, 0.4])
        n = len(rewards)

        # Original Geisler formula: w_i = (n * R_i - sum(R)) / (n-1) / n
        sum_R = rewards.sum()
        original_weights = (n * rewards - sum_R) / (n - 1) / n

        # Simplified formula: w_i = R_i - mean(R)
        # Note: These differ by a constant factor
        simplified_weights = rewards - rewards.mean()

        # They should be proportional (same direction, different magnitude)
        # original = simplified * (1 / (n-1))
        expected_factor = 1.0 / (n - 1)

        # Check proportionality
        ratio = original_weights / simplified_weights
        assert torch.allclose(ratio, torch.full((4,), expected_factor), rtol=1e-5)

    def test_single_sample_returns_zero(self):
        """With only one sample, weight should be zero."""
        rewards = torch.tensor([0.5])
        weights = calc_reinforce_weights(rewards, normalize_weights=False)

        assert weights[0].item() == 0.0

    def test_two_samples_symmetric(self):
        """With two samples, weights should be symmetric around zero."""
        rewards = torch.tensor([0.3, 0.7])
        weights = calc_reinforce_weights(rewards, normalize_weights=False)

        # weights[0] + weights[1] = 0
        assert abs(weights.sum().item()) < 1e-6

        # |weights[0]| == |weights[1]|
        assert abs(weights[0].abs().item() - weights[1].abs().item()) < 1e-6


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
