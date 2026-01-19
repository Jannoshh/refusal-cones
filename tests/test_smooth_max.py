#!/usr/bin/env python3
"""
Test smooth max loss implementation.

Verifies that:
1. Smooth max approaches hard max as temperature → 0
2. Smooth max approaches mean as temperature → ∞
3. Smooth max is differentiable
4. Weighted version gives higher weights to higher losses
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
from src.training.trainers.per_layer_training import smooth_max_loss, weighted_smooth_max_loss


def test_smooth_max_properties():
    """Test mathematical properties of smooth max."""
    print("=" * 60)
    print("Testing Smooth Max Loss Properties")
    print("=" * 60)

    # Create test losses
    losses = torch.tensor([1.0, 2.0, 5.0, 3.0, 1.5])
    print(f"\nTest losses: {losses.tolist()}")
    print(f"Hard max: {losses.max().item():.4f}")
    print(f"Mean: {losses.mean().item():.4f}")

    # Test different temperatures
    temperatures = [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]

    print("\nSmooth max at different temperatures:")
    print("Temperature | Smooth Max | Error from Max | Error from Mean")
    print("-" * 60)

    for temp in temperatures:
        sm = smooth_max_loss(losses, temperature=temp)
        error_max = abs(sm.item() - losses.max().item())
        error_mean = abs(sm.item() - losses.mean().item())
        print(f"{temp:11.2f} | {sm.item():10.4f} | {error_max:14.4f} | {error_mean:15.4f}")

    # Verify: low temperature → max
    assert smooth_max_loss(losses, 0.01) - losses.max() < 0.1, "Low temp should approach max"

    # Verify: high temperature → mean
    assert abs(smooth_max_loss(losses, 100.0) - losses.mean()) < 0.5, "High temp should approach mean"

    print("\n✓ Temperature properties verified")


def test_smooth_max_differentiability():
    """Test that smooth max is differentiable."""
    print("\n" + "=" * 60)
    print("Testing Differentiability")
    print("=" * 60)

    losses = torch.tensor([1.0, 2.0, 5.0, 3.0], requires_grad=True)
    sm = smooth_max_loss(losses, temperature=1.0)

    print(f"\nLosses: {losses.detach().tolist()}")
    print(f"Smooth max: {sm.item():.4f}")

    # Compute gradient
    sm.backward()
    grads = losses.grad

    print(f"Gradients: {grads.tolist()}")
    print(f"Sum of gradients: {grads.sum().item():.4f} (should be ≈1.0)")

    # Gradients should be positive and sum to ≈1
    assert (grads > 0).all(), "All gradients should be positive"
    assert abs(grads.sum().item() - 1.0) < 0.01, "Gradients should sum to 1"

    # Highest loss should have highest gradient
    max_idx = losses.detach().argmax()
    assert grads[max_idx] == grads.max(), "Highest loss should have highest gradient"

    print("✓ Differentiability verified")


def test_weighted_smooth_max():
    """Test weighted smooth max."""
    print("\n" + "=" * 60)
    print("Testing Weighted Smooth Max")
    print("=" * 60)

    losses = torch.tensor([1.0, 2.0, 5.0, 3.0, 1.5])
    print(f"\nLosses: {losses.tolist()}")

    weighted_loss, weights = weighted_smooth_max_loss(losses, temperature=1.0)

    print(f"\nWeighted loss: {weighted_loss.item():.4f}")
    print(f"Weights: {weights.tolist()}")
    print(f"Sum of weights: {weights.sum().item():.4f}")

    # Highest loss should have highest weight
    max_idx = losses.argmax()
    assert weights[max_idx] == weights.max(), "Highest loss should have highest weight"

    # Weights should sum to 1
    assert abs(weights.sum().item() - 1.0) < 1e-5, "Weights should sum to 1"

    # Weighted loss should be between min and max
    assert losses.min() <= weighted_loss <= losses.max(), "Weighted loss should be in range"

    print("✓ Weighted smooth max verified")


def visualize_smooth_max():
    """Visualize smooth max behavior."""
    print("\n" + "=" * 60)
    print("Visualizing Smooth Max")
    print("=" * 60)

    losses = torch.tensor([1.0, 2.0, 5.0, 3.0, 1.5])
    temperatures = np.logspace(-2, 1, 50)  # 0.01 to 10

    smooth_maxes = []
    for temp in temperatures:
        sm = smooth_max_loss(losses, temperature=temp)
        smooth_maxes.append(sm.item())

    # Create plot
    plt.figure(figsize=(10, 6))
    plt.semilogx(temperatures, smooth_maxes, 'b-', linewidth=2, label='Smooth Max')
    plt.axhline(y=losses.max().item(), color='r', linestyle='--', label=f'Hard Max = {losses.max().item():.2f}')
    plt.axhline(y=losses.mean().item(), color='g', linestyle='--', label=f'Mean = {losses.mean().item():.2f}')

    plt.xlabel('Temperature')
    plt.ylabel('Loss')
    plt.title('Smooth Max Loss vs Temperature')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save
    import os
    os.makedirs('results/plots', exist_ok=True)
    plt.savefig('results/plots/smooth_max_visualization.png', dpi=150, bbox_inches='tight')
    print("\n✓ Visualization saved to: results/plots/smooth_max_visualization.png")


def compare_loss_functions():
    """Compare different loss aggregation methods."""
    print("\n" + "=" * 60)
    print("Comparing Loss Aggregation Methods")
    print("=" * 60)

    # Simulate per-layer losses
    losses = torch.tensor([1.5, 2.0, 6.0, 2.5, 1.8, 3.0, 2.2, 5.5])
    print(f"\nPer-layer losses: {losses.tolist()}")

    methods = {
        'Mean': losses.mean(),
        'Max (hard)': losses.max(),
        'Smooth Max (τ=0.1)': smooth_max_loss(losses, 0.1),
        'Smooth Max (τ=1.0)': smooth_max_loss(losses, 1.0),
        'Smooth Max (τ=2.0)': smooth_max_loss(losses, 2.0),
    }

    print("\nAggregation Method        | Loss")
    print("-" * 40)
    for name, value in methods.items():
        print(f"{name:25} | {value.item():.4f}")

    print("\n✓ Comparison complete")


if __name__ == '__main__':
    print("Testing Smooth Max Loss Implementation\n")

    test_smooth_max_properties()
    test_smooth_max_differentiability()
    test_weighted_smooth_max()
    visualize_smooth_max()
    compare_loss_functions()

    print("\n" + "=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)
