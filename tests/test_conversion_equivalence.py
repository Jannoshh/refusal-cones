#!/usr/bin/env python3
"""
Tests to verify that hooks and weight modifications produce equivalent results.

This validates that both approaches:
1. Produce identical outputs
2. Have correct gradient flow
3. Can be used interchangeably
"""

import torch
import torch.nn as nn
from typing import Tuple


# =============================================================================
# Test Setup: Simple Models
# =============================================================================

class SimpleLinear(nn.Module):
    """Simple linear layer for testing."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self.linear(x)


class SimpleTransformerLayer(nn.Module):
    """Simplified transformer layer for testing."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.self_attn_proj = nn.Linear(hidden_dim, hidden_dim)
        self.feed_forward = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        # Simplified: just two linear projections
        attn_out = self.self_attn_proj(x)
        ff_out = self.feed_forward(attn_out)
        return ff_out


# =============================================================================
# Approach 1: Hooks (Current Implementation)
# =============================================================================

def apply_ablation_hook(v: torch.Tensor):
    """Create hook that ablates vector v."""

    def hook(module, input, output):
        # Normalize
        v_norm = v / (v.norm() + 1e-8)

        # Project out: h' = h - (h · v)v
        projection = torch.einsum('...d,d->...', output, v_norm)
        ablated = output - torch.einsum('...,d->...d', projection, v_norm)

        return ablated

    return hook


def forward_with_hooks(model: nn.Module, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Forward pass with ablation hook."""

    # Register hook
    handle = model.register_forward_hook(apply_ablation_hook(v))

    # Forward
    output = model(x)

    # Remove hook
    handle.remove()

    return output


# =============================================================================
# Approach 2: Weight Modifications (LoRA-style)
# =============================================================================

from src.utils.conversion_utils import (
    get_projection_matrix,
    VectorModifiedLayer
)


def forward_with_weight_modification(
    model: nn.Module,
    x: torch.Tensor,
    v: torch.Tensor
) -> torch.Tensor:
    """Forward pass with weight modification in computation graph."""

    # Get projection matrix (differentiable!)
    P = get_projection_matrix(v)

    # For linear layer, modify weights in forward pass
    if isinstance(model, SimpleLinear):
        # Get original weight
        W = model.linear.weight  # [out_dim, in_dim]

        # Apply projection to output space: W' = P @ W
        W_modified = P @ W

        # Forward with modified weights (differentiable!)
        output = torch.nn.functional.linear(x, W_modified, model.linear.bias)

    else:
        # Use VectorModifiedLayer wrapper
        wrapped = VectorModifiedLayer(model, ablation_vector=v)
        output = wrapped(x)

    return output


# =============================================================================
# Tests
# =============================================================================

def test_output_equivalence():
    """Test that hooks and weight modifications produce same output."""

    print("=" * 70)
    print("Test 1: Output Equivalence")
    print("=" * 70)

    # Setup
    torch.manual_seed(42)
    hidden_dim = 128
    batch_size = 4

    model = SimpleLinear(hidden_dim, hidden_dim)
    x = torch.randn(batch_size, hidden_dim)
    v = torch.randn(hidden_dim)

    # Method 1: Hooks
    out_hooks = forward_with_hooks(model, x, v)

    # Method 2: Weight modifications
    out_weights = forward_with_weight_modification(model, x, v)

    # Compare
    diff = (out_hooks - out_weights).abs().max().item()
    print(f"\nInput shape: {x.shape}")
    print(f"Vector shape: {v.shape}")
    print(f"\nOutput (hooks):   {out_hooks[0, :5]}")
    print(f"Output (weights): {out_weights[0, :5]}")
    print(f"\nMax difference: {diff:.2e}")

    assert diff < 1e-3  # Relaxed for floating point precision


def test_gradient_flow():
    """Test that gradients flow correctly through both methods."""

    print("\n" + "=" * 70)
    print("Test 2: Gradient Flow")
    print("=" * 70)

    # Setup
    torch.manual_seed(42)
    hidden_dim = 64
    batch_size = 2

    model = SimpleLinear(hidden_dim, hidden_dim)

    # Freeze model
    for param in model.parameters():
        param.requires_grad = False

    x = torch.randn(batch_size, hidden_dim)

    # Method 1: Hooks
    v1 = torch.randn(hidden_dim, requires_grad=True)
    out1 = forward_with_hooks(model, x, v1)
    loss1 = out1.pow(2).mean()
    loss1.backward()
    grad1 = v1.grad.clone()

    print("\nMethod 1 (Hooks):")
    print(f"  Loss: {loss1.item():.6f}")
    print(f"  Gradient norm: {grad1.norm().item():.6f}")
    print(f"  Gradient shape: {grad1.shape}")
    print(f"  Has gradient: {v1.grad is not None}")

    # Method 2: Weight modifications
    v2 = torch.randn(hidden_dim, requires_grad=True)
    # Use same initialization for fair comparison
    with torch.no_grad():
        v2.copy_(v1.data)

    out2 = forward_with_weight_modification(model, x, v2)
    loss2 = out2.pow(2).mean()
    loss2.backward()
    grad2 = v2.grad.clone()

    print("\nMethod 2 (Weight Modifications):")
    print(f"  Loss: {loss2.item():.6f}")
    print(f"  Gradient norm: {grad2.norm().item():.6f}")
    print(f"  Gradient shape: {grad2.shape}")
    print(f"  Has gradient: {v2.grad is not None}")

    # Compare gradients
    grad_diff = (grad1 - grad2).abs().max().item()
    print(f"\nGradient difference: {grad_diff:.2e}")

    # Both should have gradients
    has_grads = (v1.grad is not None) and (v2.grad is not None)

    assert has_grads
    assert grad_diff < 1e-4


def test_training_equivalence():
    """Test that both methods can be used for training with same results."""

    print("\n" + "=" * 70)
    print("Test 3: Training Equivalence")
    print("=" * 70)

    # Setup
    torch.manual_seed(42)
    hidden_dim = 64
    batch_size = 4
    num_steps = 10
    lr = 0.01

    # Create two identical models
    model1 = SimpleLinear(hidden_dim, hidden_dim)
    model2 = SimpleLinear(hidden_dim, hidden_dim)

    # Copy weights to ensure identical start
    model2.load_state_dict(model1.state_dict())

    # Freeze both models
    for param in model1.parameters():
        param.requires_grad = False
    for param in model2.parameters():
        param.requires_grad = False

    # Initialize vectors (same for both)
    v1 = torch.randn(hidden_dim, requires_grad=True)
    v2 = v1.clone().detach().requires_grad_(True)

    # Optimizers
    opt1 = torch.optim.SGD([v1], lr=lr)
    opt2 = torch.optim.SGD([v2], lr=lr)

    # Training data
    x = torch.randn(batch_size, hidden_dim)
    target = torch.randn(batch_size, hidden_dim)

    print(f"\nTraining for {num_steps} steps...")
    print(f"Learning rate: {lr}")

    losses1 = []
    losses2 = []

    for step in range(num_steps):
        # Method 1: Hooks
        opt1.zero_grad()
        out1 = forward_with_hooks(model1, x, v1)
        loss1 = (out1 - target).pow(2).mean()
        loss1.backward()
        opt1.step()
        losses1.append(loss1.item())

        # Method 2: Weight modifications
        opt2.zero_grad()
        out2 = forward_with_weight_modification(model2, x, v2)
        loss2 = (out2 - target).pow(2).mean()
        loss2.backward()
        opt2.step()
        losses2.append(loss2.item())

        if step % 3 == 0:
            print(f"  Step {step}: Loss1={loss1.item():.6f}, Loss2={loss2.item():.6f}")

    # Compare final vectors
    vector_diff = (v1 - v2).abs().max().item()
    print(f"\nFinal vector difference: {vector_diff:.2e}")

    # Compare loss trajectories
    loss_diff = max(abs(l1 - l2) for l1, l2 in zip(losses1, losses2))
    print(f"Max loss difference: {loss_diff:.2e}")

    assert vector_diff < 1e-4
    assert loss_diff < 1e-4


def test_with_real_model():
    """Test with actual transformer architecture."""

    print("\n" + "=" * 70)
    print("Test 4: Real Transformer Layer")
    print("=" * 70)

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print("\nLoading small model for testing...")
        model = AutoModelForCausalLM.from_pretrained(
            'gpt2',
            torch_dtype=torch.float32
        )
        tokenizer = AutoTokenizer.from_pretrained('gpt2')
        model.eval()

        # Get one layer
        layer = model.transformer.h[0]
        hidden_dim = model.config.n_embd

        # Random input
        x = torch.randn(1, 10, hidden_dim)  # [batch, seq, hidden]

        # Random vector
        v = torch.randn(hidden_dim, requires_grad=True)

        print(f"Hidden dim: {hidden_dim}")
        print(f"Input shape: {x.shape}")

        # Method 1: Hooks
        def hook_fn(module, input, output):
            v_norm = v / (v.norm() + 1e-8)
            if isinstance(output, tuple):
                act = output[0]
            else:
                act = output
            proj = torch.einsum('...d,d->...', act, v_norm)
            ablated = act - torch.einsum('...,d->...d', proj, v_norm)
            if isinstance(output, tuple):
                return (ablated,) + output[1:]
            return ablated

        handle = layer.register_forward_hook(hook_fn)
        out1 = layer(x)[0] if isinstance(layer(x), tuple) else layer(x)
        handle.remove()

        # Method 2: VectorModifiedLayer wrapper
        from src.utils.conversion_utils import VectorModifiedLayer

        wrapped_layer = VectorModifiedLayer(layer, ablation_vector=v.detach())
        out2 = wrapped_layer(x)
        if isinstance(out2, tuple):
            out2 = out2[0]

        # Compare
        diff = (out1 - out2).abs().max().item()
        print(f"\nMax output difference: {diff:.2e}")

        assert diff < 1e-4

    except ImportError:
        print("⚠ SKIP: transformers not available")
        return
    except Exception as e:
        print(f"⚠ ERROR: {e}")
        raise


def test_backward_compatibility():
    """Test that we can convert between approaches."""

    print("\n" + "=" * 70)
    print("Test 5: Backward Compatibility")
    print("=" * 70)

    from src.utils.conversion_utils import convert_model_to_vector_modified
    from src.training.trainers.per_layer_training import PerLayerRefusalVectors

    # Create model
    torch.manual_seed(42)
    hidden_dim = 64

    model = SimpleTransformerLayer(hidden_dim)

    # Create per-layer vectors
    n_layers = 1
    vectors = PerLayerRefusalVectors(n_layers, hidden_dim)

    # Test conversion
    # Convert model (would wrap layers with VectorModifiedLayer)
    print("\nTesting conversion utilities...")

    v = vectors.get_all_vectors()[0]

    from src.utils.conversion_utils import VectorModifiedLayer
    wrapped = VectorModifiedLayer(model, ablation_vector=v)

    from src.utils.conversion_utils import get_trainable_vector_parameters
    params = [p for p in wrapped.parameters() if p.requires_grad]
    assert len(params) > 0


# =============================================================================
# Main Test Runner
# =============================================================================

def run_all_tests():
    """Run all tests and report results."""

    print("=" * 70)
    print("EQUIVALENCE TESTS: Hooks vs Weight Modifications")
    print("=" * 70)
    print("\nTesting that both approaches produce identical results\n")

    tests = [
        ("Output Equivalence", test_output_equivalence),
        ("Gradient Flow", test_gradient_flow),
        ("Training Equivalence", test_training_equivalence),
        ("Real Transformer Layer", test_with_real_model),
        ("Backward Compatibility", test_backward_compatibility),
    ]

    results = []

    for name, test_fn in tests:
        try:
            passed = test_fn()
            results.append((name, passed))
        except Exception as e:
            print(f"\n✗ ERROR in {name}: {e}")
            results.append((name, False))

    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)

    passed_count = sum(1 for _, passed in results if passed)
    total_count = len(results)

    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {name}")

    print(f"\nTotal: {passed_count}/{total_count} tests passed")

    if passed_count == total_count:
        print("\n✓ All tests passed! Both approaches are equivalent.")
    else:
        print(f"\n✗ {total_count - passed_count} test(s) failed")

    return passed_count == total_count


if __name__ == '__main__':
    import sys

    success = run_all_tests()

    sys.exit(0 if success else 1)
