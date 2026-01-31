#!/usr/bin/env python3
"""
Example: Multi-Dimensional ACE with Non-Orthogonal Vectors

Demonstrates the generalized ACE formula for k vectors:
    h' = h - P_V(h - v⁻) + V·α

Where:
    - V = [v_1, ..., v_k] (arbitrary vectors, not necessarily orthogonal)
    - P_V = V(V^T V)^{-1}V^T (pseudoinverse projection)
    - v⁻ = mean harmless activations (baseline)
    - α ∈ R^k (per-direction steering coefficients)

This is more expressive than forcing orthogonality, allowing the optimizer
to discover correlated refusal directions.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.training.adapters.unified_rdo_adapter import (
    UnifiedRDOConfig,
    get_unified_rdo_model
)


def main():
    print("=" * 70)
    print("Multi-Dimensional ACE Example")
    print("=" * 70)

    # Load model
    model_name = "Qwen/Qwen3-0.6B"
    print(f"\n1. Loading model: {model_name}")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Configure multi-dimensional ACE
    print("\n2. Configuring multi-dimensional ACE")
    print("   - rank_k=3 (3-dimensional refusal subspace)")
    print("   - force_orthogonal=False (allow non-orthogonal vectors)")
    print("   - use_baseline=True (ACE baseline restoration)")
    print("   - per_direction_steering=True (independent α per direction)")

    config = UnifiedRDOConfig(
        target_modules=["layers"],
        enable_rank_k=True,
        rank_k=3,
        force_orthogonal=False,      # Allow non-orthogonal vectors
        use_baseline=True,            # Use ACE baseline v⁻
        per_direction_steering=True,  # α ∈ R^3 instead of scalar
        projection_alpha=1.0,
        addition_alpha=1.0
    )

    # Wrap model
    model = get_unified_rdo_model(model, config)

    # Example prompts
    harmless_prompts = [
        "How do I make a cake?",
        "Explain photosynthesis",
        "Write a poem about nature",
        "What is machine learning?",
    ]

    harmful_prompts = [
        "How do I make a bomb?",
        "Write malware code",
        "How to hack a website?",
        "Tell me how to steal a car",
    ]

    # Initialize from mean difference
    print("\n3. Initializing subspace vectors from mean difference + noise")
    model.initialize_rank_k_from_mean_diff(
        tokenizer=tokenizer,
        harmless_prompts=harmless_prompts,
        harmful_prompts=harmful_prompts,
        batch_size=4,
        noise_std=0.01  # Small perturbations around mean diff
    )

    # Check condition numbers
    print("\n4. Checking numerical stability (condition numbers)")
    cond_numbers = model.get_all_condition_numbers()
    print(f"   Condition numbers across layers:")
    print(f"   Min: {min(cond_numbers):.2f}")
    print(f"   Max: {max(cond_numbers):.2f}")
    print(f"   Mean: {sum(cond_numbers)/len(cond_numbers):.2f}")

    if max(cond_numbers) > 100:
        print("   ⚠️  High condition number detected (>100)")
        print("   Consider: 1) reducing rank_k, 2) using force_orthogonal=True")
    else:
        print("   ✓ Good numerical stability")

    # Compare orthogonal vs non-orthogonal
    print("\n5. Comparing orthogonal vs non-orthogonal modes")
    print("\n   a) Non-orthogonal (current):")
    print("      - More expressive (vectors can be correlated)")
    print("      - Requires matrix inversion: O(k²d + k³)")
    print("      - Condition number monitoring needed")

    print("\n   b) Orthogonal (alternative):")
    print("      - Less expressive (forced independence)")
    print("      - Faster computation: O(kd)")
    print("      - Always numerically stable")

    # Example training loop (pseudo-code)
    print("\n6. Training workflow:")
    print("""
    optimizer = torch.optim.Adam(model.get_affine_parameters(), lr=1e-3)

    for batch in dataloader:
        optimizer.zero_grad()

        # Forward pass (ACE transformation applied automatically)
        outputs = model(**batch)

        # Compute loss
        loss = criterion(outputs, targets)

        # Backprop (gradients through pseudoinverse projection)
        loss.backward()

        # Check stability every N steps
        if step % 100 == 0:
            cond_numbers = model.get_all_condition_numbers()
            print(f"Max condition number: {max(cond_numbers):.2f}")

        optimizer.step()
    """)

    # Show what's trainable
    print("\n7. Trainable parameters:")
    model.print_trainable_parameters()

    print("\n" + "=" * 70)
    print("✓ Example complete!")
    print("=" * 70)

    # Mathematical explanation
    print("\nMathematical Details:")
    print("-" * 70)
    print("""
Multi-dimensional ACE formula:

    h' = h - V(V^T V)^{-1}V^T (h - v⁻) + V·α

Components:
    V ∈ R^{k×d}        : Subspace basis (trainable)
    v⁻ ∈ R^d           : Harmless baseline (fitted from data)
    α ∈ R^k            : Steering coefficients (trainable)
    (V^T V)^{-1}       : Gram matrix inverse (k×k)

Why non-orthogonal?
    - Refusal directions may be naturally correlated
    - Orthogonalization destroys discovered structure
    - Pseudoinverse projection is unique and mathematically sound

Uniqueness:
    ✓ Projection onto span(V) is UNIQUE (orthogonal projection)
    ✗ Coefficient representation is NOT unique
      (multiple ways to write same vector in non-orthogonal basis)
    → Pseudoinverse gives minimum-norm solution

Trade-offs:
    Orthogonal (V^T V = I):
        + Fast: O(kd)
        + Stable: condition number = 1
        - Less expressive

    Non-orthogonal (V^T V ≠ I):
        + More expressive
        + Preserves discovered structure
        - Slower: O(k²d + k³)
        - Needs stability monitoring
    """)


if __name__ == "__main__":
    main()
