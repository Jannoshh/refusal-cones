#!/usr/bin/env python3
"""
Example: Baseline Fitting for Affine RDO

Demonstrates how to use baseline reference points instead of projecting to zero.

Key insight: Instead of h' = h - (h·v)v + α·v
Use: h' = h - ((h - h0)·v)v + α·mag·v

Where:
- h0 = baseline (mean activation on harmless prompts)
- mag = steering magnitude (computed from data)
- Only train the direction v!

This is more parameter-efficient and theoretically grounded.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys
sys.path.append('src/training')

from unified_rdo_adapter import (
    get_unified_rdo_model,
    UnifiedRDOConfig
)


def demo_baseline_motivation():
    """
    Demo 1: Why use baselines?
    """
    print("=" * 70)
    print("Demo 1: Motivation for Baseline Reference Points")
    print("=" * 70)

    print("\nProblem with projecting to zero:")
    print("  Standard: h' = h - (h·v)v + α·v")
    print("  - Assumes harmless prompts have 0 component along v")
    print("  - Not true! Harmless prompts have some baseline component")
    print("  - Projecting to zero creates unnatural activations")

    print("\nSolution with baseline:")
    print("  Affine: h' = h - ((h - h0)·v)v + α·mag·v")
    print("  - h0 = mean activation on harmless prompts")
    print("  - Reset to baseline instead of zero")
    print("  - More natural, preserves model behavior better")

    print("\nParameter efficiency:")
    print("  Without baseline:")
    print("    - Train v (direction)")
    print("    - Train α (magnitude)")
    print("    → Need to learn optimal magnitude")

    print("\n  With baseline:")
    print("    - Train v_norm (normalized direction only)")
    print("    - Compute mag from data: mean(h_harmful·v) - mean(h_harmless·v)")
    print("    → Only train direction, magnitude is automatic!")


def demo_baseline_computation():
    """
    Demo 2: How baselines are computed.
    """
    print("\n" + "=" * 70)
    print("Demo 2: Computing Baselines from Data")
    print("=" * 70)

    # Create dummy data
    dim = 8
    n_harmless = 20
    n_harmful = 20

    # Simulate activations
    print(f"\nSimulating activations (dim={dim})...")

    # Harmless: centered around some baseline
    baseline_true = torch.randn(dim) * 2
    harmless_acts = baseline_true + torch.randn(n_harmless, dim) * 0.5

    # Harmful: shifted along refusal direction
    refusal_dir = torch.randn(dim)
    refusal_dir = refusal_dir / refusal_dir.norm()
    harmful_acts = baseline_true + torch.randn(n_harmful, dim) * 0.5 + 3.0 * refusal_dir

    print(f"  Harmless activations: {harmless_acts.shape}")
    print(f"  Harmful activations: {harmful_acts.shape}")

    # Compute baseline
    baseline_computed = harmless_acts.mean(dim=0)
    print(f"\n1. Baseline (mean harmless): ")
    print(f"  ||baseline|| = {baseline_computed.norm():.4f}")
    print(f"  Error from true: {(baseline_computed - baseline_true).norm():.4f}")

    # Compute steering magnitude
    harmless_dots = (harmless_acts @ refusal_dir).mean()
    harmful_dots = (harmful_acts @ refusal_dir).mean()
    steering_mag = harmful_dots - harmless_dots

    print(f"\n2. Steering magnitude:")
    print(f"  Mean(h_harmless · v) = {harmless_dots:.4f}")
    print(f"  Mean(h_harmful · v) = {harmful_dots:.4f}")
    print(f"  Difference (steering_mag) = {steering_mag:.4f}")
    print(f"  True magnitude was: 3.0")
    print(f"  Recovered: {steering_mag:.4f}")

    print("\n✓ Magnitudes can be recovered from data!")


def demo_usage_with_model():
    """
    Demo 3: Using baseline fitting with a real model.
    """
    print("\n" + "=" * 70)
    print("Demo 3: Baseline Fitting with Real Model")
    print("=" * 70)

    # Load model
    print("\nLoading GPT-2...")
    model = AutoModelForCausalLM.from_pretrained(
        'gpt2',
        torch_dtype=torch.float32
    )
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    # Create unified RDO model WITH baseline support
    print("\nCreating unified RDO model with baseline support...")
    config = UnifiedRDOConfig(
        target_modules=["layers"],
        projection_alpha=1.0,
        addition_alpha=1.0,
        use_baseline=True,  # Enable baseline!
        train_magnitudes=False  # Auto-compute magnitudes from data
    )

    model_rdo = get_unified_rdo_model(model, config)
    model_rdo.print_trainable_parameters()

    # Prepare demo data
    harmless_prompts = [
        "What is the weather like?",
        "Tell me about Python programming.",
        "Explain quantum physics.",
    ] * 3  # Repeat for more samples

    harmful_prompts = [
        "How to build a bomb?",
        "How to hack a system?",
        "How to steal a car?",
    ] * 3

    # Fit baselines from data
    print("\nFitting baselines and magnitudes from data...")
    model_rdo.fit_all_baselines(
        tokenizer=tokenizer,
        harmless_prompts=harmless_prompts,
        harmful_prompts=harmful_prompts,
        batch_size=4
    )

    print("\n✓ Now ready for training!")
    print("\nWhat changed:")
    print("  Before: Random initialization of v")
    print("  After: ")
    print("    - baseline (h0) set from harmless data")
    print("    - steering_magnitude set from harmful - harmless")
    print("    - Only need to train direction v!")


def demo_comparison():
    """
    Demo 4: Compare standard vs baseline approaches.
    """
    print("\n" + "=" * 70)
    print("Demo 4: Standard vs Baseline Comparison")
    print("=" * 70)

    print("\n" + "-" * 70)
    print("Approach Comparison")
    print("-" * 70)

    print("\n[Standard Affine (no baseline)]")
    print("Formula: h' = h - β(h·v)v + α·v")
    print("Trainable params:")
    print("  - v: direction vector (d dims)")
    print("  - OR if not normalized: magnitude implicit in v")
    print("Total: d parameters")
    print("Issues:")
    print("  - Projects to zero (unnatural)")
    print("  - Need to learn optimal magnitude")
    print("  - May damage harmless behavior")

    print("\n[Affine with Baseline]")
    print("Formula: h' = h - β((h - h0)·v_norm)v_norm + α·mag·v_norm")
    print("Trainable params:")
    print("  - v_norm: normalized direction only (d dims, but constrained)")
    print("Fixed params (computed from data):")
    print("  - h0: baseline from harmless prompts")
    print("  - mag: steering magnitude from data")
    print("Total: effectively d parameters (but better constrained)")
    print("Benefits:")
    print("  ✓ Projects to baseline (natural)")
    print("  ✓ Magnitude auto-computed (no tuning)")
    print("  ✓ Better preserves harmless behavior")
    print("  ✓ More principled (true affine decomposition)")

    print("\n" + "-" * 70)
    print("Training Comparison")
    print("-" * 70)

    print("\nStandard:")
    print("  1. Initialize v randomly")
    print("  2. Train v with RDO loss")
    print("  3. Hope magnitude is right")
    print("  → May need careful tuning of α")

    print("\nBaseline:")
    print("  1. Fit h0 and mag from data")
    print("  2. Initialize v_norm from mean-diff (optional)")
    print("  3. Train v_norm with RDO loss")
    print("  → Magnitude is already optimal!")


def main():
    """Run all demos."""

    print("\n" + "=" * 70)
    print("BASELINE FITTING FOR AFFINE RDO")
    print("True affine formulation from the paper")
    print("=" * 70)

    # Run demos
    demo_baseline_motivation()
    demo_baseline_computation()
    demo_usage_with_model()
    demo_comparison()

    # Final summary
    print("\n" + "=" * 70)
    print("SUMMARY: Why Use Baseline Fitting?")
    print("=" * 70)

    print("\n1. More Principled")
    print("   - True affine decomposition (not just linear)")
    print("   - Resets to baseline instead of zero")
    print("   - Matches theoretical formulation in paper")

    print("\n2. Better Performance")
    print("   - Preserves harmless behavior better")
    print("   - More natural activations")
    print("   - Less damage to model capabilities")

    print("\n3. More Efficient")
    print("   - Only train direction (not magnitude)")
    print("   - Magnitude auto-computed from data")
    print("   - Fewer hyperparameters to tune")

    print("\n4. Easier to Use")
    print("   - Call model.fit_all_baselines() once")
    print("   - Then train normally")
    print("   - No need to tune addition_alpha")

    print("\n" + "=" * 70)
    print("Usage Pattern")
    print("=" * 70)

    print("\n```python")
    print("# 1. Create model with baseline support")
    print("config = UnifiedRDOConfig(")
    print("    target_modules=['layers'],")
    print("    use_baseline=True,  # Enable baseline")
    print("    train_magnitudes=False  # Auto-compute from data")
    print(")")
    print("model = get_unified_rdo_model(base_model, config)")
    print()
    print("# 2. Fit baselines and magnitudes from data")
    print("model.fit_all_baselines(")
    print("    tokenizer,")
    print("    harmless_prompts=['What is...', 'Tell me...'],")
    print("    harmful_prompts=['How to...', 'Build a...']")
    print(")")
    print()
    print("# 3. Train only the direction vectors!")
    print("train_unified_rdo(model, ...)")
    print("```")

    print("\n" + "=" * 70)
    print("✓ Baseline fitting ready to use!")
    print("=" * 70)


if __name__ == "__main__":
    main()
