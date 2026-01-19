#!/usr/bin/env python3
"""
Example: Unified Affine RDO

Demonstrates the unified affine transformation approach from
"Refusal in LLMs is an Affine Function" (https://arxiv.org/abs/2411.09003v3)

Key concepts:
1. Unifying projection and addition into single affine transformation
2. Simpler training (one forward pass vs two)
3. More expressive (ablate AND steer simultaneously)
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys
sys.path.append('src/training')

from unified_rdo_adapter import (
    get_unified_rdo_model,
    UnifiedRDOConfig,
    UnifiedRDOLayer
)
from unified_rdo_trainer import train_unified_rdo


def demo_affine_transformation():
    """
    Demo 1: Show how affine transformation works.
    """
    print("=" * 70)
    print("Demo 1: Understanding Affine Transformation")
    print("=" * 70)

    print("\nMathematical formula:")
    print("  h' = (I - β·vv^T)h + α·v")
    print("  h' = h - β·(h·v)v + α·v")
    print("  h' = h + [α·v - β·(h·v)v]")

    print("\nComponents:")
    print("  β = projection_alpha (controls ablation strength)")
    print("  α = addition_alpha (controls addition strength)")
    print("  v = learned steering vector")

    print("\nSpecial cases:")
    print("  β=1, α=0 → Pure projection (removes v)")
    print("  β=0, α=1 → Pure addition (adds v)")
    print("  β=1, α=1 → Full affine (remove AND add)")

    # Create dummy activation and vector
    hidden_dim = 8
    h = torch.randn(hidden_dim)
    v = torch.randn(hidden_dim)
    v = v / v.norm()  # Normalize

    print(f"\nExample with hidden_dim={hidden_dim}:")
    print(f"  Input h: {h[:3].tolist()[:3]}... (first 3 dims)")
    print(f"  Vector v: {v[:3].tolist()[:3]}... (first 3 dims)")

    # Case 1: Pure projection
    beta, alpha = 1.0, 0.0
    h_proj = h - beta * torch.dot(h, v) * v
    print(f"\n  β={beta}, α={alpha} (pure projection):")
    print(f"    h' = {h_proj[:3].tolist()[:3]}...")
    print(f"    Component removed: {torch.dot(h, v).item():.4f}")

    # Case 2: Pure addition
    beta, alpha = 0.0, 1.0
    h_add = h + alpha * v
    print(f"\n  β={beta}, α={alpha} (pure addition):")
    print(f"    h' = {h_add[:3].tolist()[:3]}...")
    print(f"    Component added: {alpha:.4f}")

    # Case 3: Full affine
    beta, alpha = 1.0, 1.0
    h_affine = h - beta * torch.dot(h, v) * v + alpha * v
    print(f"\n  β={beta}, α={alpha} (full affine):")
    print(f"    h' = {h_affine[:3].tolist()[:3]}...")
    print(f"    Removed: {torch.dot(h, v).item():.4f}, Added: {alpha:.4f}")

    print("\n✓ Affine combines both operations!")


def demo_single_vs_dual_forward():
    """
    Demo 2: Compare single forward (unified) vs dual forward (standard RDO).
    """
    print("\n" + "=" * 70)
    print("Demo 2: Single Forward vs Dual Forward")
    print("=" * 70)

    print("\nStandard RDO (dual forward):")
    print("  def compute_loss(model, harmful_input):")
    print("    # Forward 1: Ablation mode")
    print("    model.set_operation('ablate')")
    print("    outputs_ablate = model(harmful_input)")
    print("    loss_ablate = outputs_ablate.loss")
    print()
    print("    # Forward 2: Addition mode")
    print("    model.set_operation('add')")
    print("    outputs_add = model(harmful_input)")
    print("    loss_add = outputs_add.loss")
    print()
    print("    # Combine")
    print("    return λ_ablate * loss_ablate + λ_add * loss_add")
    print()
    print("  → TWO forward passes per example")
    print("  → More memory, slower training")

    print("\nUnified Affine RDO (single forward):")
    print("  def compute_loss(model, harmful_input):")
    print("    # Single forward: Affine transformation always on")
    print("    outputs = model(harmful_input)  # Applies h' = (I-vv^T)h + αv")
    print("    return λ_harmful * outputs.loss")
    print()
    print("  → ONE forward pass per example")
    print("  → Less memory, faster training")

    print("\n" + "-" * 70)
    print("Performance Comparison (estimated):")
    print("-" * 70)
    print(f"{'Metric':<30} {'Standard RDO':<20} {'Unified Affine':<20}")
    print("-" * 70)
    print(f"{'Forward passes/example':<30} {'2':<20} {'1':<20}")
    print(f"{'Training time':<30} {'~2x baseline':<20} {'~1x baseline':<20}")
    print(f"{'Memory usage':<30} {'~2x':<20} {'~1x':<20}")
    print(f"{'Gradient stability':<30} {'Moderate':<20} {'Better':<20}")
    print("-" * 70)


def demo_unified_model():
    """
    Demo 3: Create and test unified affine model.
    """
    print("\n" + "=" * 70)
    print("Demo 3: Creating Unified Affine Model")
    print("=" * 70)

    # Load small model
    print("\nLoading GPT-2...")
    model = AutoModelForCausalLM.from_pretrained(
        'gpt2',
        torch_dtype=torch.float32
    )
    tokenizer = AutoTokenizer.from_pretrained('gpt2')

    # Add unified RDO adapters
    print("\nAdding unified affine RDO adapters...")
    config = UnifiedRDOConfig(
        target_modules=["layers"],
        projection_alpha=1.0,  # Full ablation
        addition_alpha=1.0,    # Full addition
        operation='affine'
    )

    model_unified = get_unified_rdo_model(model, config)
    model_unified.print_trainable_parameters()

    # Test generation
    print("\nTesting generation with affine transformation...")
    prompt = "The future of AI is"
    inputs = tokenizer(prompt, return_tensors='pt')

    with torch.no_grad():
        outputs = model_unified.generate(
            **inputs,
            max_new_tokens=20,
            do_sample=False
        )

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"\nPrompt: {prompt}")
    print(f"Generated: {generated_text}")
    print("\n✓ Unified model working!")

    # Inspect affine components
    print("\n" + "-" * 70)
    print("Inspecting Affine Components")
    print("-" * 70)

    for i, module in enumerate(model_unified.modules()):
        if isinstance(module, UnifiedRDOLayer):
            P, b = module.get_affine_components()
            print(f"\nLayer {i}:")
            print(f"  Projection matrix P: {P.shape}")
            print(f"  Translation vector b: {b.shape}")
            print(f"  ||b|| = {b.norm():.4f}")
            break  # Just show first layer


def demo_rank_k_affine():
    """
    Demo 4: Rank-k affine transformation for complex geometry.
    """
    print("\n" + "=" * 70)
    print("Demo 4: Rank-k Affine for Complex Geometry")
    print("=" * 70)

    print("\nWhen to use rank-k:")
    print("  - Discovered geometry has intrinsic_dimension > 1")
    print("  - Multiple refusal modes found")
    print("  - Single vector doesn't capture full refusal subspace")

    print("\nRank-1 (single vector):")
    print("  h' = (I - vv^T)h + α·v")
    print("  → Removes single direction")

    print("\nRank-k (multi-vector subspace):")
    print("  h' = (I - LL^T)h + α·v_mean")
    print("  where L = [v_1, v_2, ..., v_k]")
    print("  → Removes k-dimensional subspace")

    # Load model
    print("\nLoading GPT-2...")
    model = AutoModelForCausalLM.from_pretrained(
        'gpt2',
        torch_dtype=torch.float32
    )

    # Add rank-k unified RDO
    print("\nAdding rank-3 unified RDO adapters...")
    config = UnifiedRDOConfig(
        target_modules=["layers"],
        projection_alpha=1.0,
        addition_alpha=1.0,
        enable_rank_k=True,
        rank_k=3  # 3-dimensional subspace
    )

    model_rank_k = get_unified_rdo_model(model, config)
    model_rank_k.print_trainable_parameters()

    print("\n✓ Rank-k affine ready for complex geometry!")


def demo_comparison_with_standard():
    """
    Demo 5: Side-by-side comparison with standard RDO.
    """
    print("\n" + "=" * 70)
    print("Demo 5: Side-by-Side Comparison")
    print("=" * 70)

    print("\n" + "-" * 70)
    print("Code Comparison")
    print("-" * 70)

    print("\n[Standard RDO - rdo_peft_adapter.py]")
    print("```python")
    print("class RDOProjectionLayer(nn.Module):")
    print("    def forward(self, x, operation='ablate'):")
    print("        if operation == 'ablate':")
    print("            # h' = h - (h·v)v")
    print("            return h - projection")
    print("        elif operation == 'add':")
    print("            # h' = h + α·v")
    print("            return h + alpha * v")
    print("```")

    print("\n[Unified Affine RDO - unified_rdo_adapter.py]")
    print("```python")
    print("class UnifiedRDOLayer(nn.Module):")
    print("    def forward(self, x):")
    print("        # h' = h - β·(h·v)v + α·v  (always!)")
    print("        return h - projection + addition")
    print("```")

    print("\n" + "-" * 70)
    print("Training Comparison")
    print("-" * 70)

    print("\n[Standard RDO Trainer]")
    print("```python")
    print("def compute_loss(model, inputs):")
    print("    if is_harmful:")
    print("        # Two forward passes")
    print("        model.set_operation('ablate')")
    print("        loss_ablate = model(**inputs).loss")
    print("        ")
    print("        model.set_operation('add')")
    print("        loss_add = model(**inputs).loss")
    print("        ")
    print("        return λ_ablate * loss_ablate + λ_add * loss_add")
    print("```")

    print("\n[Unified Affine Trainer]")
    print("```python")
    print("def compute_loss(model, inputs):")
    print("    # Single forward pass (affine always on)")
    print("    loss = model(**inputs).loss")
    print("    ")
    print("    if is_harmful:")
    print("        return λ_harmful * loss")
    print("    else:")
    print("        return λ_harmless * loss")
    print("```")

    print("\n" + "=" * 70)
    print("Summary: Unified is simpler, faster, and more expressive!")
    print("=" * 70)


def main():
    """Run all demos."""

    print("\n" + "=" * 70)
    print("UNIFIED AFFINE RDO - Complete Examples")
    print("Based on: Refusal in LLMs is an Affine Function")
    print("          https://arxiv.org/abs/2411.09003v3")
    print("=" * 70)

    # Run demos
    demo_affine_transformation()
    demo_single_vs_dual_forward()
    demo_unified_model()
    demo_rank_k_affine()
    demo_comparison_with_standard()

    # Final summary
    print("\n" + "=" * 70)
    print("SUMMARY: Why Use Unified Affine RDO?")
    print("=" * 70)

    print("\n1. Simpler Implementation")
    print("   ✓ Single transformation instead of mode switching")
    print("   ✓ Fewer lines of code")
    print("   ✓ Easier to understand")

    print("\n2. Faster Training")
    print("   ✓ One forward pass instead of two")
    print("   ✓ ~2x speedup on harmful examples")
    print("   ✓ Lower memory usage")

    print("\n3. More Expressive")
    print("   ✓ Ablate AND steer simultaneously")
    print("   ✓ Independent control via β and α")
    print("   ✓ Better gradient flow")

    print("\n4. Theoretically Grounded")
    print("   ✓ Based on affine decomposition theory")
    print("   ✓ Proven effective on Llama 3 70B")
    print("   ✓ Generalizes to complex geometries")

    print("\n" + "=" * 70)
    print("Next Steps")
    print("=" * 70)

    print("\n1. Quick start:")
    print("   from src.training.unified_rdo_trainer import train_unified_rdo")
    print("   model, trainer = train_unified_rdo(")
    print("       model_name='Qwen/Qwen3-0.6B',")
    print("       harmful_data=your_harmful_data,")
    print("       harmless_data=your_harmless_data,")
    print("       projection_alpha=1.0,  # Full ablation")
    print("       addition_alpha=1.0     # Full addition")
    print("   )")

    print("\n2. For discovered geometry:")
    print("   # After running gradient-based discovery")
    print("   results = discovery.discover()")
    print("   ")
    print("   # Use rank-k if intrinsic_dimension > 1")
    print("   train_unified_rdo(")
    print("       enable_rank_k=True,")
    print("       rank_k=len(results['modes'])")
    print("   )")

    print("\n3. Tuning guide:")
    print("   projection_alpha (β):")
    print("     - Start with 1.0 (full ablation)")
    print("     - Reduce if outputs become incoherent")
    print("   ")
    print("   addition_alpha (α):")
    print("     - Start with 1.0 (balanced)")
    print("     - Increase for stronger steering")

    print("\n" + "=" * 70)
    print("✓ Examples complete! Ready to train unified affine RDO.")
    print("=" * 70)


if __name__ == "__main__":
    main()
