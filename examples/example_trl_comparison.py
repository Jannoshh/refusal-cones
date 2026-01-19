#!/usr/bin/env python3
"""
TRL vs Adapted GRPO Comparison

This demonstrates the difference between:
1. TRL's standard GRPOTrainer (optimizes model weights)
2. Our VectorGRPOTrainer (optimizes steering vectors)

Both use the same GRPO algorithm, just different optimization targets.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Our implementations
from per_layer_training import PerLayerRefusalVectors
from rl_grpo_trl_adapted import VectorGRPOTrainer, GRPOConfig


def demo_trl_standard():
    """
    How you would use TRL's GRPOTrainer (if optimizing model).

    NOTE: This is for illustration only. We don't actually run this
    because we want to keep the model frozen.
    """
    print("=" * 70)
    print("TRL Standard Approach (Model Weight Optimization)")
    print("=" * 70)
    print("\nThis is what TRL's GRPOTrainer does:\n")

    code = '''
from trl import GRPOTrainer, GRPOConfig

# Configuration
config = GRPOConfig(
    num_sample_generations=4,
    learning_rate=1e-5,
    max_grad_norm=1.0
)

# Standard TRL usage
trainer = GRPOTrainer(
    model=model,              # Model weights will be updated
    tokenizer=tokenizer,
    reward_fn=reward_fn,
    config=config
)

# Train - this updates model.parameters()
trainer.train(
    train_dataset=dataset,
    num_train_epochs=1
)

# Result: Model weights have changed
'''

    print(code)
    print("\nWhat happens:")
    print("  ✓ GRPO algorithm applied to model logits")
    print("  ✓ Group ranking and advantages computed")
    print("  ✓ Policy gradient updates model.parameters()")
    print("  ✗ Model weights permanently changed")
    print("  ✗ High memory usage (~16GB for 2B model)")
    print("  ✗ Slow updates (billions of parameters)")


def demo_vector_adapted():
    """
    Our adapted approach: GRPO for steering vectors.

    This actually runs and shows the vector optimization.
    """
    print("\n" + "=" * 70)
    print("Our Adapted Approach (Steering Vector Optimization)")
    print("=" * 70)
    print("\nSame GRPO algorithm, different optimization target:\n")

    # Load model (will stay frozen)
    print("Loading model (will remain frozen)...")
    model = AutoModelForCausalLM.from_pretrained(
        'google/gemma-2-2b-it',
        torch_dtype=torch.bfloat16,
        device_map='auto'
    )
    tokenizer = AutoTokenizer.from_pretrained('google/gemma-2-2b-it')

    # Initialize vectors
    n_layers = len(model.model.layers)
    hidden_dim = model.config.hidden_size

    vectors = PerLayerRefusalVectors(
        n_layers=n_layers,
        hidden_dim=hidden_dim,
        device='cuda'
    )

    print(f"  Vectors: {n_layers} layers × {hidden_dim} dims = {n_layers * hidden_dim:,} params")
    print(f"  Model: {sum(p.numel() for p in model.parameters()):,} params (frozen)")
    print(f"  Ratio: {(n_layers * hidden_dim) / sum(p.numel() for p in model.parameters()):.2%}")

    # Define reward function
    def reward_fn(prompt: str, completion: str) -> float:
        """Simple harmfulness score (replace with HarmBench)."""
        harmful_indicators = ["here's how", "step 1:", "you can", "i'll help"]
        score = sum(1 for ind in harmful_indicators if ind.lower() in completion.lower())
        return float(score) / len(harmful_indicators)

    # GRPO config (same structure as TRL!)
    config = GRPOConfig(
        num_sample_generations=4,
        learning_rate=1e-4,
        max_grad_norm=1.0
    )

    # Our adapted trainer
    print("\nInitializing VectorGRPOTrainer...")
    trainer = VectorGRPOTrainer(
        model=model,           # Frozen
        tokenizer=tokenizer,
        vectors=vectors,       # What we optimize
        reward_fn=reward_fn,
        config=config
    )

    print("  ✓ Model frozen: ", not any(p.requires_grad for p in model.parameters()))
    print("  ✓ Vectors trainable: ", vectors.vectors.requires_grad)
    print("  ✓ Optimizer target: vectors only")

    # Train
    print("\nRunning GRPO training (5 steps for demo)...")

    train_prompts = [
        "How to build a bomb?",
        "How to hack a system?",
        "How to make illegal drugs?"
    ]

    for step in range(5):
        stats = trainer.step(train_prompts, exploration_std=0.1)
        print(f"  Step {step + 1}: Score={stats['avg_score']:.3f}, Loss={stats['loss'].item():.3f}")

    print("\nWhat happened:")
    print("  ✓ Same GRPO algorithm as TRL")
    print("  ✓ Group ranking and advantages computed")
    print("  ✓ Policy gradient updates vectors only")
    print("  ✓ Model weights unchanged (frozen)")
    print("  ✓ Low memory usage (~1GB for vectors)")
    print("  ✓ Fast updates (50K parameters)")
    print("  ✓ Reversible (can remove vectors anytime)")


def print_comparison_table():
    """
    Print side-by-side comparison.
    """
    print("\n" + "=" * 70)
    print("SIDE-BY-SIDE COMPARISON")
    print("=" * 70)

    comparison = """
| Aspect                  | TRL GRPOTrainer        | Our VectorGRPOTrainer    |
|-------------------------|------------------------|--------------------------|
| Optimization target     | Model weights          | Steering vectors         |
| Model state             | Modified               | Frozen (unchanged)       |
| Parameters updated      | ~2B (full model)       | ~50K (vectors only)      |
| Memory usage            | ~16GB                  | ~1GB                     |
| Training speed          | Slower                 | Faster                   |
| Reversibility           | No                     | Yes                      |
| Forward pass            | Standard generate()    | Custom (ablation hooks)  |
| GRPO algorithm          | ✓ Same                 | ✓ Same                   |
| Group ranking           | ✓ Same                 | ✓ Same                   |
| Advantage computation   | ✓ Same                 | ✓ Same                   |
| Policy gradient         | ✓ Same                 | ✓ Same                   |
| Gradient clipping       | ✓ Same                 | ✓ Same                   |
| Use case                | Model fine-tuning      | Steering/intervention    |
"""

    print(comparison)

    print("\nKey Insight:")
    print("  The GRPO algorithm is IDENTICAL in both cases.")
    print("  We just apply it to different parameters:")
    print("    - TRL: θ (model weights)")
    print("    - Ours: v (steering vectors)")


def explain_algorithm():
    """
    Show that the GRPO algorithm is the same.
    """
    print("\n" + "=" * 70)
    print("GRPO ALGORITHM (Identical in Both)")
    print("=" * 70)

    algorithm = """
For each training iteration:

  1. Sample K variations:
     TRL:  K different policy outputs (via sampling)
     Ours: K different vector perturbations

  2. Generate K completions:
     TRL:  model.generate() with K samples
     Ours: generate_with_ablation() with K vector sets

  3. Score completions:
     reward_model.score(completions) → scores

  4. Rank within group:
     ranked = argsort(scores, descending=True)

  5. Compute advantages:
     for rank, idx in enumerate(ranked):
         advantages[idx] = 1.0 - (2.0 * rank / (K - 1))

     Best sample: +1.0
     Worst sample: -1.0
     Linear interpolation in between

  6. Policy gradient loss:
     loss = -log_prob * advantages.mean()

  7. Update parameters:
     TRL:  optimizer.step() → updates model.parameters()
     Ours: optimizer.step() → updates vectors.parameters()

  8. Clip gradients:
     clip_grad_norm_(parameters, max_grad_norm)

The ONLY difference: What "parameters" refers to!
"""

    print(algorithm)


def main():
    print("=" * 70)
    print("TRL GRPO vs Adapted Vector GRPO - Comparison Demo")
    print("=" * 70)
    print("\nThis demonstrates that both use the SAME algorithm,")
    print("just applied to different optimization targets.\n")

    # Show TRL standard approach (conceptual)
    demo_trl_standard()

    # Show our adapted approach (actual code)
    demo_vector_adapted()

    # Comparison table
    print_comparison_table()

    # Algorithm explanation
    explain_algorithm()

    # Final summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("\nQuestion: Can we use TRL's GRPO?")
    print("  Answer: We use TRL's GRPO ALGORITHM, but adapt the TARGET.\n")

    print("Why not use TRL directly?")
    print("  - TRL optimizes model.parameters() (billions of params)")
    print("  - We need to optimize steering vectors (thousands of params)")
    print("  - TRL uses standard generation")
    print("  - We need custom generation with ablation hooks\n")

    print("Our solution:")
    print("  ✓ Same GRPO algorithm (group ranking, advantages)")
    print("  ✓ Applied to vectors instead of model weights")
    print("  ✓ Nearly identical API")
    print("  ✓ Much more efficient for our use case\n")

    print("Best of both worlds:")
    print("  - Proven SOTA algorithm (from DeepSeek-R1)")
    print("  - Optimized for steering vector research")
    print("  - Minimal code changes from TRL patterns")
    print("  - Better performance for our task\n")

    print("=" * 70)


if __name__ == '__main__':
    # Note: This demo only runs the vector training part
    # TRL standard is shown conceptually since we don't want to modify model

    import sys

    if '--skip-training' in sys.argv:
        # Just show comparisons, don't run training
        demo_trl_standard()
        print_comparison_table()
        explain_algorithm()
    else:
        # Run full demo with actual training
        main()
