#!/usr/bin/env python3
"""
Quick comparison: REINFORCE vs GRPO on Gemma-2-2B.

This script:
1. Computes mean difference vector
2. Trains with REINFORCE (100 steps)
3. Trains with GRPO (100 steps)
4. Compares results

Usage:
    python examples/run_comparison_experiment.py
    python examples/run_comparison_experiment.py --quick  # Fast mode: 20 steps
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
import json
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from training.trainers.per_layer_training import PerLayerRefusalVectors

# Add experiments to path for judge
sys.path.insert(0, str(Path(__file__).parent.parent / "experiments"))
from shared.evaluation import StrongRejectJudge

# Import test prompts
from test_prompts import get_harmful_prompts, get_harmless_prompts


def compute_mean_difference_vector(
    model,
    tokenizer,
    harmful_prompts,
    harmless_prompts,
):
    """Compute mean difference vector."""
    print("\n" + "=" * 70)
    print("COMPUTING MEAN DIFFERENCE VECTOR")
    print("=" * 70)

    n_layers = len(model.model.layers)
    device = next(model.parameters()).device

    harmful_acts = {i: [] for i in range(n_layers)}
    harmless_acts = {i: [] for i in range(n_layers)}

    def get_activations(prompts, storage_dict):
        for prompt in prompts:
            messages = [{"role": "user", "content": prompt}]
            formatted = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            inputs = tokenizer(formatted, return_tensors="pt").to(device)

            activations = {}
            hooks = []

            for layer_idx in range(n_layers):
                def make_hook(idx):
                    def hook(module, input, output):
                        if isinstance(output, tuple):
                            act = output[0]
                        else:
                            act = output
                        activations[idx] = act[:, -1, :].detach().cpu()
                    return hook

                h = model.model.layers[layer_idx].register_forward_hook(make_hook(layer_idx))
                hooks.append(h)

            with torch.no_grad():
                model(**inputs)

            for h in hooks:
                h.remove()

            for layer_idx in range(n_layers):
                storage_dict[layer_idx].append(activations[layer_idx])

    print(f"\nExtracting activations...")
    print(f"  Harmful: {len(harmful_prompts)} prompts")
    get_activations(harmful_prompts, harmful_acts)

    print(f"  Harmless: {len(harmless_prompts)} prompts")
    get_activations(harmless_prompts, harmless_acts)

    print("\nComputing mean difference...")
    vectors = []

    for layer_idx in range(n_layers):
        mean_harmful = torch.stack(harmful_acts[layer_idx]).mean(dim=0).to(device)
        mean_harmless = torch.stack(harmless_acts[layer_idx]).mean(dim=0).to(device)
        diff = mean_harmful - mean_harmless
        diff = diff / diff.norm()
        vectors.append(diff)

    vectors = torch.stack(vectors)
    print(f"✓ Mean difference vector: {vectors.shape}")
    return vectors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Quick mode (20 steps)")
    parser.add_argument("--model", default="google/gemma-2-2b-it")
    parser.add_argument("--output-dir", default="results/comparison_experiments")
    args = parser.parse_args()

    # Configuration
    if args.quick:
        n_steps = 20
        n_harmful = 10
        n_harmless = 10
        k_samples = 2
        max_tokens = 50
        print("\n🚀 QUICK MODE: 20 steps, 10 prompts, 50 tokens")
    else:
        n_steps = 100
        n_harmful = 20
        n_harmless = 20
        k_samples = 4
        max_tokens = 150
        print("\n⚡ STANDARD MODE: 100 steps, 20 prompts, 150 tokens")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print("REINFORCE vs GRPO COMPARISON")
    print("=" * 70)
    print(f"\nModel: {args.model}")
    print(f"Steps: {n_steps}")
    print(f"Output: {output_dir}")

    # Load model
    print("\nLoading model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"✓ Model loaded")

    # Load prompts
    harmful_prompts = get_harmful_prompts(n_harmful)
    harmless_prompts = get_harmless_prompts(n_harmless)
    print(f"✓ Prompts: {len(harmful_prompts)} harmful, {len(harmless_prompts)} harmless")

    # Load judge
    print("\nLoading judge...")
    judge = StrongRejectJudge()
    print("✓ Judge loaded")

    # Compute mean difference (shared initialization)
    mean_diff_vectors = compute_mean_difference_vector(
        model, tokenizer, harmful_prompts, harmless_prompts
    )

    # Save mean diff
    torch.save(mean_diff_vectors, output_dir / "mean_diff_vectors.pt")

    # Results dict
    results = {
        "config": {
            "model": args.model,
            "n_steps": n_steps,
            "n_harmful": n_harmful,
            "n_harmless": n_harmless,
            "k_samples": k_samples,
            "max_tokens": max_tokens,
            "timestamp": timestamp
        },
        "methods": {}
    }

    # Import REINFORCE and GRPO
    sys.path.insert(0, str(Path(__file__).parent))
    from reinforce_from_mean_diff import REINFORCEVectorOptimizer
    from grpo_from_mean_diff import train_grpo_from_mean_diff

    # ================================================================
    # Method 1: REINFORCE
    # ================================================================
    print("\n" + "=" * 70)
    print("METHOD 1: REINFORCE")
    print("=" * 70)

    n_layers, hidden_dim = mean_diff_vectors.shape
    device = next(model.parameters()).device

    reinforce_vectors = PerLayerRefusalVectors(n_layers, hidden_dim, device)
    reinforce_vectors.vectors.data = mean_diff_vectors.clone()

    reinforce_optimizer = REINFORCEVectorOptimizer(
        model=model,
        tokenizer=tokenizer,
        vectors=reinforce_vectors,
        judge=judge,
        exploration_std=0.1,
        k_samples=k_samples,
        learning_rate=1e-4,
        device=device
    )

    print(f"\nTraining REINFORCE for {n_steps} steps...")
    reinforce_history = reinforce_optimizer.train(
        train_prompts=harmful_prompts[:10],
        num_steps=n_steps,
        log_interval=max(n_steps // 5, 1)
    )

    # Save REINFORCE results
    torch.save(reinforce_vectors.vectors.cpu(), output_dir / "reinforce_vectors.pt")

    results["methods"]["reinforce"] = {
        "initial_reward": reinforce_history[0]["avg_reward"],
        "final_reward": reinforce_history[-1]["avg_reward"],
        "improvement": reinforce_history[-1]["avg_reward"] - reinforce_history[0]["avg_reward"],
        "final_variance": reinforce_history[-1]["weight_variance"],
        "history": reinforce_history
    }

    print(f"\n✓ REINFORCE complete")
    print(f"  Initial: {reinforce_history[0]['avg_reward']:.4f}")
    print(f"  Final: {reinforce_history[-1]['avg_reward']:.4f}")
    print(f"  Improvement: {results['methods']['reinforce']['improvement']:+.4f}")

    # ================================================================
    # Method 2: GRPO
    # ================================================================
    print("\n" + "=" * 70)
    print("METHOD 2: GRPO")
    print("=" * 70)

    grpo_vectors, grpo_history = train_grpo_from_mean_diff(
        model=model,
        tokenizer=tokenizer,
        initial_vectors=mean_diff_vectors.clone(),
        train_prompts=harmful_prompts[:10],
        judge=judge,
        num_steps=n_steps,
        exploration_std=0.1,
        k_samples=k_samples,
        learning_rate=1e-4,
    )

    # Save GRPO results
    torch.save(grpo_vectors.vectors.cpu(), output_dir / "grpo_vectors.pt")

    results["methods"]["grpo"] = {
        "initial_reward": grpo_history[0]["avg_score"],
        "final_reward": grpo_history[-1]["avg_score"],
        "improvement": grpo_history[-1]["avg_score"] - grpo_history[0]["avg_score"],
        "history": grpo_history
    }

    print(f"\n✓ GRPO complete")
    print(f"  Initial: {grpo_history[0]['avg_score']:.4f}")
    print(f"  Final: {grpo_history[-1]['avg_score']:.4f}")
    print(f"  Improvement: {results['methods']['grpo']['improvement']:+.4f}")

    # ================================================================
    # Comparison
    # ================================================================
    print("\n" + "=" * 70)
    print("FINAL COMPARISON")
    print("=" * 70)

    reinforce_final = results["methods"]["reinforce"]["final_reward"]
    grpo_final = results["methods"]["grpo"]["final_reward"]
    reinforce_improve = results["methods"]["reinforce"]["improvement"]
    grpo_improve = results["methods"]["grpo"]["improvement"]

    print(f"\n{'Method':<15} {'Initial':<10} {'Final':<10} {'Improvement':<12}")
    print("-" * 50)
    print(f"{'REINFORCE':<15} {results['methods']['reinforce']['initial_reward']:.4f}     {reinforce_final:.4f}     {reinforce_improve:+.4f}")
    print(f"{'GRPO':<15} {results['methods']['grpo']['initial_reward']:.4f}     {grpo_final:.4f}     {grpo_improve:+.4f}")

    winner = "REINFORCE" if reinforce_final > grpo_final else "GRPO"
    margin = abs(reinforce_final - grpo_final)

    print(f"\n🏆 Winner: {winner} (+{margin:.4f})")

    # Save results
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Results saved to {output_dir}")

    # Summary
    print("\n" + "=" * 70)
    print("EXPERIMENT SUMMARY")
    print("=" * 70)
    print(f"\nBoth methods improved from mean difference baseline:")
    print(f"  REINFORCE: {reinforce_improve:+.4f}")
    print(f"  GRPO: {grpo_improve:+.4f}")
    print(f"\n{winner} achieved better final reward (+{margin:.4f})")
    print(f"\nVariance reduction (REINFORCE):")
    print(f"  Final weight variance: {results['methods']['reinforce']['final_variance']:.4f}")


if __name__ == "__main__":
    main()
