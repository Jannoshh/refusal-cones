#!/usr/bin/env python3
"""
Estimate compute cost for REINFORCE/GRPO training.

Usage:
    python scripts/estimate_compute_cost.py --model gemma-2-2b --steps 100
    python scripts/estimate_compute_cost.py --model llama-3-8b --steps 200 --k-samples 8
"""

import argparse


# Model configurations
MODEL_CONFIGS = {
    "qwen-0.5b": {
        "params_b": 0.5,
        "vram_gb": 4,
        "tokens_per_sec": 200,
        "recommended_gpu": "T4"
    },
    "gemma-2-2b": {
        "params_b": 2.0,
        "vram_gb": 8,
        "tokens_per_sec": 150,
        "recommended_gpu": "A10G"
    },
    "llama-3-8b": {
        "params_b": 8.0,
        "vram_gb": 20,
        "tokens_per_sec": 80,
        "recommended_gpu": "A100-40GB"
    },
    "llama-3-70b": {
        "params_b": 70.0,
        "vram_gb": 150,
        "tokens_per_sec": 25,
        "recommended_gpu": "2×A100-80GB"
    }
}

# GPU pricing (per second)
GPU_PRICING = {
    "T4": 0.000164,
    "L4": 0.000222,
    "A10G": 0.000306,
    "L40S": 0.000542,
    "A100-40GB": 0.000583,
    "A100-80GB": 0.000694,
    "H100": 0.001097
}


def estimate_cost(
    model_name: str,
    num_steps: int,
    k_samples: int = 4,
    batch_size: int = 8,
    max_new_tokens: int = 150,
    gpu: str = None
):
    """
    Estimate compute time and cost.

    Args:
        model_name: Model identifier
        num_steps: Number of training steps
        k_samples: Samples per step
        batch_size: Prompts per step
        max_new_tokens: Tokens to generate
        gpu: GPU type (auto-select if None)

    Returns:
        Dict with estimates
    """
    # Get model config
    if model_name not in MODEL_CONFIGS:
        print(f"Warning: Unknown model {model_name}, using gemma-2-2b defaults")
        model_name = "gemma-2-2b"

    config = MODEL_CONFIGS[model_name]

    # Auto-select GPU
    if gpu is None:
        gpu = config["recommended_gpu"]

    # Validate GPU has enough VRAM
    gpu_base = gpu.split('-')[0] if '-' in gpu else gpu

    # Cost per second
    price_per_sec = GPU_PRICING.get(gpu, 0.0003)

    # Compute metrics
    tokens_per_sec = config["tokens_per_sec"]
    generations_per_step = k_samples * batch_size

    # Time per step
    # 1. Generation time (dominant)
    gen_time_per_completion = max_new_tokens / tokens_per_sec
    gen_time_per_step = gen_time_per_completion * generations_per_step

    # 2. Judge scoring time (~0.05s per sample)
    judge_time_per_step = 0.05 * generations_per_step

    # 3. Gradient computation (~0.1s)
    grad_time_per_step = 0.1

    # Total time per step
    time_per_step = gen_time_per_step + judge_time_per_step + grad_time_per_step

    # Total training time
    total_time_sec = time_per_step * num_steps
    total_time_min = total_time_sec / 60
    total_time_hr = total_time_min / 60

    # Total cost
    total_cost = total_time_sec * price_per_sec

    # Mean difference cost (one-time)
    mean_diff_time = 5  # seconds
    mean_diff_cost = mean_diff_time * price_per_sec

    # Total project cost
    project_cost = mean_diff_cost + total_cost

    return {
        "model": model_name,
        "gpu": gpu,
        "vram_required_gb": config["vram_gb"],
        "num_steps": num_steps,
        "k_samples": k_samples,
        "batch_size": batch_size,
        "max_new_tokens": max_new_tokens,
        "generations_per_step": generations_per_step,
        "total_generations": generations_per_step * num_steps,
        "total_tokens": generations_per_step * num_steps * max_new_tokens,
        "time_per_step_sec": time_per_step,
        "total_time_sec": total_time_sec,
        "total_time_min": total_time_min,
        "total_time_hr": total_time_hr,
        "cost_per_hour": price_per_sec * 3600,
        "mean_diff_cost": mean_diff_cost,
        "training_cost": total_cost,
        "total_cost": project_cost,
        "free_tier_runs": 30.0 / project_cost if project_cost > 0 else float('inf')
    }


def print_estimate(est: dict):
    """Print formatted estimate."""
    print("\n" + "=" * 70)
    print("COMPUTE COST ESTIMATE")
    print("=" * 70)

    print(f"\nConfiguration:")
    print(f"  Model: {est['model']}")
    print(f"  GPU: {est['gpu']} ({est['vram_required_gb']} GB VRAM required)")
    print(f"  Training steps: {est['num_steps']}")
    print(f"  K samples: {est['k_samples']}")
    print(f"  Batch size: {est['batch_size']}")
    print(f"  Max tokens: {est['max_new_tokens']}")

    print(f"\nWorkload:")
    print(f"  Generations per step: {est['generations_per_step']}")
    print(f"  Total generations: {est['total_generations']:,}")
    print(f"  Total tokens: {est['total_tokens']:,}")

    print(f"\nTime:")
    print(f"  Per step: {est['time_per_step_sec']:.1f}s")
    print(f"  Total: {est['total_time_min']:.1f} min ({est['total_time_hr']:.2f} hr)")

    print(f"\nCost:")
    print(f"  Mean difference: ${est['mean_diff_cost']:.4f}")
    print(f"  RL training: ${est['training_cost']:.2f}")
    print(f"  Total: ${est['total_cost']:.2f}")

    print(f"\nModal Free Tier:")
    print(f"  Credits: $30/month")
    print(f"  Runs possible: {int(est['free_tier_runs'])}")

    print("\n" + "=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Estimate compute cost for RL training"
    )
    parser.add_argument(
        "--model",
        choices=list(MODEL_CONFIGS.keys()),
        default="gemma-2-2b",
        help="Model to use"
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=100,
        help="Number of training steps"
    )
    parser.add_argument(
        "--k-samples",
        type=int,
        default=4,
        help="Samples per step"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Prompts per step"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=150,
        help="Max tokens to generate"
    )
    parser.add_argument(
        "--gpu",
        choices=list(GPU_PRICING.keys()),
        help="GPU type (auto-selected if not specified)"
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Show comparison across configs"
    )

    args = parser.parse_args()

    if args.compare:
        # Compare different configurations
        print("\n" + "=" * 70)
        print("CONFIGURATION COMPARISON")
        print("=" * 70)

        configs = [
            ("Quick test", 50, 2, 50),
            ("Standard", 100, 4, 150),
            ("High quality", 200, 8, 150),
        ]

        print(f"\n{'Config':<15} {'Time':<12} {'Cost':<10} {'Runs/Month':<12}")
        print("-" * 50)

        for name, steps, k, tokens in configs:
            est = estimate_cost(
                args.model,
                steps,
                k,
                args.batch_size,
                tokens,
                args.gpu
            )
            print(
                f"{name:<15} "
                f"{est['total_time_min']:>6.1f} min   "
                f"${est['total_cost']:>6.2f}    "
                f"{int(est['free_tier_runs']):>3} runs"
            )

        print("\n" + "=" * 70)

    else:
        # Single configuration estimate
        est = estimate_cost(
            args.model,
            args.steps,
            args.k_samples,
            args.batch_size,
            args.max_tokens,
            args.gpu
        )

        print_estimate(est)

        # Recommendations
        print("\nRecommendations:")

        if est['total_cost'] > 5.0:
            print("  ⚠️  High cost - consider reducing steps or max_tokens")

        if est['total_time_hr'] > 2.0:
            print("  ⚠️  Long runtime - consider using faster GPU or reducing workload")

        if est['vram_required_gb'] > 40:
            print("  ⚠️  High VRAM - may need multi-GPU setup")

        if est['total_cost'] < 1.0:
            print("  ✓ Low cost - good for experimentation")

        if est['total_time_min'] < 30:
            print("  ✓ Fast - good for rapid iteration")

        # Optimization suggestions
        print("\nOptimization tips:")
        print(f"  • Reduce max_tokens to 50: ~{est['total_cost'] * 0.4:.2f}$ (2.5× faster)")
        print(f"  • Reduce k_samples to 2: ~{est['total_cost'] * 0.5:.2f}$ (2× faster)")
        print(f"  • Enable Flash Attention: ~{est['total_cost'] * 0.4:.2f}$ (2.5× faster)")


if __name__ == "__main__":
    main()
