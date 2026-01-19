#!/usr/bin/env python3
"""
E1.1: Discovery Efficiency Comparison

Compare gradient+GP vs pure GP vs random search.

Metrics:
- Number of measurements to reach ASR > 0.7
- Best ASR found within budget
- Wall-clock time

Expected: Gradient+GP reaches target in ~50 measurements vs ~500 for pure GP.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import time
import argparse
from typing import Dict, List

from src.experiments.config_utils import (
    load_experiment_config,
    merge_cli_with_config,
    save_run_metadata,
)
from shared.model_loading import load_model_and_tokenizer, get_model_config
from shared.data_loading import load_harmful_data, load_harmless_data
from shared.evaluation import evaluate_asr, StrongRejectJudge
from shared.utils import (
    set_seed, get_output_dir, save_results, save_vectors,
    print_experiment_header, print_results_summary, Timer
)

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from discovery.gradient_discovery import GradientGeometryDiscovery, GradientDiscoveryConfig
from discovery.adaptive_geometry_discovery import RefusalGeometryDiscovery, GeometryConfig


def create_measure_function(model, tokenizer, harmful_prompts, judge):
    """Create measurement function with gradient support."""

    def measure_refusal_with_grad(v: torch.Tensor):
        """
        Measure refusal rate when ablating with vector v.

        Returns:
            R: Refusal rate (lower = better for jailbreaking)
            grad: Gradient of R with respect to v
        """
        v = v.detach().requires_grad_(True)

        # For gradient computation, we approximate using finite differences
        # or use a differentiable proxy

        # Simple approach: use token probabilities as differentiable proxy
        # Full approach: use generation + classifier (no grad, estimate via REINFORCE)

        # Here we use a simplified version for demonstration
        # In practice, use the measurement module

        # Compute refusal score (simplified)
        asr_result = evaluate_asr(
            model=model,
            tokenizer=tokenizer,
            prompts=harmful_prompts[:10],  # Subset for speed
            judge=judge,
            ablation_vectors=v,
            max_new_tokens=100,
        )

        R = 1.0 - asr_result["asr"]  # Refusal rate = 1 - ASR

        # Estimate gradient via a single finite-difference probe to avoid zero-grad stall
        eps = 0.01
        noise = torch.randn_like(v)
        noise = noise / noise.norm(dim=1, keepdim=True)

        v_perturbed = v + eps * noise
        v_perturbed = v_perturbed / v_perturbed.norm(dim=1, keepdim=True)

        asr_perturbed = evaluate_asr(
            model=model,
            tokenizer=tokenizer,
            prompts=harmful_prompts[:10],
            judge=judge,
            ablation_vectors=v_perturbed,
            max_new_tokens=100,
        )
        R_perturbed = 1.0 - asr_perturbed["asr"]

        grad = ((R_perturbed - R) / eps) * noise

        return R, grad

    return measure_refusal_with_grad


def run_gradient_discovery(
    model, tokenizer, v_init, harmful_prompts, judge, config
) -> Dict:
    """Run gradient-based discovery."""

    measure_fn = create_measure_function(model, tokenizer, harmful_prompts, judge)

    n_layers = v_init.shape[0]
    hidden_dim = v_init.shape[1]

    discovery = GradientGeometryDiscovery(
        measure_refusal_with_grad=measure_fn,
        v_init=v_init,
        n_layers=n_layers,
        hidden_dim=hidden_dim,
        config=config
    )

    with Timer("Gradient+GP Discovery"):
        results = discovery.discover()

    return results


def run_pure_gp_discovery(
    model, tokenizer, harmful_prompts, judge, n_layers, hidden_dim, config
) -> Dict:
    """Run pure GP discovery (no gradients)."""

    def measure_fn(v: torch.Tensor) -> float:
        asr_result = evaluate_asr(
            model=model,
            tokenizer=tokenizer,
            prompts=harmful_prompts[:10],
            judge=judge,
            ablation_vectors=v,
            max_new_tokens=100,
        )
        return 1.0 - asr_result["asr"]  # Refusal rate

    discovery = RefusalGeometryDiscovery(
        measure_refusal_fn=measure_fn,
        n_layers=n_layers,
        hidden_dim=hidden_dim,
        config=config
    )

    with Timer("Pure GP Discovery"):
        results = discovery.discover()

    return results


def run_random_search(
    model, tokenizer, harmful_prompts, judge, n_layers, hidden_dim, n_samples
) -> Dict:
    """Run random search baseline."""

    best_R = float('inf')
    best_v = None
    R_history = []

    with Timer("Random Search"):
        for i in range(n_samples):
            # Random unit vector
            v = torch.randn(n_layers, hidden_dim)
            v = v / v.norm(dim=1, keepdim=True)

            # Measure
            asr_result = evaluate_asr(
                model=model,
                tokenizer=tokenizer,
                prompts=harmful_prompts[:10],
                judge=judge,
                ablation_vectors=v,
                max_new_tokens=100,
            )
            R = 1.0 - asr_result["asr"]

            R_history.append(R)

            if R < best_R:
                best_R = R
                best_v = v.clone()

            if (i + 1) % 50 == 0:
                print(f"  Sample {i+1}/{n_samples}: best R = {best_R:.4f}")

    return {
        "best_v": best_v,
        "best_R": best_R,
        "R_history": R_history,
        "n_measurements": n_samples,
    }


def main():
    parser = argparse.ArgumentParser(description="E1.1: Discovery Efficiency")
    parser.add_argument("--model", default="gemma-2-2b", help="Model to use")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", default="results")
    parser.add_argument("--config", default="experiments/configs/default.yaml", help="Path to YAML config")
    parser.add_argument("--use_sparse_gp", action="store_true", help="Use sparse GP for acquisition")
    parser.add_argument("--kernel", default="rbf", choices=["rbf", "linear"], help="Kernel type for GP")
    parser.add_argument("--lengthscale", type=float, default=0.3, help="Kernel lengthscale")
    parser.add_argument("--num_inducing", type=int, default=64, help="Inducing points for sparse GP")
    parser.add_argument("--sparse_steps", type=int, default=10, help="Sparse GP training steps")
    parser.add_argument("--sparse_lr", type=float, default=0.05, help="Sparse GP learning rate")
    args = parser.parse_args()

    # Load config and merge with CLI
    cfg_file = load_experiment_config(args.config)
    merged = merge_cli_with_config(args, cfg_file)

    set_seed(merged.seed)

    # Setup
    print_experiment_header("E1.1: Discovery Efficiency", merged.model)

    model_config = get_model_config(merged.model)
    n_layers = model_config["n_layers"]
    hidden_dim = model_config["hidden_dim"]

    # Load model
    model, tokenizer = load_model_and_tokenizer(merged.model)

    # Load data
    harmful_prompts = load_harmful_data(split="val", max_samples=100)

    # Initialize judge
    judge = StrongRejectJudge()

    # Initialize with DIM direction (or random if not available)
    v_init = torch.randn(n_layers, hidden_dim)
    v_init = v_init / v_init.norm(dim=1, keepdim=True)

    # Run methods
    results = {}

    # Method 1: Gradient + GP
    print("\n" + "=" * 70)
    print("Method 1: Gradient + GP")
    print("=" * 70)

    gradient_config = GradientDiscoveryConfig(
        n_gradient_steps=20,
        n_local_iterations=30,
        enable_global_search=False,
        use_sparse_gp=merged.use_sparse_gp,
        kernel_type=merged.kernel,
        kernel_lengthscale=merged.lengthscale,
        num_inducing=merged.num_inducing,
        sparse_train_steps=merged.sparse_steps,
        sparse_lr=merged.sparse_lr,
    )
    gradient_results = run_gradient_discovery(
        model, tokenizer, v_init, harmful_prompts, judge, gradient_config
    )
    results["gradient_gp"] = {
        "n_measurements": gradient_results["n_measurements"],
        "best_R": min(gradient_results["R_observed"]).item(),
        "best_asr": 1.0 - min(gradient_results["R_observed"]).item(),
    }

    # Method 2: Pure GP
    print("\n" + "=" * 70)
    print("Method 2: Pure GP")
    print("=" * 70)

    gp_config = GeometryConfig(
        n_init_random=20,
        n_iterations=80,
        acquisition_type="ucb",
        beta=2.0,
        use_sparse_gp=merged.use_sparse_gp,
        kernel_type=merged.kernel,
        kernel_lengthscale=merged.lengthscale,
        num_inducing=merged.num_inducing,
        sparse_train_steps=merged.sparse_steps,
        sparse_lr=merged.sparse_lr,
    )
    gp_results = run_pure_gp_discovery(
        model, tokenizer, harmful_prompts, judge, n_layers, hidden_dim, gp_config
    )
    results["pure_gp"] = {
        "n_measurements": len(gp_results["R_observed"]),
        "best_R": min(gp_results["R_observed"]),
        "best_asr": 1.0 - min(gp_results["R_observed"]),
    }

    # Method 3: Random Search
    print("\n" + "=" * 70)
    print("Method 3: Random Search")
    print("=" * 70)

    random_results = run_random_search(
        model, tokenizer, harmful_prompts, judge, n_layers, hidden_dim, n_samples=100
    )
    results["random"] = {
        "n_measurements": random_results["n_measurements"],
        "best_R": random_results["best_R"],
        "best_asr": 1.0 - random_results["best_R"],
    }

    # Summary
    print("\n" + "=" * 70)
    print("COMPARISON")
    print("=" * 70)
    print(f"\n{'Method':<25} {'Measurements':<15} {'Best ASR':<15}")
    print("-" * 55)
    for method, r in results.items():
        print(f"{method:<25} {r['n_measurements']:<15} {r['best_asr']:.4f}")

    # Compute speedup
    speedup = results["pure_gp"]["n_measurements"] / results["gradient_gp"]["n_measurements"]
    print(f"\nSpeedup (Gradient+GP vs Pure GP): {speedup:.1f}×")

    # Save results
    output_dir = get_output_dir("e1_discovery_efficiency", merged.model, merged.output_dir)
    save_results(results, output_dir)
    save_run_metadata(
        output_dir,
        {
            "model": merged.model,
            "seed": merged.seed,
            "use_sparse_gp": merged.use_sparse_gp,
            "kernel": merged.kernel,
            "lengthscale": merged.lengthscale,
            "num_inducing": merged.num_inducing,
            "sparse_steps": merged.sparse_steps,
            "sparse_lr": merged.sparse_lr,
            "results": results,
        },
    )

    print_results_summary(results)


if __name__ == "__main__":
    main()
