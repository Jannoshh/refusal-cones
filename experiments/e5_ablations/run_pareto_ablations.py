#!/usr/bin/env python3
"""
Hyperparameter ablations for Pareto boundary discovery.

Tests key hyperparameters and reports quality metrics:
- Hypervolume (HV): volume dominated by Pareto frontier
- Pareto size: number of non-dominated vectors
- Best refusal: minimum refusal score achieved
- Best KL: minimum KL divergence achieved
- Spread: distribution of Pareto points
"""

import torch
import json
import itertools
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional
from pathlib import Path
import time


@dataclass
class AblationConfig:
    """Single ablation configuration."""
    name: str
    n_init_samples: int = 30
    n_pareto_iterations: int = 70
    kernel_lengthscale: float = 0.3
    beta: float = 1.96
    n_candidates: int = 500
    use_sparse_gp: bool = False
    num_inducing: int = 64


@dataclass
class AblationResult:
    """Results from a single ablation run."""
    config: AblationConfig
    hypervolume: float
    pareto_size: int
    best_refusal: float
    best_kl: float
    spread: float  # std of Pareto points along objectives
    n_measurements: int
    wall_time: float
    hv_per_measurement: float  # efficiency metric

    def to_dict(self) -> Dict[str, Any]:
        return {
            'config': asdict(self.config),
            'hypervolume': self.hypervolume,
            'pareto_size': self.pareto_size,
            'best_refusal': self.best_refusal,
            'best_kl': self.best_kl,
            'spread': self.spread,
            'n_measurements': self.n_measurements,
            'wall_time': self.wall_time,
            'hv_per_measurement': self.hv_per_measurement
        }


def compute_spread(pareto_scores: Dict[str, torch.Tensor]) -> float:
    """Compute spread of Pareto points (average std across objectives)."""
    spreads = []
    for obj in ['refusal_score', 'kl_score']:
        if obj in pareto_scores and len(pareto_scores[obj]) > 1:
            spreads.append(pareto_scores[obj].std().item())
    return sum(spreads) / len(spreads) if spreads else 0.0


def get_ablation_configs() -> List[AblationConfig]:
    """Generate ablation configurations to test."""
    configs = []

    # Baseline
    configs.append(AblationConfig(name="baseline"))

    # Vary n_init_samples
    for n_init in [10, 20, 50]:
        configs.append(AblationConfig(
            name=f"n_init_{n_init}",
            n_init_samples=n_init
        ))

    # Vary n_pareto_iterations
    for n_iter in [30, 50, 100]:
        configs.append(AblationConfig(
            name=f"n_iter_{n_iter}",
            n_pareto_iterations=n_iter
        ))

    # Vary kernel lengthscale
    for ls in [0.1, 0.2, 0.5, 1.0]:
        configs.append(AblationConfig(
            name=f"lengthscale_{ls}",
            kernel_lengthscale=ls
        ))

    # Vary beta (exploration)
    for beta in [0.5, 1.0, 3.0, 5.0]:
        configs.append(AblationConfig(
            name=f"beta_{beta}",
            beta=beta
        ))

    # Vary n_candidates
    for n_cand in [100, 250, 1000]:
        configs.append(AblationConfig(
            name=f"n_candidates_{n_cand}",
            n_candidates=n_cand
        ))

    # Sparse GP comparison
    configs.append(AblationConfig(
        name="sparse_gp",
        use_sparse_gp=True,
        num_inducing=64
    ))
    configs.append(AblationConfig(
        name="sparse_gp_128",
        use_sparse_gp=True,
        num_inducing=128
    ))

    return configs


def run_single_ablation(
    config: AblationConfig,
    model,
    tokenizer,
    harmful_prompts: List[str],
    harmless_prompts: List[str],
    refusal_toks: torch.Tensor,
    n_layers: int,
    hidden_dim: int,
    harmful_completions: Optional[List[str]] = None,
    n_score_tokens: int = 1,
    seed: int = 42
) -> AblationResult:
    """Run a single ablation configuration."""
    from src.discovery.pareto_boundary_discovery import (
        ParetoDiscoveryConfig,
        ParetoGeometryDiscovery,
        MultiObjectiveScorer
    )

    torch.manual_seed(seed)

    # Build config
    pareto_config = ParetoDiscoveryConfig(
        n_init_samples=config.n_init_samples,
        n_pareto_iterations=config.n_pareto_iterations,
        kernel_lengthscale=config.kernel_lengthscale,
        beta=config.beta,
        n_candidates=config.n_candidates,
        use_sparse_gp=config.use_sparse_gp,
        num_inducing=config.num_inducing
    )

    # Build scorer
    scorer = MultiObjectiveScorer(
        model=model,
        tokenizer=tokenizer,
        harmful_prompts=harmful_prompts,
        harmless_prompts=harmless_prompts,
        refusal_toks=refusal_toks,
        harmful_completions=harmful_completions,
        n_score_tokens=n_score_tokens
    )

    # Run discovery
    discovery = ParetoGeometryDiscovery(
        scorer=scorer,
        n_layers=n_layers,
        hidden_dim=hidden_dim,
        config=pareto_config
    )

    start_time = time.time()
    results = discovery.discover()
    wall_time = time.time() - start_time

    # Extract metrics
    spread = compute_spread(results.pareto_scores)
    best_refusal = results.pareto_scores['refusal_score'].min().item()
    best_kl = results.pareto_scores['kl_score'].min().item()

    return AblationResult(
        config=config,
        hypervolume=results.hypervolume,
        pareto_size=len(results.pareto_vectors),
        best_refusal=best_refusal,
        best_kl=best_kl,
        spread=spread,
        n_measurements=results.n_measurements,
        wall_time=wall_time,
        hv_per_measurement=results.hypervolume / results.n_measurements
    )


def run_ablations(
    model,
    tokenizer,
    harmful_prompts: List[str],
    harmless_prompts: List[str],
    refusal_toks: torch.Tensor,
    n_layers: int,
    hidden_dim: int,
    harmful_completions: Optional[List[str]] = None,
    n_score_tokens: int = 1,
    configs: Optional[List[AblationConfig]] = None,
    output_dir: str = "ablation_results"
) -> List[AblationResult]:
    """Run all ablation configurations."""

    if configs is None:
        configs = get_ablation_configs()

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    results = []

    for i, config in enumerate(configs):
        print(f"\n{'='*70}")
        print(f"Ablation {i+1}/{len(configs)}: {config.name}")
        print(f"{'='*70}")

        try:
            result = run_single_ablation(
                config=config,
                model=model,
                tokenizer=tokenizer,
                harmful_prompts=harmful_prompts,
                harmless_prompts=harmless_prompts,
                refusal_toks=refusal_toks,
                n_layers=n_layers,
                hidden_dim=hidden_dim,
                harmful_completions=harmful_completions,
                n_score_tokens=n_score_tokens
            )
            results.append(result)

            print(f"\nResults for {config.name}:")
            print(f"  HV: {result.hypervolume:.4f}")
            print(f"  Pareto size: {result.pareto_size}")
            print(f"  Best refusal: {result.best_refusal:.4f}")
            print(f"  Best KL: {result.best_kl:.4f}")
            print(f"  Spread: {result.spread:.4f}")
            print(f"  Efficiency (HV/meas): {result.hv_per_measurement:.4f}")
            print(f"  Time: {result.wall_time:.1f}s")

        except Exception as e:
            print(f"ERROR in {config.name}: {e}")
            continue

    # Save results
    results_data = [r.to_dict() for r in results]
    with open(f"{output_dir}/ablation_results.json", "w") as f:
        json.dump(results_data, f, indent=2)

    # Print summary table
    print_summary_table(results)

    return results


def print_summary_table(results: List[AblationResult]):
    """Print a summary table of all results."""
    print("\n" + "="*100)
    print("ABLATION SUMMARY")
    print("="*100)

    # Header
    print(f"{'Config':<25} {'HV':>10} {'Pareto':>8} {'Refusal':>10} {'KL':>10} {'Spread':>8} {'Eff':>8} {'Time':>8}")
    print("-"*100)

    # Sort by hypervolume
    sorted_results = sorted(results, key=lambda r: r.hypervolume, reverse=True)

    for r in sorted_results:
        print(f"{r.config.name:<25} {r.hypervolume:>10.2f} {r.pareto_size:>8} "
              f"{r.best_refusal:>10.4f} {r.best_kl:>10.4f} {r.spread:>8.4f} "
              f"{r.hv_per_measurement:>8.4f} {r.wall_time:>7.1f}s")

    print("="*100)

    # Find best configs for each metric
    print("\nBest configurations:")
    print(f"  Highest HV: {sorted_results[0].config.name} ({sorted_results[0].hypervolume:.2f})")

    best_refusal = min(results, key=lambda r: r.best_refusal)
    print(f"  Best refusal: {best_refusal.config.name} ({best_refusal.best_refusal:.4f})")

    best_kl = min(results, key=lambda r: r.best_kl)
    print(f"  Best KL: {best_kl.config.name} ({best_kl.best_kl:.4f})")

    best_efficiency = max(results, key=lambda r: r.hv_per_measurement)
    print(f"  Best efficiency: {best_efficiency.config.name} ({best_efficiency.hv_per_measurement:.4f} HV/meas)")


if __name__ == "__main__":
    # This is meant to be run on Modal - see modal_app.py for the remote function
    print("Run ablations via Modal: modal run modal_app.py::run_pareto_ablations")
