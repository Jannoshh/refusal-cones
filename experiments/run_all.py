#!/usr/bin/env python3
"""
Main runner for all experiments.

Usage:
    python run_all.py --experiments e1 e2 --model gemma-2-2b
    python run_all.py --all --model gemma-2-9b
"""

import argparse
import subprocess
import sys
from pathlib import Path
from datetime import datetime


EXPERIMENTS = {
    "e1.1": {
        "name": "Discovery Efficiency",
        "script": "e1_discovery/run_efficiency.py",
        "description": "Compare gradient+GP vs pure GP vs random search",
        "gpu_hours": 10,
    },
    "e1.2": {
        "name": "Discovery Quality",
        "script": "e1_discovery/run_quality.py",
        "description": "Compare DIM vs RDO vs adaptive discovery",
        "gpu_hours": 5,
    },
    "e2.1": {
        "name": "Dimensionality by Layer",
        "script": "e2_per_layer/run_dimensionality.py",
        "description": "Measure intrinsic dimension at each layer",
        "gpu_hours": 30,
    },
    "e2.2": {
        "name": "Per-Layer ASR",
        "script": "e2_per_layer/run_layer_asr.py",
        "description": "Train single-layer vectors, measure ASR",
        "gpu_hours": 40,
    },
    "e3.1": {
        "name": "Unified vs Separate",
        "script": "e3_unified_affine/run_comparison.py",
        "description": "Compare unified affine vs separate operations",
        "gpu_hours": 20,
    },
    "e4.1": {
        "name": "SFT → RL Pipeline",
        "script": "e4_rl_optimization/run_sft_rl_pipeline.py",
        "description": "Train SFT then RL, measure gains",
        "gpu_hours": 30,
    },
    "e4.3": {
        "name": "Ceiling Analysis",
        "script": "e4_rl_optimization/run_ceiling_analysis.py",
        "description": "Diagnose ceiling (steering vs capability vs judge)",
        "gpu_hours": 40,
    },
    "e5.1": {
        "name": "Component Ablation",
        "script": "e5_ablations/run_component_ablation.py",
        "description": "Measure contribution of each component",
        "gpu_hours": 20,
    },
}


def run_experiment(exp_id: str, model: str, extra_args: list = None):
    """Run a single experiment."""
    exp = EXPERIMENTS[exp_id]
    script_path = Path(__file__).parent / exp["script"]

    print(f"\n{'='*70}")
    print(f"Running: {exp_id} - {exp['name']}")
    print(f"Script: {script_path}")
    print(f"Model: {model}")
    print(f"{'='*70}\n")

    cmd = [sys.executable, str(script_path), "--model", model]
    if extra_args:
        cmd.extend(extra_args)

    result = subprocess.run(cmd, cwd=Path(__file__).parent)

    if result.returncode != 0:
        print(f"\n⚠ Experiment {exp_id} failed with code {result.returncode}")
        return False

    print(f"\n✓ Experiment {exp_id} completed successfully")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Run experiments for 'Beyond Concept Cones' paper"
    )
    parser.add_argument(
        "--experiments", "-e", nargs="+",
        choices=list(EXPERIMENTS.keys()) + ["all", "e1", "e2", "e3", "e4", "e5"],
        default=["all"],
        help="Experiments to run"
    )
    parser.add_argument("--model", "-m", default="gemma-2-2b", help="Model to use")
    parser.add_argument("--list", "-l", action="store_true", help="List all experiments")
    parser.add_argument("--dry-run", action="store_true", help="Show what would run")
    args = parser.parse_args()

    if args.list:
        print("\nAvailable experiments:")
        print("-" * 70)
        total_hours = 0
        for exp_id, exp in EXPERIMENTS.items():
            print(f"  {exp_id:<8} {exp['name']:<25} ~{exp['gpu_hours']}h")
            print(f"           {exp['description']}")
            total_hours += exp['gpu_hours']
        print("-" * 70)
        print(f"Total estimated GPU hours: ~{total_hours}h")
        return

    # Expand experiment groups
    to_run = []
    for e in args.experiments:
        if e == "all":
            to_run = list(EXPERIMENTS.keys())
            break
        elif e == "e1":
            to_run.extend(["e1.1", "e1.2"])
        elif e == "e2":
            to_run.extend(["e2.1", "e2.2"])
        elif e == "e3":
            to_run.extend(["e3.1"])
        elif e == "e4":
            to_run.extend(["e4.1", "e4.3"])
        elif e == "e5":
            to_run.extend(["e5.1"])
        else:
            to_run.append(e)

    # Remove duplicates, preserve order
    to_run = list(dict.fromkeys(to_run))

    # Estimate time
    total_hours = sum(EXPERIMENTS[e]["gpu_hours"] for e in to_run)

    print(f"\n{'='*70}")
    print("EXPERIMENT RUNNER")
    print(f"{'='*70}")
    print(f"Model: {args.model}")
    print(f"Experiments to run: {', '.join(to_run)}")
    print(f"Estimated GPU hours: ~{total_hours}h")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    if args.dry_run:
        print("\n[DRY RUN] Would run:")
        for exp_id in to_run:
            exp = EXPERIMENTS[exp_id]
            print(f"  - {exp_id}: {exp['name']}")
        return

    # Run experiments
    results = {}
    for exp_id in to_run:
        success = run_experiment(exp_id, args.model)
        results[exp_id] = "✓" if success else "✗"

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    for exp_id, status in results.items():
        print(f"  {status} {exp_id}: {EXPERIMENTS[exp_id]['name']}")

    n_success = sum(1 for s in results.values() if s == "✓")
    print(f"\nCompleted: {n_success}/{len(results)}")
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
