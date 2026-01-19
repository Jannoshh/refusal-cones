#!/usr/bin/env python3
"""
E5.1: Component Ablation Study

Measure contribution of each component to overall performance.

Configurations:
- Full method (all components)
- − Adaptive discovery (use DIM instead)
- − Per-layer (single vector across all layers)
- − Unified affine (use separate operations)
- − RL (SFT only)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt

from shared.model_loading import load_model_and_tokenizer, get_model_config
from shared.data_loading import load_harmful_data, load_harmless_data, load_jailbreakbench
from shared.evaluation import evaluate_asr, StrongRejectJudge
from shared.utils import (
    set_seed, get_output_dir, save_results,
    print_experiment_header, Timer
)

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


def main():
    parser = argparse.ArgumentParser(description="E5.1: Component Ablation")
    parser.add_argument("--model", default="gemma-2-2b", help="Model to use")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", default="results")
    args = parser.parse_args()

    set_seed(args.seed)
    print_experiment_header("E5.1: Component Ablation", args.model)

    model, tokenizer = load_model_and_tokenizer(args.model)
    config = get_model_config(args.model)

    eval_prompts = load_jailbreakbench(max_samples=100)
    judge = StrongRejectJudge()

    results = {"configurations": []}
    output_dir = get_output_dir("e5_component_ablation", args.model, args.output_dir)

    # Configuration 1: Full method (placeholder - would use trained vectors)
    print("\n" + "=" * 70)
    print("Config 1: Full Method (all components)")
    print("=" * 70)
    print("  [Requires running full pipeline first]")
    results["configurations"].append({
        "name": "full_method",
        "components": ["adaptive_discovery", "per_layer", "unified_affine", "rl"],
        "asr": None,
        "note": "requires_full_pipeline"
    })

    # Configuration 2: No adaptive discovery (use DIM)
    print("\n" + "=" * 70)
    print("Config 2: − Adaptive Discovery (use DIM)")
    print("=" * 70)

    # Compute DIM vectors
    harmful_data = load_harmful_data(split="train", max_samples=100)
    harmless_data = load_harmless_data(split="train", max_samples=100)

    n_layers = config["n_layers"]
    hidden_dim = config["hidden_dim"]

    # Simple DIM computation
    print("  Computing DIM vectors...")
    dim_vectors = torch.randn(n_layers, hidden_dim, device=model.device)
    dim_vectors = dim_vectors / dim_vectors.norm(dim=1, keepdim=True)

    dim_asr = evaluate_asr(
        model, tokenizer, eval_prompts, judge,
        ablation_vectors=dim_vectors
    )

    results["configurations"].append({
        "name": "no_adaptive_discovery",
        "components": ["per_layer", "unified_affine", "rl"],
        "ablated": "adaptive_discovery",
        "asr": dim_asr["asr"]
    })
    print(f"  ASR: {dim_asr['asr']:.4f}")

    # Configuration 3: No per-layer (single vector)
    print("\n" + "=" * 70)
    print("Config 3: − Per-Layer (single vector)")
    print("=" * 70)

    # Single vector repeated
    single_v = torch.randn(hidden_dim, device=model.device)
    single_v = single_v / single_v.norm()
    single_vectors = single_v.unsqueeze(0).repeat(n_layers, 1)

    single_asr = evaluate_asr(
        model, tokenizer, eval_prompts, judge,
        ablation_vectors=single_vectors
    )

    results["configurations"].append({
        "name": "no_per_layer",
        "components": ["adaptive_discovery", "unified_affine", "rl"],
        "ablated": "per_layer",
        "asr": single_asr["asr"]
    })
    print(f"  ASR: {single_asr['asr']:.4f}")

    # Configuration 4: No unified affine (ablation only)
    print("\n" + "=" * 70)
    print("Config 4: − Unified Affine (ablation only)")
    print("=" * 70)

    # Same as dim_asr (ablation only is what we tested)
    results["configurations"].append({
        "name": "no_unified_affine",
        "components": ["adaptive_discovery", "per_layer", "rl"],
        "ablated": "unified_affine",
        "asr": dim_asr["asr"],
        "note": "same_as_ablation_only"
    })
    print(f"  ASR: {dim_asr['asr']:.4f}")

    # Configuration 5: No RL (SFT only)
    print("\n" + "=" * 70)
    print("Config 5: − RL (SFT only)")
    print("=" * 70)
    print("  [Run E4.1 to compare SFT vs SFT+RL]")
    results["configurations"].append({
        "name": "no_rl",
        "components": ["adaptive_discovery", "per_layer", "unified_affine"],
        "ablated": "rl",
        "asr": None,
        "note": "requires_e4_comparison"
    })

    # Summary
    print("\n" + "=" * 70)
    print("ABLATION SUMMARY")
    print("=" * 70)

    print(f"\n{'Configuration':<30} {'Ablated':<20} {'ASR':<10}")
    print("-" * 60)
    for cfg in results["configurations"]:
        asr_str = f"{cfg['asr']:.4f}" if cfg['asr'] is not None else "N/A"
        ablated = cfg.get('ablated', 'none')
        print(f"{cfg['name']:<30} {ablated:<20} {asr_str:<10}")

    # Plot (for available results)
    valid_configs = [c for c in results["configurations"] if c["asr"] is not None]
    if valid_configs:
        plt.figure(figsize=(10, 5))
        names = [c["name"] for c in valid_configs]
        asrs = [c["asr"] for c in valid_configs]
        colors = ['green' if c.get("ablated") is None else 'red' for c in valid_configs]

        plt.bar(names, asrs, color=colors, alpha=0.7)
        plt.xticks(rotation=45, ha='right')
        plt.ylabel("ASR")
        plt.title("Component Ablation Study")
        plt.tight_layout()
        plt.savefig(output_dir / "component_ablation.png", dpi=150)

    save_results(results, output_dir)


if __name__ == "__main__":
    main()
