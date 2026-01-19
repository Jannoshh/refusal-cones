#!/usr/bin/env python3
"""
E2.1: Dimensionality by Layer

Measure intrinsic dimension of refusal geometry at each layer.

Hypothesis: Inverted-U pattern
- Early layers: Low-dimensional (refusal not yet formed)
- Middle layers: High-dimensional (refusal actively computed)
- Late layers: Low-dimensional (refusal decision consolidated)

Metrics:
- Intrinsic dimension at each layer
- Cone quality (ASR of sampled directions) by layer
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List
from tqdm import tqdm

from shared.model_loading import load_model_and_tokenizer, get_model_config
from shared.data_loading import load_harmful_data
from shared.evaluation import evaluate_asr, StrongRejectJudge
from shared.utils import (
    set_seed, get_output_dir, save_results,
    print_experiment_header, Timer
)

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


def estimate_layer_dimensionality(
    model,
    tokenizer,
    layer_idx: int,
    harmful_prompts: List[Dict],
    judge: StrongRejectJudge,
    n_samples: int = 50,
    asr_threshold: float = 0.3,
) -> Dict:
    """
    Estimate intrinsic dimensionality of refusal at a specific layer.

    Method:
    1. Sample random directions at this layer
    2. Measure ASR for each
    3. Use PCA on effective directions to estimate dimension

    Args:
        layer_idx: Layer to analyze
        n_samples: Number of random directions to try
        asr_threshold: Min ASR to consider direction effective

    Returns:
        Dict with dimension estimate and samples
    """
    hidden_dim = model.config.hidden_size
    n_layers = len(model.model.layers)

    effective_vectors = []
    asr_values = []

    print(f"  Layer {layer_idx}: Sampling {n_samples} directions...")

    for _ in tqdm(range(n_samples), desc=f"Layer {layer_idx}", leave=False):
        # Create vector for single layer
        v = torch.zeros(n_layers, hidden_dim, device=model.device)
        v[layer_idx] = torch.randn(hidden_dim, device=model.device)
        v[layer_idx] = v[layer_idx] / v[layer_idx].norm()

        # Evaluate ASR
        result = evaluate_asr(
            model, tokenizer,
            harmful_prompts[:20],  # Subset for speed
            judge,
            ablation_vectors=v,
            ablation_layers=[layer_idx],
            max_new_tokens=100
        )

        asr_values.append(result["asr"])

        if result["asr"] > asr_threshold:
            effective_vectors.append(v[layer_idx].cpu())

    # Estimate dimension using PCA
    intrinsic_dim = 1  # Default
    if len(effective_vectors) >= 10:
        from sklearn.decomposition import PCA

        vectors_np = torch.stack(effective_vectors).numpy()
        pca = PCA()
        pca.fit(vectors_np)

        # Dimension = number of components for 95% variance
        cumsum = np.cumsum(pca.explained_variance_ratio_)
        intrinsic_dim = int(np.argmax(cumsum > 0.95) + 1)

    return {
        "layer_idx": layer_idx,
        "intrinsic_dim": intrinsic_dim,
        "n_effective": len(effective_vectors),
        "mean_asr": np.mean(asr_values),
        "max_asr": np.max(asr_values),
        "asr_values": asr_values,
    }


def main():
    parser = argparse.ArgumentParser(description="E2.1: Dimensionality by Layer")
    parser.add_argument("--model", default="gemma-2-2b", help="Model to use")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_samples", type=int, default=50, help="Samples per layer")
    parser.add_argument("--output_dir", default="results")
    args = parser.parse_args()

    set_seed(args.seed)
    print_experiment_header("E2.1: Dimensionality by Layer", args.model)

    # Load model
    model, tokenizer = load_model_and_tokenizer(args.model)
    model_config = get_model_config(args.model)
    n_layers = model_config["n_layers"]

    # Load data
    harmful_prompts = load_harmful_data(split="val", max_samples=50)

    # Initialize judge
    judge = StrongRejectJudge()

    # Analyze each layer
    results = {
        "layers": [],
        "model": args.model,
        "n_layers": n_layers,
    }

    for layer_idx in range(n_layers):
        with Timer(f"Layer {layer_idx}"):
            layer_result = estimate_layer_dimensionality(
                model, tokenizer, layer_idx, harmful_prompts, judge,
                n_samples=args.n_samples
            )
            results["layers"].append(layer_result)

            print(f"  Intrinsic dim: {layer_result['intrinsic_dim']}")
            print(f"  Mean ASR: {layer_result['mean_asr']:.4f}")

    # Summary statistics
    dims = [r["intrinsic_dim"] for r in results["layers"]]
    asrs = [r["mean_asr"] for r in results["layers"]]

    results["summary"] = {
        "max_dim": max(dims),
        "max_dim_layer": dims.index(max(dims)),
        "mean_dim": np.mean(dims),
        "best_asr": max(asrs),
        "best_asr_layer": asrs.index(max(asrs)),
    }

    # Plot
    output_dir = get_output_dir("e2_per_layer_dim", args.model, args.output_dir)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Dimension by layer
    ax1.plot(range(n_layers), dims, 'b-o', markersize=4)
    ax1.set_xlabel("Layer")
    ax1.set_ylabel("Intrinsic Dimension")
    ax1.set_title(f"Refusal Dimensionality by Layer ({args.model})")
    ax1.axvline(x=n_layers//2, color='r', linestyle='--', alpha=0.5, label='Middle')
    ax1.legend()

    # ASR by layer
    ax2.plot(range(n_layers), asrs, 'g-o', markersize=4)
    ax2.set_xlabel("Layer")
    ax2.set_ylabel("Mean ASR")
    ax2.set_title(f"Single-Layer ASR ({args.model})")

    plt.tight_layout()
    plt.savefig(output_dir / "dimensionality_by_layer.png", dpi=150)
    print(f"Plot saved to {output_dir}/dimensionality_by_layer.png")

    # Save results
    save_results(results, output_dir)

    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Max dimension: {results['summary']['max_dim']} at layer {results['summary']['max_dim_layer']}")
    print(f"Best ASR: {results['summary']['best_asr']:.4f} at layer {results['summary']['best_asr_layer']}")

    # Check hypothesis
    middle = n_layers // 2
    early_dims = np.mean(dims[:n_layers//4])
    middle_dims = np.mean(dims[n_layers//4:3*n_layers//4])
    late_dims = np.mean(dims[3*n_layers//4:])

    print(f"\nDimensionality pattern:")
    print(f"  Early (layers 0-{n_layers//4}): {early_dims:.1f}")
    print(f"  Middle (layers {n_layers//4}-{3*n_layers//4}): {middle_dims:.1f}")
    print(f"  Late (layers {3*n_layers//4}-{n_layers}): {late_dims:.1f}")

    if middle_dims > early_dims and middle_dims > late_dims:
        print("\n✓ Inverted-U hypothesis SUPPORTED")
    else:
        print("\n✗ Inverted-U hypothesis NOT supported")


if __name__ == "__main__":
    main()
