#!/usr/bin/env python3
"""
Analyze per-layer structure of Pareto vectors.

Usage:
    uv run python scripts/analyze_layers.py results/pareto_ace_20260119/
    uv run python scripts/analyze_layers.py results/high_budget_20260119_204649/
"""

import argparse
import json
from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt


def find_vectors_file(results_dir: Path) -> Path:
    """Find the vectors .pt file in the results directory."""
    pt_files = list(results_dir.glob("*vectors*.pt"))
    if not pt_files:
        raise FileNotFoundError(f"No vectors .pt file found in {results_dir}")
    return pt_files[0]


def find_json_file(results_dir: Path) -> Path:
    """Find the results .json file in the results directory."""
    json_files = list(results_dir.glob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"No .json file found in {results_dir}")
    return json_files[0]


def analyze_layers(results_dir: Path, output_path: Path = None):
    """
    Analyze per-layer structure of Pareto/discovery vectors.

    Creates a 3-panel visualization:
    1. Per-layer norms for each vector
    2. Mean norm with std across vectors
    3. Layer direction similarity heatmap
    """
    results_dir = Path(results_dir)

    # Load vectors
    vectors_path = find_vectors_file(results_dir)
    vectors = torch.load(vectors_path)
    print(f"Loaded vectors from: {vectors_path}")
    print(f"Shape: {vectors.shape}")

    if vectors.dim() == 2:
        # Single vector [n_layers, hidden_dim] -> add batch dim
        vectors = vectors.unsqueeze(0)

    n_vectors, n_layers, hidden_dim = vectors.shape

    # Try to load scores
    try:
        json_path = find_json_file(results_dir)
        with open(json_path) as f:
            results = json.load(f)

        # Handle different JSON formats
        if "pareto_scores" in results:
            refusal_scores = results["pareto_scores"]["refusal_score"]
            kl_scores = results["pareto_scores"]["kl_score"]
        elif "scores_observed" in results:
            refusal_scores = results["scores_observed"]["refusal_score"][:n_vectors]
            kl_scores = results["scores_observed"]["kl_score"][:n_vectors]
        else:
            refusal_scores = None
            kl_scores = None
    except Exception as e:
        print(f"Could not load scores: {e}")
        refusal_scores = None
        kl_scores = None

    # Compute per-layer norms
    layer_norms = vectors.norm(dim=2)  # [n_vectors, n_layers]

    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Panel 1: Per-layer norms for each vector
    ax1 = axes[0]
    for i in range(min(n_vectors, 10)):  # Limit to 10 vectors for readability
        if refusal_scores is not None and kl_scores is not None:
            label = f"v{i+1} (r={refusal_scores[i]:.1f}, kl={kl_scores[i]:.2f})"
        else:
            label = f"v{i+1}"
        ax1.plot(range(n_layers), layer_norms[i].numpy(), marker='o', markersize=3, label=label)
    ax1.set_xlabel("Layer")
    ax1.set_ylabel("Norm")
    ax1.set_title(f"Per-Layer Norms ({n_vectors} vectors)")
    ax1.legend(fontsize=8, loc='upper right')
    ax1.grid(True, alpha=0.3)

    # Panel 2: Mean norm with std
    ax2 = axes[1]
    mean_norms = layer_norms.mean(dim=0).numpy()
    std_norms = layer_norms.std(dim=0).numpy()
    layers = np.arange(n_layers)
    ax2.fill_between(layers, mean_norms - std_norms, mean_norms + std_norms, alpha=0.3)
    ax2.plot(layers, mean_norms, 'b-', linewidth=2, label='Mean')
    ax2.set_xlabel("Layer")
    ax2.set_ylabel("Norm")
    ax2.set_title("Mean Per-Layer Norm (± std)")
    ax2.grid(True, alpha=0.3)

    # Annotate top layers
    top_layers = np.argsort(mean_norms)[-5:][::-1]
    for l in top_layers:
        ax2.annotate(f"L{l}", (l, mean_norms[l]), textcoords="offset points",
                     xytext=(0, 10), ha='center', fontsize=8)

    # Panel 3: Layer direction similarity heatmap
    ax3 = axes[2]

    # Compute cosine similarity between layer directions across vectors
    layer_similarity = torch.zeros(n_layers, n_layers)
    for i in range(n_layers):
        for j in range(n_layers):
            vi = vectors[:, i, :]  # [n_vectors, hidden_dim]
            vj = vectors[:, j, :]
            vi_norm = vi / (vi.norm(dim=1, keepdim=True) + 1e-8)
            vj_norm = vj / (vj.norm(dim=1, keepdim=True) + 1e-8)
            cos_sim = (vi_norm * vj_norm).sum(dim=1).mean()
            layer_similarity[i, j] = cos_sim

    im = ax3.imshow(layer_similarity.numpy(), cmap='RdBu_r', vmin=-1, vmax=1)
    ax3.set_xlabel("Layer")
    ax3.set_ylabel("Layer")
    ax3.set_title("Layer Direction Similarity\n(should show diagonal band if smooth)")
    plt.colorbar(im, ax=ax3, label="Cosine Similarity")

    plt.tight_layout()

    # Save
    if output_path is None:
        output_path = results_dir / "layer_analysis.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()

    # Print summary
    print(f"\nLayer Analysis Summary:")
    print(f"  Vectors: {n_vectors}")
    print(f"  Layers: {n_layers}")
    print(f"  Hidden dim: {hidden_dim}")

    print(f"\nTop 5 layers by mean norm:")
    for i, l in enumerate(top_layers):
        print(f"  {i+1}. Layer {l}: {mean_norms[l]:.4f}")

    print(f"\nBottom 5 layers by mean norm:")
    bottom_layers = np.argsort(mean_norms)[:5]
    for i, l in enumerate(bottom_layers):
        print(f"  {i+1}. Layer {l}: {mean_norms[l]:.4f}")

    # Check for layer smoothness
    off_diag_sim = []
    for i in range(n_layers - 1):
        off_diag_sim.append(layer_similarity[i, i+1].item())
    mean_adjacent_sim = np.mean(off_diag_sim)

    print(f"\nLayer smoothness:")
    print(f"  Mean adjacent-layer similarity: {mean_adjacent_sim:.4f}")
    if mean_adjacent_sim > 0.5:
        print("  ✓ Good layer smoothness (adjacent layers correlated)")
    elif mean_adjacent_sim > 0.2:
        print("  ~ Moderate layer smoothness")
    else:
        print("  ✗ Poor layer smoothness (directions nearly independent per layer)")
        print("    Consider using gp_type='structured' with higher layer_lengthscale")

    return {
        "n_vectors": n_vectors,
        "n_layers": n_layers,
        "hidden_dim": hidden_dim,
        "mean_norms": mean_norms.tolist(),
        "top_layers": top_layers.tolist(),
        "mean_adjacent_similarity": mean_adjacent_sim,
    }


def main():
    parser = argparse.ArgumentParser(description="Analyze per-layer structure of vectors")
    parser.add_argument("results_dir", type=Path, help="Path to results directory")
    parser.add_argument("-o", "--output", type=Path, help="Output path for plot")

    args = parser.parse_args()

    analyze_layers(args.results_dir, args.output)


if __name__ == "__main__":
    main()
