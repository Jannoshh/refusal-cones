#!/usr/bin/env python3
"""
Visualize single-vector discovery results without rerunning.

Usage:
    python scripts/visualize_results.py results/single_vector_20260120_170834/
    python scripts/visualize_results.py  # uses most recent
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def load_results(run_dir: Path) -> dict:
    """Load results from a run directory."""
    vectors_path = run_dir / "vectors.pt"
    results_path = run_dir / "results.json"

    data = torch.load(vectors_path, map_location="cpu")
    with open(results_path) as f:
        summary = json.load(f)

    return {
        "best_vector": data["best_vector"],
        "pareto_vectors": data["pareto_vectors"],
        "V_observed": data["V_observed"],
        "scores": data["scores_observed"],
        "summary": summary,
    }


def find_latest_run(results_dir: Path, prefix: str = "single_vector") -> Path:
    """Find the most recent run directory."""
    runs = sorted(results_dir.glob(f"{prefix}_*"))
    if not runs:
        raise FileNotFoundError(f"No {prefix}_* directories found in {results_dir}")
    return runs[-1]


def compute_pareto_mask(refusal: np.ndarray, kl: np.ndarray) -> np.ndarray:
    """Compute which points are Pareto-optimal (minimizing both)."""
    n = len(refusal)
    is_pareto = np.ones(n, dtype=bool)

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            # j dominates i if j <= i on both and < on at least one
            if (refusal[j] <= refusal[i] and kl[j] <= kl[i] and
                (refusal[j] < refusal[i] or kl[j] < kl[i])):
                is_pareto[i] = False
                break

    return is_pareto


def visualize_full(data: dict, output_path: Path, n_layers: int = 28):
    """Create comprehensive visualization of the discovery space."""
    V = data["V_observed"].numpy()
    refusal = data["scores"]["refusal_score"].numpy()
    kl = data["scores"]["kl_score"].numpy()

    n_total = len(refusal)

    # Compute Pareto mask
    pareto_mask = compute_pareto_mask(refusal, kl)

    # Identify r_i points (first n_layers)
    r_i_mask = np.zeros(n_total, dtype=bool)
    r_i_mask[:min(n_layers, n_total)] = True

    # Create figure with multiple subplots
    fig = plt.figure(figsize=(16, 12))

    # 1. Objective space - full view
    ax1 = fig.add_subplot(2, 3, 1)
    ax1.scatter(refusal[~pareto_mask & ~r_i_mask], kl[~pareto_mask & ~r_i_mask],
                c="lightgray", alpha=0.5, s=30, label="Explored")
    ax1.scatter(refusal[r_i_mask], kl[r_i_mask],
                c="blue", s=60, marker="s", alpha=0.7, label="r_i (per-layer)")
    ax1.scatter(refusal[pareto_mask], kl[pareto_mask],
                c="red", s=120, marker="*", label="Pareto", zorder=5)

    # Connect Pareto frontier
    pareto_pairs = sorted(zip(refusal[pareto_mask], kl[pareto_mask]))
    if pareto_pairs:
        pr, pk = zip(*pareto_pairs)
        ax1.plot(pr, pk, "r--", alpha=0.7, linewidth=2)

    ax1.set_xlabel("Refusal Score (lower = better)")
    ax1.set_ylabel("KL Score (lower = better)")
    ax1.set_title("Objective Space")
    ax1.legend(loc="upper right")
    ax1.grid(True, alpha=0.3)

    # 2. Objective space - zoomed to interesting region
    ax2 = fig.add_subplot(2, 3, 2)
    # Focus on the good region (low refusal, low KL)
    good_mask = (refusal < np.percentile(refusal, 50)) | pareto_mask
    ax2.scatter(refusal[good_mask & ~pareto_mask], kl[good_mask & ~pareto_mask],
                c=np.arange(n_total)[good_mask & ~pareto_mask], cmap="viridis",
                alpha=0.7, s=40)
    ax2.scatter(refusal[pareto_mask], kl[pareto_mask],
                c="red", s=120, marker="*", edgecolors="black", linewidths=1, zorder=5)
    if pareto_pairs:
        ax2.plot(pr, pk, "r--", alpha=0.7, linewidth=2)
    ax2.set_xlabel("Refusal Score")
    ax2.set_ylabel("KL Score")
    ax2.set_title("Good Region (colored by iteration)")
    ax2.grid(True, alpha=0.3)

    # 3. Refusal score by layer (for r_i points)
    ax3 = fig.add_subplot(2, 3, 3)
    n_r_i = min(n_layers, n_total)
    layers = np.arange(n_r_i)
    ax3.bar(layers, refusal[:n_r_i], color="steelblue", alpha=0.7)
    ax3.axhline(y=refusal.min(), color="red", linestyle="--", label=f"Best: {refusal.min():.2f}")
    ax3.set_xlabel("Layer")
    ax3.set_ylabel("Refusal Score")
    ax3.set_title("Per-Layer r_i Performance")
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis="y")

    # 4. PCA of vector space
    ax4 = fig.add_subplot(2, 3, 4)
    pca = PCA(n_components=2)
    V_pca = pca.fit_transform(V)
    scatter = ax4.scatter(V_pca[:, 0], V_pca[:, 1], c=refusal, cmap="viridis",
                          alpha=0.7, s=40)
    ax4.scatter(V_pca[pareto_mask, 0], V_pca[pareto_mask, 1],
                c="red", s=120, marker="*", edgecolors="black", linewidths=1, zorder=5)
    plt.colorbar(scatter, ax=ax4, label="Refusal Score")
    ax4.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} var)")
    ax4.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} var)")
    ax4.set_title("PCA of Vector Space")

    # 5. t-SNE of vector space
    ax5 = fig.add_subplot(2, 3, 5)
    perplexity = min(30, n_total - 1)
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    V_tsne = tsne.fit_transform(V)
    scatter = ax5.scatter(V_tsne[:, 0], V_tsne[:, 1], c=refusal, cmap="viridis",
                          alpha=0.7, s=40)
    ax5.scatter(V_tsne[pareto_mask, 0], V_tsne[pareto_mask, 1],
                c="red", s=120, marker="*", edgecolors="black", linewidths=1, zorder=5)
    plt.colorbar(scatter, ax=ax5, label="Refusal Score")
    ax5.set_xlabel("t-SNE 1")
    ax5.set_ylabel("t-SNE 2")
    ax5.set_title("t-SNE of Vector Space")

    # 6. Discovery progress
    ax6 = fig.add_subplot(2, 3, 6)
    best_so_far = np.minimum.accumulate(refusal)
    ax6.plot(np.arange(n_total), refusal, "o-", alpha=0.3, markersize=3, label="Each sample")
    ax6.plot(np.arange(n_total), best_so_far, "r-", linewidth=2, label="Best so far")
    ax6.axvline(x=n_layers, color="blue", linestyle="--", alpha=0.5, label="End of r_i init")
    ax6.set_xlabel("Sample #")
    ax6.set_ylabel("Refusal Score")
    ax6.set_title("Discovery Progress")
    ax6.legend()
    ax6.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize single-vector discovery results")
    parser.add_argument("run_dir", nargs="?", help="Path to run directory (default: most recent)")
    parser.add_argument("--output", "-o", help="Output path (default: run_dir/full_visualization.png)")
    args = parser.parse_args()

    results_dir = Path("results")

    if args.run_dir:
        run_dir = Path(args.run_dir)
    else:
        run_dir = find_latest_run(results_dir)
        print(f"Using most recent run: {run_dir}")

    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    data = load_results(run_dir)

    output_path = Path(args.output) if args.output else run_dir / "full_visualization.png"
    visualize_full(data, output_path)

    # Also show basic stats
    summary = data["summary"]
    print(f"\nRun summary:")
    print(f"  Measurements: {summary['n_measurements']}")
    print(f"  Best refusal: {summary['best_scores']['refusal_score']:.4f}")
    print(f"  Best KL: {summary['best_scores']['kl_score']:.4f}")
    print(f"  Pareto vectors: {summary['n_pareto']}")
    print(f"  Hypervolume: {summary['hypervolume']:.4f}")


if __name__ == "__main__":
    main()
