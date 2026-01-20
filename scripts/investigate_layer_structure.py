#!/usr/bin/env python3
"""
Investigate if steering vectors at layer i can predict vectors at layer i+1.

Key questions:
1. Are vectors at adjacent layers linearly related?
2. Do they lie in a shared low-dimensional subspace?
3. Can we predict v[i+1] from v[i] using the model's weights?
4. Could we reduce optimization to a subset of layers?

Usage:
    # From pre-computed vectors
    uv run python scripts/investigate_layer_structure.py --vectors results/pareto_vectors.pt

    # From mean-difference vectors (requires model)
    uv run python scripts/investigate_layer_structure.py --model Qwen/Qwen2.5-0.5B-Instruct \
        --harmful-file data/splits/harmful_train.json --harmless-file data/splits/harmless_train.json
"""

import argparse
from pathlib import Path
from typing import Optional

import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer


def compute_mean_difference_vectors(
    model_name: str,
    harmful_file: Path,
    harmless_file: Path,
    n_prompts: int = 32,
    device: str = "cuda"
) -> torch.Tensor:
    """
    Compute mean-difference refusal vectors at each layer.

    Returns:
        vectors: [n_layers, hidden_dim]
    """
    import json
    from tqdm import tqdm

    # Load model
    print(f"Loading model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map=device
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load prompts
    with open(harmful_file) as f:
        harmful_data = json.load(f)[:n_prompts]
    with open(harmless_file) as f:
        harmless_data = json.load(f)[:n_prompts]

    harmful_prompts = [d["instruction"] for d in harmful_data]
    harmless_prompts = [d["instruction"] for d in harmless_data]

    print(f"Computing activations for {len(harmful_prompts)} harmful, {len(harmless_prompts)} harmless prompts")

    # Get activations
    def get_activations(prompts):
        all_acts = []
        for prompt in tqdm(prompts, desc="Getting activations"):
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)

            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)

            # Get last token activations at each layer
            # hidden_states: tuple of (n_layers+1) x [batch, seq_len, hidden_dim]
            hidden_states = outputs.hidden_states
            acts = torch.stack([h[0, -1, :] for h in hidden_states[1:]])  # [n_layers, hidden_dim]
            all_acts.append(acts.cpu())

        return torch.stack(all_acts).float()  # [n_prompts, n_layers, hidden_dim]

    harmful_acts = get_activations(harmful_prompts)
    harmless_acts = get_activations(harmless_prompts)

    # Compute mean difference
    harmful_mean = harmful_acts.mean(dim=0)  # [n_layers, hidden_dim]
    harmless_mean = harmless_acts.mean(dim=0)  # [n_layers, hidden_dim]

    vectors = harmful_mean - harmless_mean  # [n_layers, hidden_dim]

    # Normalize
    vectors = vectors / (vectors.norm(dim=1, keepdim=True) + 1e-8)

    print(f"Computed vectors shape: {vectors.shape}")
    return vectors


def analyze_layer_relationships(vectors: torch.Tensor, model=None):
    """
    Analyze relationships between vectors at different layers.

    Args:
        vectors: [n_vectors, n_layers, hidden_dim] or [n_layers, hidden_dim]
        model: Optional model to analyze weight-based predictions
    """
    if vectors.dim() == 2:
        vectors = vectors.unsqueeze(0)  # Add batch dim

    n_vectors, n_layers, hidden_dim = vectors.shape
    print(f"\nAnalyzing {n_vectors} vectors across {n_layers} layers (dim={hidden_dim})")

    # === 1. Adjacent layer similarity ===
    print("\n=== 1. Adjacent Layer Similarity ===")

    similarities = []
    for vec in vectors:
        vec_norm = vec / (vec.norm(dim=1, keepdim=True) + 1e-8)

        # Cosine similarity between adjacent layers
        sims = []
        for i in range(n_layers - 1):
            sim = (vec_norm[i] @ vec_norm[i+1]).item()
            sims.append(sim)
        similarities.append(sims)

    similarities = np.array(similarities)  # [n_vectors, n_layers-1]
    mean_sim = similarities.mean(axis=0)

    print(f"Mean cosine similarity between adjacent layers:")
    print(f"  Overall: {mean_sim.mean():.4f} ± {mean_sim.std():.4f}")
    print(f"  Range: [{mean_sim.min():.4f}, {mean_sim.max():.4f}]")

    # Highlight layers with low similarity (potential "jumps")
    low_sim_threshold = 0.3
    low_sim_transitions = np.where(mean_sim < low_sim_threshold)[0]
    if len(low_sim_transitions) > 0:
        print(f"\n  Low similarity transitions (< {low_sim_threshold}):")
        for i in low_sim_transitions:
            print(f"    Layer {i} -> {i+1}: {mean_sim[i]:.4f}")

    # === 2. Cross-layer PCA ===
    print("\n=== 2. Shared Subspace Analysis (PCA) ===")

    # Stack all layer vectors and run PCA
    all_layer_vecs = vectors.reshape(-1, hidden_dim)  # [n_vectors*n_layers, hidden_dim]

    # Compute PCA
    U, S, Vt = torch.svd(all_layer_vecs.float())
    explained_var = (S ** 2) / (S ** 2).sum()
    cumulative_var = torch.cumsum(explained_var, dim=0)

    # How many PCs to explain 90%, 95%, 99%?
    for threshold in [0.90, 0.95, 0.99]:
        n_components = (cumulative_var < threshold).sum().item() + 1
        print(f"  PCs for {threshold*100:.0f}% variance: {n_components} (dim reduction: {hidden_dim} -> {n_components})")

    # === 3. Linear prediction: can we predict v[i+1] from v[i]? ===
    print("\n=== 3. Linear Predictability ===")

    # For each vector, try to predict v[i+1] from v[i] using linear regression
    # v[i+1] ≈ W @ v[i] + b

    # Use first half of layers for training, second half for testing
    mid = n_layers // 2

    for vec_idx, vec in enumerate(vectors):
        if n_vectors > 1:
            print(f"\n  Vector {vec_idx + 1}/{n_vectors}:")

        # Training data: layers 0 to mid
        X_train = vec[:mid].numpy()  # [mid, hidden_dim]
        y_train = vec[1:mid+1].numpy()  # [mid, hidden_dim]

        # Solve for W: y = X @ W.T
        # W.T = (X.T @ X)^-1 @ X.T @ y
        # But hidden_dim >> mid, so use ridge regression
        alpha = 1.0  # Regularization
        W = np.linalg.solve(
            X_train.T @ X_train + alpha * np.eye(hidden_dim),
            X_train.T @ y_train
        ).T  # [hidden_dim, hidden_dim]

        # Test on remaining layers
        X_test = vec[mid:-1].numpy()
        y_test = vec[mid+1:].numpy()
        y_pred = X_test @ W.T

        # R² score
        ss_tot = ((y_test - y_test.mean(axis=0)) ** 2).sum()
        ss_res = ((y_test - y_pred) ** 2).sum()
        r2 = 1 - ss_res / ss_tot

        # Cosine similarity
        y_test_norm = y_test / (np.linalg.norm(y_test, axis=1, keepdims=True) + 1e-8)
        y_pred_norm = y_pred / (np.linalg.norm(y_pred, axis=1, keepdims=True) + 1e-8)
        cos_sims = (y_test_norm * y_pred_norm).sum(axis=1)

        print(f"    R² score: {r2:.4f}")
        print(f"    Mean cosine similarity: {cos_sims.mean():.4f} ± {cos_sims.std():.4f}")

        if cos_sims.mean() > 0.7:
            print(f"    ✓ Strong linear relationship! Could predict layers from neighbors.")
        elif cos_sims.mean() > 0.4:
            print(f"    ~ Moderate linear relationship.")
        else:
            print(f"    ✗ Weak linear relationship. Layers may be independent.")

    # === 4. Subset optimization feasibility ===
    print("\n=== 4. Subset Optimization Analysis ===")

    # If we only optimize every Kth layer, can we interpolate the others?
    for K in [2, 4, 8]:
        if K >= n_layers:
            continue

        # Select every Kth layer
        subset_indices = list(range(0, n_layers, K))
        n_subset = len(subset_indices)

        # For first vector, interpolate missing layers
        vec = vectors[0]
        vec_subset = vec[subset_indices]

        # Linear interpolation
        vec_interp = torch.zeros_like(vec)
        for i in range(len(subset_indices) - 1):
            start_idx = subset_indices[i]
            end_idx = subset_indices[i + 1]

            # Interpolate
            for j in range(start_idx, end_idx + 1):
                alpha = (j - start_idx) / (end_idx - start_idx)
                vec_interp[j] = (1 - alpha) * vec_subset[i] + alpha * vec_subset[i + 1]

        # Handle last segment
        vec_interp[subset_indices[-1]:] = vec_subset[-1]

        # Compare to original
        vec_norm = vec / (vec.norm(dim=1, keepdim=True) + 1e-8)
        interp_norm = vec_interp / (vec_interp.norm(dim=1, keepdim=True) + 1e-8)

        cos_sim = (vec_norm * interp_norm).sum(dim=1).mean().item()

        print(f"  Optimize every {K}th layer ({n_subset}/{n_layers} layers):")
        print(f"    Interpolation quality: {cos_sim:.4f}")
        print(f"    Dimension reduction: {n_layers * hidden_dim} -> {n_subset * hidden_dim} ({100*n_subset/n_layers:.1f}%)")

        if cos_sim > 0.95:
            print(f"    ✓ Excellent! Could safely use {K}× subsampling.")
        elif cos_sim > 0.85:
            print(f"    ~ Good. {K}× subsampling may work with minor quality loss.")
        else:
            print(f"    ✗ Poor interpolation. Don't use {K}× subsampling.")

    # === Visualization ===
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Plot 1: Adjacent layer similarity over layers
    ax = axes[0, 0]
    ax.plot(range(len(mean_sim)), mean_sim, 'o-', linewidth=2)
    ax.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='threshold=0.5')
    ax.set_xlabel("Layer Transition (i -> i+1)")
    ax.set_ylabel("Cosine Similarity")
    ax.set_title("Adjacent Layer Similarity")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: PCA explained variance
    ax = axes[0, 1]
    ax.plot(range(1, min(51, len(explained_var)+1)), explained_var[:50].numpy(), 'o-')
    ax.set_xlabel("Principal Component")
    ax.set_ylabel("Explained Variance Ratio")
    ax.set_title("PCA: Shared Subspace Across Layers")
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)

    # Plot 3: Layer similarity heatmap
    ax = axes[1, 0]
    vec = vectors[0]
    vec_norm = vec / (vec.norm(dim=1, keepdim=True) + 1e-8)
    sim_matrix = (vec_norm @ vec_norm.T).numpy()
    im = ax.imshow(sim_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    ax.set_xlabel("Layer j")
    ax.set_ylabel("Layer i")
    ax.set_title("Layer-to-Layer Similarity Heatmap")
    plt.colorbar(im, ax=ax, label="Cosine Similarity")

    # Plot 4: Per-layer norms
    ax = axes[1, 1]
    for i, vec in enumerate(vectors[:5]):  # Max 5 vectors
        norms = vec.norm(dim=1).numpy()
        ax.plot(range(n_layers), norms, 'o-', label=f'Vector {i+1}', alpha=0.7)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Norm")
    ax.set_title("Per-Layer Vector Norms")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    output_path = Path("layer_structure_analysis.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved visualization: {output_path}")
    plt.close()

    return {
        "mean_adjacent_similarity": mean_sim.mean(),
        "pca_dims_90pct": (cumulative_var < 0.90).sum().item() + 1,
        "pca_dims_95pct": (cumulative_var < 0.95).sum().item() + 1,
        "pca_dims_99pct": (cumulative_var < 0.99).sum().item() + 1,
    }


def main():
    parser = argparse.ArgumentParser(description="Investigate layer structure of steering vectors")
    parser.add_argument("--vectors", type=Path, help="Path to pre-computed vectors (.pt file)")
    parser.add_argument("--model", type=str, help="Model name to compute mean-diff vectors")
    parser.add_argument("--harmful-file", type=Path, help="Harmful prompts JSON")
    parser.add_argument("--harmless-file", type=Path, help="Harmless prompts JSON")
    parser.add_argument("--n-prompts", type=int, default=32, help="Number of prompts to use")
    parser.add_argument("--device", type=str, default="cuda", help="Device")

    args = parser.parse_args()

    if args.vectors:
        # Load pre-computed vectors
        print(f"Loading vectors from: {args.vectors}")
        vectors = torch.load(args.vectors)
        print(f"Loaded shape: {vectors.shape}")
        model = None
    elif args.model:
        # Compute mean-difference vectors
        if not args.harmful_file or not args.harmless_file:
            raise ValueError("--harmful-file and --harmless-file required when using --model")

        vectors = compute_mean_difference_vectors(
            args.model,
            args.harmful_file,
            args.harmless_file,
            args.n_prompts,
            args.device
        )

        # Optionally load model for weight-based analysis
        model = None  # Could load here if needed
    else:
        raise ValueError("Must provide either --vectors or --model")

    # Run analysis
    results = analyze_layer_relationships(vectors, model)

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Adjacent layer similarity: {results['mean_adjacent_similarity']:.4f}")
    print(f"Intrinsic dimensionality:")
    print(f"  90% variance: {results['pca_dims_90pct']} dimensions")
    print(f"  95% variance: {results['pca_dims_95pct']} dimensions")
    print(f"  99% variance: {results['pca_dims_99pct']} dimensions")
    print("\nSee layer_structure_analysis.png for visualizations.")


if __name__ == "__main__":
    main()
