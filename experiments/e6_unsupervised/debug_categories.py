"""
Check if different harm categories have different residual structure.
"""

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from pathlib import Path

from experiments.e6_unsupervised.run_on_qwen import (
    HARMFUL_PROMPTS, HARMLESS_PROMPTS, format_chat_prompt, get_activations
)

# Categorize harmful prompts
CATEGORIES = {
    'violence': [0, 1, 2, 3, 4, 21, 22, 23, 24],  # violence-related
    'illegal': [5, 6, 7, 8, 9, 25, 26, 27, 28, 29],  # hacking, theft, etc
    'harmful_content': [10, 11, 12, 13, 14],  # scams, manipulation
    'self_harm': [15, 16, 17],  # self-harm related
    'deception': [18, 19, 20],  # fake news, misinfo
}


def main():
    print("="*60)
    print("CHECKING HARM CATEGORY STRUCTURE IN RESIDUALS")
    print("="*60)

    # Print categories
    print("\nCategories:")
    for cat, indices in CATEGORIES.items():
        print(f"  {cat}: {len(indices)} prompts")
        for i in indices[:2]:
            print(f"    - {HARMFUL_PROMPTS[i]}")

    # Load model
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model_name = "Qwen/Qwen3-0.6B"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float32, trust_remote_code=True
    ).to(device)
    model.eval()

    # Get activations
    layer = 14
    harmful_acts = get_activations(model, tokenizer, HARMFUL_PROMPTS, [layer], device)[layer]
    harmless_acts = get_activations(model, tokenizer, HARMLESS_PROMPTS, [layer], device)[layer]

    # Compute mean-diff and residuals
    mean_diff = harmful_acts.mean(0) - harmless_acts.mean(0)
    mean_diff = mean_diff / np.linalg.norm(mean_diff)

    harmful_residual = harmful_acts - np.outer(harmful_acts @ mean_diff, mean_diff)

    # PCA on residuals
    pca = PCA(n_components=10)
    harmful_pca = pca.fit_transform(harmful_residual)

    print(f"\nPCA on harmful residuals:")
    print(f"  Explained variance: {pca.explained_variance_ratio_[:5]}")
    print(f"  Top 5 cumulative: {pca.explained_variance_ratio_[:5].sum():.3f}")

    # Check if categories separate in PCA space
    print("\nCategory centroids in PC1-PC2:")
    for cat, indices in CATEGORIES.items():
        centroid = harmful_pca[indices, :2].mean(axis=0)
        std = harmful_pca[indices, :2].std(axis=0)
        print(f"  {cat:15s}: PC1={centroid[0]:+.2f}±{std[0]:.2f}, PC2={centroid[1]:+.2f}±{std[1]:.2f}")

    # Compute inter-category distances
    print("\nInter-category distances (L2 in first 5 PCs):")
    cat_names = list(CATEGORIES.keys())
    cat_centroids = {cat: harmful_pca[indices, :5].mean(axis=0) for cat, indices in CATEGORIES.items()}

    for i, cat1 in enumerate(cat_names):
        for cat2 in cat_names[i+1:]:
            dist = np.linalg.norm(cat_centroids[cat1] - cat_centroids[cat2])
            print(f"  {cat1} <-> {cat2}: {dist:.2f}")

    # Within-category vs between-category variance
    within_var = []
    for cat, indices in CATEGORIES.items():
        if len(indices) > 1:
            var = np.var(harmful_pca[indices, :5])
            within_var.append(var)

    between_var = np.var([cat_centroids[cat] for cat in cat_names], axis=0).mean()
    mean_within = np.mean(within_var)

    print(f"\nVariance analysis:")
    print(f"  Mean within-category variance: {mean_within:.4f}")
    print(f"  Between-category variance:     {between_var:.4f}")
    print(f"  Ratio (between/within):        {between_var/mean_within:.2f}")
    print(f"  → >1 means categories are separated; <1 means they overlap")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    colors = {'violence': 'red', 'illegal': 'blue', 'harmful_content': 'green',
              'self_harm': 'purple', 'deception': 'orange'}

    # PC1 vs PC2 by category
    for cat, indices in CATEGORIES.items():
        axes[0].scatter(harmful_pca[indices, 0], harmful_pca[indices, 1],
                       c=colors[cat], label=cat, alpha=0.7, s=50)
    axes[0].set_xlabel('PC1 of residuals')
    axes[0].set_ylabel('PC2 of residuals')
    axes[0].legend()
    axes[0].set_title('Harm categories in residual PCA space')

    # PC1 vs PC3
    for cat, indices in CATEGORIES.items():
        axes[1].scatter(harmful_pca[indices, 0], harmful_pca[indices, 2],
                       c=colors[cat], label=cat, alpha=0.7, s=50)
    axes[1].set_xlabel('PC1 of residuals')
    axes[1].set_ylabel('PC3 of residuals')
    axes[1].legend()
    axes[1].set_title('Harm categories in residual PCA space (PC1 vs PC3)')

    plt.tight_layout()
    output_dir = Path(__file__).parent / "outputs"
    plt.savefig(output_dir / "debug_categories.png", dpi=150)
    print(f"\nSaved to {output_dir / 'debug_categories.png'}")

    # Test: Can we classify categories from residuals?
    print("\n" + "="*60)
    print("CATEGORY CLASSIFICATION TEST")
    print("="*60)

    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import cross_val_score

    # Create category labels
    category_labels = np.zeros(len(HARMFUL_PROMPTS), dtype=int)
    for i, (cat, indices) in enumerate(CATEGORIES.items()):
        for idx in indices:
            category_labels[idx] = i

    # Can we classify categories from residuals?
    scores = cross_val_score(KNeighborsClassifier(n_neighbors=3),
                            harmful_residual, category_labels, cv=5)
    print(f"KNN category classification from residuals: {scores.mean():.3f} ± {scores.std():.3f}")
    print(f"Random baseline (5 categories): {1/5:.3f}")

    # From raw activations?
    scores_raw = cross_val_score(KNeighborsClassifier(n_neighbors=3),
                                 harmful_acts, category_labels, cv=5)
    print(f"KNN category classification from raw acts:  {scores_raw.mean():.3f} ± {scores_raw.std():.3f}")


if __name__ == "__main__":
    main()
