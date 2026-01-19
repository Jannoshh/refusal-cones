"""
Test: Is the residual classification signal coming from variance differences?
"""

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from pathlib import Path

# Load activations from debug
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from experiments.e6_unsupervised.run_on_qwen import (
    HARMFUL_PROMPTS, HARMLESS_PROMPTS, format_chat_prompt, get_activations
)


def main():
    print("="*60)
    print("HYPOTHESIS: Residual signal is from VARIANCE, not structure")
    print("="*60)

    # Load model and get activations
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model_name = "Qwen/Qwen3-0.6B"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float32, trust_remote_code=True
    ).to(device)
    model.eval()

    layer = 14
    harmful_acts = get_activations(model, tokenizer, HARMFUL_PROMPTS, [layer], device)[layer]
    harmless_acts = get_activations(model, tokenizer, HARMLESS_PROMPTS, [layer], device)[layer]

    # Compute mean-diff
    mean_diff = harmful_acts.mean(0) - harmless_acts.mean(0)
    mean_diff = mean_diff / np.linalg.norm(mean_diff)

    # Compute residuals
    X = np.vstack([harmful_acts, harmless_acts])
    y = np.array([1]*len(harmful_acts) + [0]*len(harmless_acts))
    X_residual = X - np.outer(X @ mean_diff, mean_diff)

    harmful_residual = X_residual[:len(harmful_acts)]
    harmless_residual = X_residual[len(harmful_acts):]

    # Test 1: Compare per-sample norms (variance proxy)
    print("\nTest 1: Per-sample L2 norms of residuals")
    harmful_norms = np.linalg.norm(harmful_residual, axis=1)
    harmless_norms = np.linalg.norm(harmless_residual, axis=1)

    print(f"  Harmful residual norms:  {harmful_norms.mean():.3f} ± {harmful_norms.std():.3f}")
    print(f"  Harmless residual norms: {harmless_norms.mean():.3f} ± {harmless_norms.std():.3f}")

    # Can we classify just from norms?
    norms = np.concatenate([harmful_norms, harmless_norms]).reshape(-1, 1)
    scores = cross_val_score(KNeighborsClassifier(n_neighbors=5), norms, y, cv=5)
    print(f"  KNN on norms alone: {scores.mean():.3f} ± {scores.std():.3f}")

    # Test 2: Per-dimension variance
    print("\nTest 2: Per-dimension variance")
    harmful_var = np.var(harmful_residual, axis=0)
    harmless_var = np.var(harmless_residual, axis=0)

    print(f"  Harmful per-dim variance:  mean={harmful_var.mean():.4f}, max={harmful_var.max():.4f}")
    print(f"  Harmless per-dim variance: mean={harmless_var.mean():.4f}, max={harmless_var.max():.4f}")

    # Find dimensions with biggest variance difference
    var_diff = harmful_var - harmless_var
    top_diff_dims = np.argsort(np.abs(var_diff))[-10:]
    print(f"  Top 10 dims with variance difference: {top_diff_dims}")

    # Test 3: Standardize residuals to remove variance effects
    print("\nTest 3: Standardize residuals (remove variance info)")
    # Center each class at origin and scale to unit variance
    harmful_std = (harmful_residual - harmful_residual.mean(0)) / (harmful_residual.std(0) + 1e-8)
    harmless_std = (harmless_residual - harmless_residual.mean(0)) / (harmless_residual.std(0) + 1e-8)
    X_std = np.vstack([harmful_std, harmless_std])

    scores = cross_val_score(KNeighborsClassifier(n_neighbors=5), X_std, y, cv=5)
    print(f"  KNN on standardized residuals: {scores.mean():.3f} ± {scores.std():.3f}")

    # Compare to original
    scores_orig = cross_val_score(KNeighborsClassifier(n_neighbors=5), X_residual, y, cv=5)
    print(f"  KNN on original residuals:     {scores_orig.mean():.3f} ± {scores_orig.std():.3f}")

    # Test 4: Classification using ONLY variance features
    print("\nTest 4: Classification using only variance-based features")
    # For each sample, compute its "variance profile" vs class mean
    harmful_mean = harmful_residual.mean(0)
    harmless_mean = harmless_residual.mean(0)

    # Distance to each class mean
    dist_to_harmful = np.linalg.norm(X_residual - harmful_mean, axis=1)
    dist_to_harmless = np.linalg.norm(X_residual - harmless_mean, axis=1)
    dist_features = np.column_stack([dist_to_harmful, dist_to_harmless])

    scores = cross_val_score(KNeighborsClassifier(n_neighbors=5), dist_features, y, cv=5)
    print(f"  KNN on distance-to-class-mean: {scores.mean():.3f} ± {scores.std():.3f}")

    # Test 5: What if harmful just has outliers?
    print("\nTest 5: Check for outliers")
    all_norms = np.concatenate([harmful_norms, harmless_norms])
    threshold = np.percentile(all_norms, 90)
    print(f"  90th percentile norm: {threshold:.3f}")
    print(f"  Harmful samples above 90th: {(harmful_norms > threshold).sum()}/{len(harmful_norms)}")
    print(f"  Harmless samples above 90th: {(harmless_norms > threshold).sum()}/{len(harmless_norms)}")

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Norm distribution
    axes[0].hist(harmful_norms, bins=15, alpha=0.6, label='Harmful', color='red')
    axes[0].hist(harmless_norms, bins=15, alpha=0.6, label='Harmless', color='blue')
    axes[0].set_xlabel('L2 norm of residual')
    axes[0].set_ylabel('Count')
    axes[0].legend()
    axes[0].set_title('Residual norms by class')

    # Variance per dimension
    axes[1].scatter(range(len(harmful_var)), np.sort(harmful_var)[::-1], alpha=0.5, s=5, label='Harmful')
    axes[1].scatter(range(len(harmless_var)), np.sort(harmless_var)[::-1], alpha=0.5, s=5, label='Harmless')
    axes[1].set_xlabel('Dimension (sorted by variance)')
    axes[1].set_ylabel('Variance')
    axes[1].legend()
    axes[1].set_title('Per-dimension variance')
    axes[1].set_xlim(0, 100)

    # Distance to class means
    axes[2].scatter(dist_to_harmful[y==1], dist_to_harmless[y==1], c='red', alpha=0.6, label='Harmful')
    axes[2].scatter(dist_to_harmful[y==0], dist_to_harmless[y==0], c='blue', alpha=0.6, label='Harmless')
    axes[2].set_xlabel('Distance to harmful mean')
    axes[2].set_ylabel('Distance to harmless mean')
    axes[2].legend()
    axes[2].set_title('Distance to class means (residuals)')
    axes[2].plot([0, 50], [0, 50], 'k--', alpha=0.3)

    plt.tight_layout()
    output_dir = Path(__file__).parent / "outputs"
    plt.savefig(output_dir / "debug_variance.png", dpi=150)
    print(f"\nSaved to {output_dir / 'debug_variance.png'}")

    # Conclusion
    print("\n" + "="*60)
    print("CONCLUSION")
    print("="*60)
    print("""
If KNN on norms alone works well → variance is the signal
If KNN on standardized residuals drops → variance was the key
If KNN on standardized residuals stays high → there's real structure
""")


if __name__ == "__main__":
    main()
