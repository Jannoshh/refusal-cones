"""
ICA vs PCA comparison for finding refusal directions.

This experiment demonstrates:
1. PCA finds directions of maximum variance (can mix independent sources)
2. ICA finds statistically independent directions (unmixes sources)
3. Local linear approximation for curved manifolds

Run: python -m experiments.e6_unsupervised.ica_vs_pca_demo
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, FastICA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from pathlib import Path


def create_synthetic_refusal_data(n_samples=500, noise=0.1, seed=42):
    """
    Create synthetic activation data with known ground truth.

    Simulates two independent refusal mechanisms:
    - Source 1: "violence refusal" (uniform distribution)
    - Source 2: "illegal activity refusal" (super-gaussian/laplacian)

    These are mixed linearly to create observed activations.
    """
    np.random.seed(seed)

    # Ground truth independent sources (non-Gaussian for ICA to work)
    s1 = np.random.uniform(-1, 1, n_samples)  # Uniform (sub-gaussian)
    s2 = np.random.laplace(0, 0.5, n_samples)  # Laplacian (super-gaussian)

    sources = np.column_stack([s1, s2])

    # Unknown mixing matrix (what we want to recover)
    # This represents how the two refusal mechanisms combine in activation space
    mixing_matrix = np.array([
        [0.8, 0.6],   # Observed dim 1 = 0.8*violence + 0.6*illegal
        [0.6, -0.8],  # Observed dim 2 = 0.6*violence - 0.8*illegal
    ])

    # Mix the sources (this is what we observe)
    mixed = sources @ mixing_matrix.T

    # Add noise
    mixed += noise * np.random.randn(*mixed.shape)

    return mixed, sources, mixing_matrix


def create_curved_manifold_data(n_samples=500, noise=0.05, seed=42):
    """
    Create data that lies on a curved manifold (S-curve).

    This simulates a case where refusal isn't a linear subspace
    but a curved surface in activation space.
    """
    np.random.seed(seed)

    # Parameter along the curve
    t = np.random.uniform(-np.pi, np.pi, n_samples)

    # S-curve in 2D, embedded in 3D
    x = np.sin(t)
    y = t
    z = np.sign(t) * (np.cos(t) - 1)

    data = np.column_stack([x, y, z])
    data += noise * np.random.randn(*data.shape)

    return data, t


def compare_pca_ica(mixed_data, true_sources, title_prefix=""):
    """Compare PCA and ICA on mixed data."""

    # Standardize
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(mixed_data)

    # PCA
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(data_scaled)

    # ICA
    ica = FastICA(n_components=2, random_state=42, max_iter=1000)
    ica_result = ica.fit_transform(data_scaled)

    # Plotting
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    # Original mixed data
    axes[0].scatter(mixed_data[:, 0], mixed_data[:, 1], alpha=0.5, s=10)
    axes[0].set_title(f"{title_prefix}Observed (Mixed) Data")
    axes[0].set_xlabel("Activation dim 1")
    axes[0].set_ylabel("Activation dim 2")

    # True sources
    axes[1].scatter(true_sources[:, 0], true_sources[:, 1], alpha=0.5, s=10, c='green')
    axes[1].set_title("Ground Truth Sources\n(what we want to recover)")
    axes[1].set_xlabel("Violence refusal")
    axes[1].set_ylabel("Illegal activity refusal")

    # PCA result
    axes[2].scatter(pca_result[:, 0], pca_result[:, 1], alpha=0.5, s=10, c='blue')
    axes[2].set_title("PCA Result\n(finds variance, not independence)")
    axes[2].set_xlabel("PC1")
    axes[2].set_ylabel("PC2")

    # ICA result
    axes[3].scatter(ica_result[:, 0], ica_result[:, 1], alpha=0.5, s=10, c='red')
    axes[3].set_title("ICA Result\n(recovers independent sources)")
    axes[3].set_xlabel("IC1")
    axes[3].set_ylabel("IC2")

    plt.tight_layout()
    return fig, pca, ica, pca_result, ica_result


def measure_recovery_quality(estimated, true_sources):
    """
    Measure how well we recovered the true sources.

    Uses absolute correlation since ICA can flip signs/order.
    """
    n_components = true_sources.shape[1]

    # Compute correlation matrix
    correlations = np.abs(np.corrcoef(estimated.T, true_sources.T)[:n_components, n_components:])

    # Best matching (Hungarian algorithm simplified for 2x2)
    if n_components == 2:
        # Try both assignments
        score1 = correlations[0, 0] + correlations[1, 1]
        score2 = correlations[0, 1] + correlations[1, 0]
        best_score = max(score1, score2) / n_components
    else:
        # For larger, just use diagonal as approximation
        best_score = np.mean(np.max(correlations, axis=1))

    return best_score


def local_linear_approximation(data, n_clusters=5, labels=None):
    """
    Fit local linear directions per cluster.

    For curved manifolds, this captures the local tangent direction
    at different points along the curve.

    Args:
        data: [n_samples, n_features]
        n_clusters: number of local regions
        labels: optional binary labels for harmful/harmless

    Returns:
        cluster_centers: [n_clusters, n_features]
        local_directions: [n_clusters, n_features] - local PCA direction per cluster
    """
    # Cluster the data
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(data)

    local_directions = []

    for i in range(n_clusters):
        mask = cluster_labels == i
        cluster_data = data[mask]

        if len(cluster_data) < 3:
            # Not enough points, use global direction
            local_directions.append(np.zeros(data.shape[1]))
            continue

        # Fit local PCA (first principal component = local tangent)
        local_pca = PCA(n_components=1)
        local_pca.fit(cluster_data)
        local_directions.append(local_pca.components_[0])

    return kmeans.cluster_centers_, np.array(local_directions), cluster_labels


def visualize_local_approximation(data, centers, directions, cluster_labels, param=None):
    """Visualize local linear approximation on curved data."""

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Color by parameter if available, else by cluster
    if param is not None:
        colors = param
        cmap = 'viridis'
    else:
        colors = cluster_labels
        cmap = 'tab10'

    # 3D view
    ax = fig.add_subplot(131, projection='3d')
    ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=colors, cmap=cmap, alpha=0.5, s=10)

    # Draw local directions as arrows
    for center, direction in zip(centers, directions):
        if np.linalg.norm(direction) > 0:
            ax.quiver(center[0], center[1], center[2],
                     direction[0], direction[1], direction[2],
                     color='red', linewidth=2, arrow_length_ratio=0.3)

    ax.set_title("Local Linear Approximation\n(red arrows = local tangent directions)")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    # 2D projection (X-Y)
    axes[1].scatter(data[:, 0], data[:, 1], c=colors, cmap=cmap, alpha=0.5, s=10)
    for center, direction in zip(centers, directions):
        if np.linalg.norm(direction) > 0:
            axes[1].arrow(center[0], center[1], direction[0]*0.5, direction[1]*0.5,
                         head_width=0.1, color='red', linewidth=2)
    axes[1].set_title("X-Y Projection")
    axes[1].set_xlabel("X")
    axes[1].set_ylabel("Y")

    # 2D projection (Y-Z)
    axes[2].scatter(data[:, 1], data[:, 2], c=colors, cmap=cmap, alpha=0.5, s=10)
    for center, direction in zip(centers, directions):
        if np.linalg.norm(direction) > 0:
            axes[2].arrow(center[1], center[2], direction[1]*0.5, direction[2]*0.5,
                         head_width=0.1, color='red', linewidth=2)
    axes[2].set_title("Y-Z Projection")
    axes[2].set_xlabel("Y")
    axes[2].set_ylabel("Z")

    plt.tight_layout()
    return fig


def global_vs_local_comparison(data, param):
    """Compare global PCA vs local linear approximation."""

    # Global PCA
    pca = PCA(n_components=1)
    pca.fit(data)
    global_direction = pca.components_[0]

    # Local approximation
    centers, local_directions, cluster_labels = local_linear_approximation(data, n_clusters=8)

    # Measure reconstruction error
    # Global: project to line, measure distance
    global_proj = data @ global_direction.reshape(-1, 1) @ global_direction.reshape(1, -1)
    global_error = np.mean(np.linalg.norm(data - global_proj, axis=1))

    # Local: project to local line per cluster
    local_proj = np.zeros_like(data)
    for i in range(len(centers)):
        mask = cluster_labels == i
        if np.linalg.norm(local_directions[i]) > 0:
            d = local_directions[i]
            centered = data[mask] - centers[i]
            local_proj[mask] = centers[i] + centered @ d.reshape(-1, 1) @ d.reshape(1, -1)
        else:
            local_proj[mask] = centers[i]

    local_error = np.mean(np.linalg.norm(data - local_proj, axis=1))

    print(f"\nReconstruction Error Comparison:")
    print(f"  Global PCA (1 direction):     {global_error:.4f}")
    print(f"  Local Linear (8 directions):  {local_error:.4f}")
    print(f"  Improvement: {(global_error - local_error) / global_error * 100:.1f}%")

    return global_direction, centers, local_directions, cluster_labels


def main():
    """Run all experiments."""

    output_dir = Path(__file__).parent / "outputs"
    output_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print("EXPERIMENT 1: ICA vs PCA on Mixed Independent Sources")
    print("=" * 60)
    print("\nScenario: Two independent refusal mechanisms (violence, illegal)")
    print("are linearly mixed in activation space. Can we unmix them?\n")

    # Create synthetic data
    mixed_data, true_sources, mixing_matrix = create_synthetic_refusal_data(n_samples=1000)

    print(f"Ground truth mixing matrix:\n{mixing_matrix}")
    print(f"\nThis means:")
    print(f"  Observed dim 1 = {mixing_matrix[0,0]:.1f}×violence + {mixing_matrix[0,1]:.1f}×illegal")
    print(f"  Observed dim 2 = {mixing_matrix[1,0]:.1f}×violence + {mixing_matrix[1,1]:.1f}×illegal")

    # Compare PCA and ICA
    fig, pca, ica, pca_result, ica_result = compare_pca_ica(mixed_data, true_sources)
    fig.savefig(output_dir / "ica_vs_pca_comparison.png", dpi=150, bbox_inches='tight')
    print(f"\nSaved comparison plot to {output_dir / 'ica_vs_pca_comparison.png'}")

    # Measure recovery quality
    pca_score = measure_recovery_quality(pca_result, true_sources)
    ica_score = measure_recovery_quality(ica_result, true_sources)

    print(f"\nSource Recovery Quality (correlation with ground truth):")
    print(f"  PCA: {pca_score:.3f}")
    print(f"  ICA: {ica_score:.3f}")
    print(f"  → ICA recovers {(ica_score - pca_score) / pca_score * 100:.1f}% better")

    # Show the unmixing matrices
    print(f"\nPCA components (eigenvectors of covariance):")
    print(pca.components_)
    print(f"\nICA unmixing matrix (should approximate inverse of mixing):")
    print(ica.components_)
    print(f"\nTrue inverse mixing matrix:")
    print(np.linalg.inv(mixing_matrix))

    print("\n" + "=" * 60)
    print("EXPERIMENT 2: Local Linear Approximation for Curved Manifolds")
    print("=" * 60)
    print("\nScenario: Refusal directions form a curved S-surface.")
    print("Global PCA misses the curve; local directions follow it.\n")

    # Create curved manifold data
    curved_data, t_param = create_curved_manifold_data(n_samples=800)

    # Compare global vs local
    global_dir, centers, local_dirs, cluster_labels = global_vs_local_comparison(curved_data, t_param)

    # Visualize
    fig = visualize_local_approximation(curved_data, centers, local_dirs, cluster_labels, t_param)
    fig.savefig(output_dir / "local_linear_approximation.png", dpi=150, bbox_inches='tight')
    print(f"\nSaved local approximation plot to {output_dir / 'local_linear_approximation.png'}")

    print("\n" + "=" * 60)
    print("KEY TAKEAWAYS FOR REFUSAL DIRECTION DISCOVERY")
    print("=" * 60)
    print("""
1. PCA vs ICA:
   - PCA finds directions of maximum variance
   - ICA finds statistically independent directions
   - If multiple refusal mechanisms exist, ICA can separate them
   - Use ICA on residuals after removing mean-diff direction

2. Local Linear Approximation:
   - If refusal manifold is curved, one global direction won't work
   - Cluster activations, fit local directions per cluster
   - At inference: find nearest cluster, use its direction
   - Trade-off: more directions = better fit, but more complexity

3. Practical Recipe:
   a) Compute mean-diff direction (baseline)
   b) Project it out, run ICA on residuals → find additional directions
   c) Test if refusal is curved: cluster activations, measure local variance
   d) If curved: use local linear approximation with K directions
""")

    plt.close('all')
    print(f"\nAll outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
