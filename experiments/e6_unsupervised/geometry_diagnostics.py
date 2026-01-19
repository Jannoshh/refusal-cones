"""
Geometry Diagnostics: Determine which model best describes refusal geometry.

Tests three hypotheses:
1. Linear (single direction) - mean-diff is sufficient
2. Multi-linear (ICA) - multiple independent directions
3. Curved manifold - need local approximation

Run: python -m experiments.e6_unsupervised.geometry_diagnostics
"""

import numpy as np
import torch
from sklearn.decomposition import PCA, FastICA
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List
import warnings
warnings.filterwarnings('ignore')


@dataclass
class GeometryDiagnostics:
    """Results from geometry diagnostic tests."""
    # Hypothesis scores (higher = more likely)
    linear_score: float
    multi_linear_score: float
    curved_score: float

    # Detailed metrics
    residual_variance_ratio: float  # Var(residuals) / Var(original)
    ica_separation_quality: float   # How well ICA separates categories
    curvature_estimate: float       # Average local direction variance
    intrinsic_dimension: int        # Estimated intrinsic dim

    # Recommended approach
    recommended: str

    def __repr__(self):
        return f"""
Geometry Diagnostics Results:
{'='*50}
Hypothesis Scores (higher = more likely):
  Linear (1 direction):     {self.linear_score:.3f}
  Multi-linear (ICA):       {self.multi_linear_score:.3f}
  Curved manifold:          {self.curved_score:.3f}

Detailed Metrics:
  Residual variance ratio:  {self.residual_variance_ratio:.3f}
  ICA separation quality:   {self.ica_separation_quality:.3f}
  Curvature estimate:       {self.curvature_estimate:.3f}
  Intrinsic dimension:      {self.intrinsic_dimension}

Recommendation: {self.recommended}
{'='*50}
"""


def compute_mean_diff_direction(harmful_acts: np.ndarray,
                                 harmless_acts: np.ndarray) -> np.ndarray:
    """Compute standard mean difference direction."""
    direction = harmful_acts.mean(axis=0) - harmless_acts.mean(axis=0)
    return direction / np.linalg.norm(direction)


def test_residual_structure(harmful_acts: np.ndarray,
                            harmless_acts: np.ndarray,
                            direction: np.ndarray) -> Tuple[float, float]:
    """
    Test 1: After removing mean-diff direction, is there structure in residuals?

    If linear hypothesis is correct:
    - Residuals should be noise
    - Classification accuracy on residuals should be ~50%

    Returns:
        residual_variance_ratio: Var(residual) / Var(original)
        residual_classification_acc: Accuracy of classifying H vs B from residuals
    """
    # Project out the direction
    def remove_direction(X, d):
        return X - np.outer(X @ d, d)

    harmful_residual = remove_direction(harmful_acts, direction)
    harmless_residual = remove_direction(harmless_acts, direction)

    # Variance ratio
    original_var = np.var(np.vstack([harmful_acts, harmless_acts]))
    residual_var = np.var(np.vstack([harmful_residual, harmless_residual]))
    variance_ratio = residual_var / original_var

    # Can we still classify from residuals?
    X_residual = np.vstack([harmful_residual, harmless_residual])
    y = np.array([1] * len(harmful_residual) + [0] * len(harmless_residual))

    clf = LogisticRegression(max_iter=1000)
    try:
        scores = cross_val_score(clf, X_residual, y, cv=5)
        classification_acc = scores.mean()
        # Check for inverted learning: if acc < 0.5, classifier learned opposite
        # Report the "true" accuracy (max of acc and 1-acc)
        classification_acc = max(classification_acc, 1 - classification_acc)
    except:
        classification_acc = 0.5

    return variance_ratio, classification_acc


def test_ica_separation(harmful_acts: np.ndarray,
                        harmless_acts: np.ndarray,
                        categories: Optional[np.ndarray] = None,
                        n_components: int = 5) -> Tuple[float, np.ndarray]:
    """
    Test 2: Do ICA components separate different harm categories?

    If multi-linear hypothesis is correct:
    - Different ICs should correlate with different categories
    - ICs should have distinct "specializations"

    Returns:
        separation_quality: How well ICs separate categories (0-1)
        ica_components: The learned ICA directions
    """
    # Combine data
    X = np.vstack([harmful_acts, harmless_acts])

    # Fit ICA
    ica = FastICA(n_components=n_components, random_state=42, max_iter=1000)
    try:
        X_ica = ica.fit_transform(X)
    except:
        return 0.0, np.zeros((n_components, harmful_acts.shape[1]))

    # Separate back
    harmful_ica = X_ica[:len(harmful_acts)]
    harmless_ica = X_ica[len(harmful_acts):]

    # Measure separation: how different are harmful vs harmless on each IC?
    separation_scores = []
    for i in range(n_components):
        harmful_mean = np.abs(harmful_ica[:, i].mean())
        harmless_mean = np.abs(harmless_ica[:, i].mean())
        harmful_std = harmful_ica[:, i].std()
        harmless_std = harmless_ica[:, i].std()

        # Cohen's d-like measure
        pooled_std = np.sqrt((harmful_std**2 + harmless_std**2) / 2)
        if pooled_std > 0:
            d = np.abs(harmful_mean - harmless_mean) / pooled_std
            separation_scores.append(min(d, 3.0) / 3.0)  # Cap at 3, normalize to 0-1
        else:
            separation_scores.append(0.0)

    # If categories provided, check if different ICs specialize
    category_specialization = 0.0
    if categories is not None and len(np.unique(categories)) > 1:
        # Check if each IC correlates differently with categories
        from scipy.stats import spearmanr
        correlations = []
        for i in range(n_components):
            corr, _ = spearmanr(X_ica[:len(harmful_acts), i], categories)
            correlations.append(np.abs(corr) if not np.isnan(corr) else 0)
        category_specialization = np.std(correlations)  # High std = different specializations

    separation_quality = np.mean(separation_scores) + category_specialization
    return min(separation_quality, 1.0), ica.components_


def test_curvature(harmful_acts: np.ndarray,
                   n_clusters: int = 10,
                   n_neighbors: int = 20) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Test 3: Does the refusal manifold have significant curvature?

    If curved hypothesis is correct:
    - Local PCA directions should vary significantly across the manifold
    - Local reconstruction should be much better than global

    Returns:
        curvature_estimate: Average angular variance of local directions (0-1)
        cluster_centers: Centers of local clusters
        local_directions: Local PCA directions at each cluster
    """
    if len(harmful_acts) < n_clusters * 3:
        n_clusters = max(2, len(harmful_acts) // 3)

    # Cluster
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(harmful_acts)

    # Fit local PCA at each cluster
    local_directions = []
    for i in range(n_clusters):
        cluster_data = harmful_acts[labels == i]
        if len(cluster_data) >= 3:
            pca = PCA(n_components=1)
            pca.fit(cluster_data)
            local_directions.append(pca.components_[0])
        else:
            local_directions.append(np.zeros(harmful_acts.shape[1]))

    local_directions = np.array(local_directions)

    # Measure direction variance (curvature proxy)
    # Compute pairwise angles between local directions
    angles = []
    for i in range(len(local_directions)):
        for j in range(i + 1, len(local_directions)):
            d1, d2 = local_directions[i], local_directions[j]
            if np.linalg.norm(d1) > 0 and np.linalg.norm(d2) > 0:
                cos_angle = np.abs(np.dot(d1, d2) / (np.linalg.norm(d1) * np.linalg.norm(d2)))
                cos_angle = np.clip(cos_angle, -1, 1)
                angle = np.arccos(cos_angle)
                angles.append(angle)

    if len(angles) > 0:
        # Normalize: 0 = all parallel (flat), 1 = orthogonal on average (curved)
        mean_angle = np.mean(angles)
        curvature = mean_angle / (np.pi / 2)  # Normalize by max angle
    else:
        curvature = 0.0

    return curvature, kmeans.cluster_centers_, local_directions


def estimate_intrinsic_dimension(X: np.ndarray,
                                  method: str = 'pca_ratio') -> int:
    """
    Estimate intrinsic dimensionality of the data.

    Methods:
    - pca_ratio: Number of PCs explaining 95% variance
    - mle: Maximum likelihood estimation (Levina-Bickel)
    """
    if method == 'pca_ratio':
        pca = PCA()
        pca.fit(X)
        cumvar = np.cumsum(pca.explained_variance_ratio_)
        dim = np.searchsorted(cumvar, 0.95) + 1
        return min(dim, X.shape[1])

    elif method == 'mle':
        # Levina-Bickel MLE estimator
        k = min(20, len(X) - 1)
        nbrs = NearestNeighbors(n_neighbors=k + 1).fit(X)
        distances, _ = nbrs.kneighbors(X)
        distances = distances[:, 1:]  # Exclude self

        # MLE estimate
        dims = []
        for i in range(len(X)):
            dist = distances[i]
            dist = dist[dist > 0]
            if len(dist) >= 2:
                log_ratios = np.log(dist[-1] / dist[:-1])
                dim_est = (len(dist) - 1) / np.sum(log_ratios)
                dims.append(dim_est)

        return int(np.median(dims)) if dims else 1

    return 1


def test_ablation_effectiveness(harmful_acts: np.ndarray,
                                 harmless_acts: np.ndarray,
                                 directions: Dict[str, np.ndarray],
                                 measure_fn: Optional[callable] = None) -> Dict[str, float]:
    """
    Test 4: Which directions most effectively separate harmful/harmless?

    Uses classification accuracy as proxy for ablation effectiveness.
    (In practice, you'd use actual refusal rate measurement)

    Args:
        directions: Dict mapping method name to direction(s)
        measure_fn: Optional actual measurement function

    Returns:
        Dict mapping method name to effectiveness score
    """
    results = {}

    for name, dirs in directions.items():
        if dirs.ndim == 1:
            dirs = dirs.reshape(1, -1)

        # Project data onto directions
        harmful_proj = harmful_acts @ dirs.T
        harmless_proj = harmless_acts @ dirs.T

        # Classification accuracy as proxy
        X = np.vstack([harmful_proj, harmless_proj])
        y = np.array([1] * len(harmful_proj) + [0] * len(harmless_proj))

        clf = LogisticRegression(max_iter=1000)
        try:
            scores = cross_val_score(clf, X, y, cv=5)
            results[name] = scores.mean()
        except:
            results[name] = 0.5

    return results


def run_full_diagnostics(harmful_acts: np.ndarray,
                          harmless_acts: np.ndarray,
                          categories: Optional[np.ndarray] = None,
                          verbose: bool = True) -> GeometryDiagnostics:
    """
    Run all diagnostic tests and determine most likely geometry.

    Args:
        harmful_acts: [n_harmful, hidden_dim] activations for harmful prompts
        harmless_acts: [n_harmless, hidden_dim] activations for harmless prompts
        categories: Optional [n_harmful] category labels for harmful prompts

    Returns:
        GeometryDiagnostics with scores and recommendation
    """
    if verbose:
        print("Running geometry diagnostics...")
        print(f"  Harmful samples: {len(harmful_acts)}")
        print(f"  Harmless samples: {len(harmless_acts)}")
        print(f"  Hidden dim: {harmful_acts.shape[1]}")

    # Compute baseline direction
    mean_diff = compute_mean_diff_direction(harmful_acts, harmless_acts)

    # Test 1: Residual structure
    if verbose:
        print("\nTest 1: Residual structure after mean-diff removal...")
    variance_ratio, residual_acc = test_residual_structure(
        harmful_acts, harmless_acts, mean_diff
    )
    if verbose:
        print(f"  Residual variance ratio: {variance_ratio:.3f}")
        print(f"  Residual classification acc: {residual_acc:.3f}")

    # Test 2: ICA separation
    if verbose:
        print("\nTest 2: ICA component separation...")
    ica_quality, ica_components = test_ica_separation(
        harmful_acts, harmless_acts, categories
    )
    if verbose:
        print(f"  ICA separation quality: {ica_quality:.3f}")

    # Test 3: Curvature
    if verbose:
        print("\nTest 3: Manifold curvature estimation...")
    curvature, centers, local_dirs = test_curvature(harmful_acts)
    if verbose:
        print(f"  Curvature estimate: {curvature:.3f}")

    # Intrinsic dimension
    if verbose:
        print("\nEstimating intrinsic dimension...")
    intrinsic_dim = estimate_intrinsic_dimension(harmful_acts)
    if verbose:
        print(f"  Intrinsic dimension: {intrinsic_dim}")

    # Test 4: Compare ablation effectiveness
    if verbose:
        print("\nTest 4: Comparing direction effectiveness...")

    # Prepare directions for comparison
    directions_to_test = {
        'mean_diff': mean_diff,
        'pca_top3': PCA(n_components=min(3, harmful_acts.shape[1])).fit(
            np.vstack([harmful_acts, harmless_acts])
        ).components_,
        'ica_top3': ica_components[:3] if len(ica_components) >= 3 else ica_components,
    }

    effectiveness = test_ablation_effectiveness(
        harmful_acts, harmless_acts, directions_to_test
    )
    if verbose:
        for name, score in effectiveness.items():
            print(f"  {name}: {score:.3f}")

    # Compute hypothesis scores
    # Linear score: high if residuals are noise (low acc), low variance ratio
    linear_score = (1 - residual_acc) + (1 - variance_ratio) * 0.5
    linear_score = linear_score / 1.5  # Normalize

    # Multi-linear score: high if ICA separates well, intrinsic dim > 1
    multi_linear_score = ica_quality * 0.7 + min(intrinsic_dim / 5, 1) * 0.3

    # Curved score: high curvature, local > global
    curved_score = curvature * 0.7 + (1 - linear_score) * 0.3

    # Normalize scores
    total = linear_score + multi_linear_score + curved_score
    if total > 0:
        linear_score /= total
        multi_linear_score /= total
        curved_score /= total

    # Recommendation
    scores = {
        'Linear (single mean-diff direction)': linear_score,
        'Multi-linear (ICA components)': multi_linear_score,
        'Curved (local linear approximation)': curved_score
    }
    recommended = max(scores, key=scores.get)

    return GeometryDiagnostics(
        linear_score=linear_score,
        multi_linear_score=multi_linear_score,
        curved_score=curved_score,
        residual_variance_ratio=variance_ratio,
        ica_separation_quality=ica_quality,
        curvature_estimate=curvature,
        intrinsic_dimension=intrinsic_dim,
        recommended=recommended
    )


# ============================================================
# Demo with synthetic data
# ============================================================

def create_linear_geometry(n_samples=200, hidden_dim=100, noise=0.1, seed=42):
    """Create data where linear (1D) hypothesis is true."""
    np.random.seed(seed)

    # True refusal direction
    true_direction = np.random.randn(hidden_dim)
    true_direction /= np.linalg.norm(true_direction)

    # Harmful: positive projection onto direction
    harmful_proj = np.random.uniform(0.5, 1.5, n_samples)
    harmful = np.outer(harmful_proj, true_direction)
    harmful += noise * np.random.randn(n_samples, hidden_dim)

    # Harmless: negative projection
    harmless_proj = np.random.uniform(-1.5, -0.5, n_samples)
    harmless = np.outer(harmless_proj, true_direction)
    harmless += noise * np.random.randn(n_samples, hidden_dim)

    return harmful, harmless, "linear"


def create_multilinear_geometry(n_samples=200, hidden_dim=100, n_sources=3, noise=0.1, seed=42):
    """Create data where multi-linear (ICA) hypothesis is true."""
    np.random.seed(seed)

    # Multiple independent refusal sources
    sources_harmful = []
    sources_harmless = []

    for i in range(n_sources):
        # Each source has different distribution (for ICA to work)
        if i % 3 == 0:
            s_harm = np.random.uniform(0.3, 1.0, n_samples)
            s_harmless = np.random.uniform(-1.0, -0.3, n_samples)
        elif i % 3 == 1:
            s_harm = np.abs(np.random.laplace(0.5, 0.3, n_samples))
            s_harmless = -np.abs(np.random.laplace(0.5, 0.3, n_samples))
        else:
            s_harm = np.random.exponential(0.5, n_samples)
            s_harmless = -np.random.exponential(0.5, n_samples)

        sources_harmful.append(s_harm)
        sources_harmless.append(s_harmless)

    sources_harmful = np.column_stack(sources_harmful)
    sources_harmless = np.column_stack(sources_harmless)

    # Random mixing matrix (embedding into high-dim space)
    mixing = np.random.randn(n_sources, hidden_dim)

    harmful = sources_harmful @ mixing + noise * np.random.randn(n_samples, hidden_dim)
    harmless = sources_harmless @ mixing + noise * np.random.randn(n_samples, hidden_dim)

    return harmful, harmless, "multi-linear"


def create_curved_geometry(n_samples=200, hidden_dim=100, noise=0.05, seed=42):
    """Create data where curved manifold hypothesis is true."""
    np.random.seed(seed)

    # Parameter along curve
    t_harmful = np.random.uniform(0, np.pi, n_samples)
    t_harmless = np.random.uniform(-np.pi, 0, n_samples)

    # Embed curve into high-dim space
    def curve_embedding(t, hidden_dim):
        # Create a smooth curve using multiple frequencies
        coords = []
        for freq in range(1, 6):
            coords.append(np.sin(freq * t))
            coords.append(np.cos(freq * t))

        base = np.column_stack(coords)  # [n, 10]

        # Embed into higher dim
        embedding = np.random.RandomState(42).randn(base.shape[1], hidden_dim)
        return base @ embedding

    harmful = curve_embedding(t_harmful, hidden_dim)
    harmless = curve_embedding(t_harmless, hidden_dim)

    harmful += noise * np.random.randn(*harmful.shape)
    harmless += noise * np.random.randn(*harmless.shape)

    return harmful, harmless, "curved"


def demo():
    """Run diagnostics on synthetic data to validate the tests."""
    print("=" * 60)
    print("GEOMETRY DIAGNOSTICS DEMO")
    print("=" * 60)
    print("\nTesting diagnostic accuracy on synthetic data with known geometry.\n")

    results = []

    for create_fn, name in [
        (create_linear_geometry, "Linear"),
        (create_multilinear_geometry, "Multi-linear"),
        (create_curved_geometry, "Curved"),
    ]:
        print(f"\n{'='*60}")
        print(f"Testing: {name} geometry (ground truth)")
        print("=" * 60)

        harmful, harmless, true_type = create_fn()
        diagnostics = run_full_diagnostics(harmful, harmless)

        print(diagnostics)

        # Check if recommendation matches ground truth
        correct = true_type.lower() in diagnostics.recommended.lower()
        results.append((name, correct, diagnostics.recommended))

    print("\n" + "=" * 60)
    print("DIAGNOSTIC ACCURACY SUMMARY")
    print("=" * 60)
    for name, correct, recommended in results:
        status = "✓" if correct else "✗"
        print(f"  {status} {name}: recommended '{recommended}'")

    accuracy = sum(c for _, c, _ in results) / len(results)
    print(f"\nOverall accuracy: {accuracy:.0%}")


if __name__ == "__main__":
    demo()
