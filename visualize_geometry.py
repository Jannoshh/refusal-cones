#!/usr/bin/env python3
"""
Visualize Cone vs Adaptive Sampling in 3D

Shows the fundamental difference in sampling spaces:
- Cones: Restricted to k-dimensional subspace (circle in 3D for k=2)
- Adaptive: Full hypersphere (entire 2-sphere surface in 3D)
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def visualize_cone_vs_adaptive():
    """Compare cone sampling vs adaptive discovery in 3D."""

    fig = plt.figure(figsize=(18, 6))

    # === Cone Sampling (k=2 in 3D) ===
    ax1 = fig.add_subplot(131, projection='3d')

    # Define 2D subspace (cone basis)
    v1 = np.array([1, 0, 0])
    v2 = np.array([0, 1, 0])

    # Sample from circle in this subspace
    theta = np.linspace(0, 2*np.pi, 100)
    cone_samples = np.outer(np.cos(theta), v1) + np.outer(np.sin(theta), v2)

    # Plot the circle
    ax1.plot(cone_samples[:, 0], cone_samples[:, 1], cone_samples[:, 2], 'b-', linewidth=3, label='Cone manifold')
    ax1.scatter(cone_samples[::10, 0], cone_samples[::10, 1], cone_samples[::10, 2], c='blue', s=50, alpha=0.6)

    # Show basis vectors
    ax1.quiver(0, 0, 0, v1[0], v1[1], v1[2], color='green', arrow_length_ratio=0.1, linewidth=2, label='v₁')
    ax1.quiver(0, 0, 0, v2[0], v2[1], v2[2], color='orange', arrow_length_ratio=0.1, linewidth=2, label='v₂')

    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title('Cone Sampling (k=2)\nS¹ embedded in S²', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.set_box_aspect([1,1,1])

    # === Adaptive Sampling (full sphere) ===
    ax2 = fig.add_subplot(132, projection='3d')

    # Sample uniformly from sphere
    n_samples = 1000
    adaptive_samples = np.random.randn(n_samples, 3)
    adaptive_samples /= np.linalg.norm(adaptive_samples, axis=1)[:, None]

    ax2.scatter(adaptive_samples[:, 0], adaptive_samples[:, 1], adaptive_samples[:, 2],
                c='red', alpha=0.2, s=10)

    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.set_title('Adaptive Sampling\nFull S²', fontsize=12, fontweight='bold')
    ax2.set_box_aspect([1,1,1])

    # === Refusal Landscape R(v) ===
    ax3 = fig.add_subplot(133, projection='3d')

    # Create mock refusal function
    # High refusal near two principal directions (bi-modal)
    principal_dir1 = np.array([1, 0, 0])
    principal_dir2 = np.array([0, 0.7, 0.7]) / np.linalg.norm([0, 0.7, 0.7])

    def R(v):
        """Mock refusal strength function."""
        # Bi-modal: high near two principal directions
        alignment1 = np.abs(np.dot(v, principal_dir1))
        alignment2 = np.abs(np.dot(v, principal_dir2))
        return max(
            np.exp(-10 * (1 - alignment1)),
            np.exp(-10 * (1 - alignment2))
        )

    # Evaluate on sphere
    refusal_strength = np.array([R(v) for v in adaptive_samples])

    # Plot as heat map
    scatter = ax3.scatter(adaptive_samples[:, 0], adaptive_samples[:, 1], adaptive_samples[:, 2],
                         c=refusal_strength, cmap='hot', s=20, alpha=0.6)

    # Mark principal directions
    ax3.quiver(0, 0, 0, principal_dir1[0], principal_dir1[1], principal_dir1[2],
              color='cyan', arrow_length_ratio=0.15, linewidth=3, label='Mode 1')
    ax3.quiver(0, 0, 0, principal_dir2[0], principal_dir2[1], principal_dir2[2],
              color='lime', arrow_length_ratio=0.15, linewidth=3, label='Mode 2')

    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_zlabel('Z')
    ax3.set_title('Refusal Landscape R(v)\nBi-modal Distribution', fontsize=12, fontweight='bold')
    plt.colorbar(scatter, ax=ax3, label='R(v)', shrink=0.6)
    ax3.legend()
    ax3.set_box_aspect([1,1,1])

    plt.tight_layout()
    plt.savefig('cone_vs_adaptive_3d.png', dpi=200, bbox_inches='tight')
    print("✓ Saved visualization to cone_vs_adaptive_3d.png")

    return fig


def visualize_discovered_geometry_scenarios():
    """Show different scenarios of discovered geometry."""

    fig = plt.figure(figsize=(18, 12))

    scenarios = [
        {
            'name': 'Low-Dim Linear (Use Cone)',
            'type': 'cone',
            'k': 2,
            'description': 'Intrinsic dim=2, flat subspace'
        },
        {
            'name': 'Curved Manifold (Use GP)',
            'type': 'curved',
            'description': 'Intrinsic dim=2, but curved'
        },
        {
            'name': 'Multi-Modal (Use GP)',
            'type': 'multimodal',
            'description': 'Multiple disconnected modes'
        },
        {
            'name': 'High-Dim Diffuse (Use Field)',
            'type': 'diffuse',
            'description': 'Intrinsic dim=50+, no clear structure'
        }
    ]

    n_samples = 1000
    adaptive_samples = np.random.randn(n_samples, 3)
    adaptive_samples /= np.linalg.norm(adaptive_samples, axis=1)[:, None]

    for idx, scenario in enumerate(scenarios):
        ax = fig.add_subplot(2, 2, idx+1, projection='3d')

        if scenario['type'] == 'cone':
            # Low-dimensional linear subspace
            v1 = np.array([1, 0, 0])
            v2 = np.array([0, 1, 0])

            def R(v):
                # Project onto subspace
                proj = np.dot(v, v1) * v1 + np.dot(v, v2) * v2
                proj_norm = np.linalg.norm(proj)
                return np.exp(-5 * (1 - proj_norm))

        elif scenario['type'] == 'curved':
            # Curved band (not flat)
            def R(v):
                # Refusal along a curved path (e.g., spiral)
                theta = np.arctan2(v[1], v[0])
                phi = np.arccos(v[2])
                # High R along a sinusoidal band
                band_center = 0.5 * np.sin(3 * theta)
                distance_to_band = abs(phi - (np.pi/2 + band_center))
                return np.exp(-10 * distance_to_band)

        elif scenario['type'] == 'multimodal':
            # Multiple disconnected peaks
            modes = [
                np.array([1, 0, 0]),
                np.array([0, 1, 0]),
                np.array([0, 0, 1]),
                np.array([-0.5, -0.5, -0.7]) / np.linalg.norm([-0.5, -0.5, -0.7])
            ]

            def R(v):
                return max(np.exp(-10 * (1 - abs(np.dot(v, mode)))) for mode in modes)

        else:  # diffuse
            # High-dimensional, spread out
            def R(v):
                # Many weak components
                return 0.3 + 0.2 * np.random.rand() + 0.3 * abs(np.dot(v, [1, 1, 1]) / np.sqrt(3))

        # Evaluate refusal
        refusal_strength = np.array([R(v) for v in adaptive_samples])

        # Plot
        scatter = ax.scatter(adaptive_samples[:, 0], adaptive_samples[:, 1], adaptive_samples[:, 2],
                           c=refusal_strength, cmap='hot', s=15, alpha=0.5)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f"{scenario['name']}\n{scenario['description']}", fontsize=11, fontweight='bold')
        plt.colorbar(scatter, ax=ax, label='R(v)', shrink=0.6)
        ax.set_box_aspect([1,1,1])

    plt.tight_layout()
    plt.savefig('geometry_scenarios.png', dpi=200, bbox_inches='tight')
    print("✓ Saved geometry scenarios to geometry_scenarios.png")

    return fig


def visualize_sampling_density():
    """Show sampling density comparison."""

    fig = plt.figure(figsize=(15, 5))

    # Cone sampling (restricted)
    ax1 = fig.add_subplot(131, projection='3d')

    # 2D subspace
    v1 = np.array([1, 0, 0])
    v2 = np.array([0, 1, 0])

    n_cone = 100
    theta = np.linspace(0, 2*np.pi, n_cone)
    cone_samples = np.outer(np.cos(theta), v1) + np.outer(np.sin(theta), v2)

    ax1.scatter(cone_samples[:, 0], cone_samples[:, 1], cone_samples[:, 2], c='blue', s=30)
    ax1.set_title(f'Cone: {n_cone} samples\non S¹ (1D manifold)', fontweight='bold')
    ax1.set_box_aspect([1,1,1])

    # Adaptive sampling (full sphere, same count)
    ax2 = fig.add_subplot(132, projection='3d')

    adaptive_100 = np.random.randn(n_cone, 3)
    adaptive_100 /= np.linalg.norm(adaptive_100, axis=1)[:, None]

    ax2.scatter(adaptive_100[:, 0], adaptive_100[:, 1], adaptive_100[:, 2], c='red', s=30, alpha=0.6)
    ax2.set_title(f'Adaptive: {n_cone} samples\non S² (2D manifold)\nSPARSE!', fontweight='bold')
    ax2.set_box_aspect([1,1,1])

    # Adaptive with more samples
    ax3 = fig.add_subplot(133, projection='3d')

    adaptive_1000 = np.random.randn(1000, 3)
    adaptive_1000 /= np.linalg.norm(adaptive_1000, axis=1)[:, None]

    ax3.scatter(adaptive_1000[:, 0], adaptive_1000[:, 1], adaptive_1000[:, 2], c='red', s=10, alpha=0.3)
    ax3.set_title('Adaptive: 1000 samples\non S² (2D manifold)\nDENSE', fontweight='bold')
    ax3.set_box_aspect([1,1,1])

    plt.tight_layout()
    plt.savefig('sampling_density.png', dpi=200, bbox_inches='tight')
    print("✓ Saved sampling density comparison to sampling_density.png")

    return fig


def main():
    """Generate all visualizations."""

    print("=" * 70)
    print("Geometry Visualization: Cones vs Adaptive Discovery")
    print("=" * 70)

    print("\n1. Generating cone vs adaptive comparison...")
    visualize_cone_vs_adaptive()

    print("\n2. Generating geometry scenario examples...")
    visualize_discovered_geometry_scenarios()

    print("\n3. Generating sampling density comparison...")
    visualize_sampling_density()

    print("\n" + "=" * 70)
    print("Visualizations complete!")
    print("=" * 70)
    print("\nGenerated files:")
    print("  - cone_vs_adaptive_3d.png")
    print("  - geometry_scenarios.png")
    print("  - sampling_density.png")
    print("\nKey insights:")
    print("  • Cones: Sample from S^(k-1) (circle in 3D for k=2)")
    print("  • Adaptive: Sample from S^(d-1) (full sphere in 3D)")
    print("  • Discovered geometry determines if cone approximation works")
    print("  • Complex geometry (curved/multi-modal) needs GP or neural field")


if __name__ == '__main__':
    try:
        main()
        plt.show()
    except ImportError as e:
        print(f"⚠ Missing dependency: {e}")
        print("\nTo run visualizations, install:")
        print("  pip install matplotlib numpy")
