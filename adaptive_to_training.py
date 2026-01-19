#!/usr/bin/env python3
"""
Complete Pipeline: Adaptive Discovery → Training

Shows how to:
1. Run adaptive geometry discovery
2. Get recommendation (cone vs GP/field)
3. Initialize appropriate model
4. Train with RDO

This answers: "Do we still need cones?"
Answer: Only if discovered geometry is low-dimensional and simple!
"""

import torch
from typing import Callable

from adaptive_geometry_discovery import RefusalGeometryDiscovery, GeometryConfig
from rdo_peft_adapter import get_rdo_model, RDOConfig, get_cone_model
from rdo_peft_trainer import train_rdo_with_peft


def adaptive_discovery_to_training_pipeline(
    model_name: str,
    measure_refusal_fn: Callable,
    harmful_data: list,
    harmless_data: list,
    n_layers: int = 26,
    hidden_dim: int = 2048,
    discovery_config: GeometryConfig = None,
    output_dir: str = "rdo_adapters_adaptive"
):
    """
    Complete pipeline from adaptive discovery to training.

    Args:
        model_name: Model to train
        measure_refusal_fn: Function to measure R(v) during discovery
        harmful_data: Harmful examples for RDO training
        harmless_data: Harmless examples for RDO training
        n_layers: Number of layers
        hidden_dim: Hidden dimension
        discovery_config: Config for adaptive discovery
        output_dir: Where to save trained adapters

    Returns:
        Trained model and discovery results
    """

    print("=" * 70)
    print("ADAPTIVE DISCOVERY → TRAINING PIPELINE")
    print("=" * 70)

    # ========================================================================
    # PHASE 1: ADAPTIVE DISCOVERY
    # ========================================================================

    print("\n" + "=" * 70)
    print("PHASE 1: Adaptive Geometry Discovery")
    print("=" * 70)
    print("\nGoal: Discover true geometry of refusal subspace")
    print("Method: GP + active exploration on FULL hypersphere S^(d-1)")
    print(f"Space: d = {n_layers} × {hidden_dim} = {n_layers * hidden_dim:,} dimensions")
    print(f"Sampling: All unit vectors, NOT restricted to subspace!")

    discovery_config = discovery_config or GeometryConfig(
        n_init_random=20,
        n_iterations=50,
        n_candidates=500,
        acquisition_type='ucb',
        beta=2.0
    )

    discovery = RefusalGeometryDiscovery(
        measure_refusal_fn=measure_refusal_fn,
        n_layers=n_layers,
        hidden_dim=hidden_dim,
        config=discovery_config
    )

    results = discovery.discover()

    # ========================================================================
    # PHASE 2: ANALYZE DISCOVERED GEOMETRY
    # ========================================================================

    print("\n" + "=" * 70)
    print("PHASE 2: Geometry Analysis")
    print("=" * 70)

    geometry = results['geometry']
    recommendation = results['recommendation']

    print("\nDiscovered properties:")
    print(f"  • Principal modes: {geometry.get('n_modes', 'N/A')}")
    print(f"  • Intrinsic dimension: {geometry.get('intrinsic_dimension', 'N/A')}")
    if 'avg_intra_distance' in geometry:
        print(f"  • Average spread: {geometry['avg_intra_distance']:.3f}")

    print("\nRepresentation recommendation:")
    print(f"  • Use cone: {recommendation['use_cone']}")
    print(f"  • Reasoning: {recommendation['reasoning']}")

    if recommendation['use_cone']:
        print(f"  • Cone rank: {recommendation['cone_rank']}")
        if 'efficiency_gain' in recommendation:
            print(f"  • Efficiency: {recommendation['efficiency_gain']}")
    else:
        print(f"  • Alternative: {recommendation.get('alternative', 'N/A')}")

    # ========================================================================
    # PHASE 3: INITIALIZE MODEL BASED ON RECOMMENDATION
    # ========================================================================

    print("\n" + "=" * 70)
    print("PHASE 3: Model Initialization")
    print("=" * 70)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if recommendation['use_cone']:
        # Use cone initialized with discovered principal directions
        print("\n✓ Using CONE representation")
        print(f"  Initializing {recommendation['cone_rank']}-dimensional cone")
        print(f"  with discovered principal directions")

        cone_vectors = geometry['principal_directions'][:recommendation['cone_rank']]

        rdo_config = RDOConfig(
            target_modules=["layers"],
            operation='both',
            enable_cone=True,
            cone_rank=recommendation['cone_rank'],
            projection_alpha=1.0,
            addition_alpha=1.0,
            normalize_vectors=True
        )

        model = get_rdo_model(base_model, rdo_config)

        # Initialize cone with discovered vectors
        print("  Initializing cone vectors from discovery...")
        for i, module in enumerate(model.modules()):
            if hasattr(module, 'cone_vectors'):
                # Initialize with discovered directions (if available for this layer)
                if i < len(cone_vectors):
                    with torch.no_grad():
                        module.cone_vectors.data = cone_vectors[i].to(module.cone_vectors.device)
                print(f"    Layer {i}: initialized with discovered directions")

    else:
        # Use standard RDO (will learn directions from scratch)
        # OR: Could use GP guidance (more advanced)
        print("\n✓ Using STANDARD RDO (single vector per layer)")
        print("  Geometry is too complex for cone approximation")
        print("  Alternative: Could use GP-guided training (future work)")

        rdo_config = RDOConfig(
            target_modules=["layers"],
            operation='both',
            enable_cone=False,  # No cone, just single vectors
            projection_alpha=1.0,
            addition_alpha=1.0,
            normalize_vectors=True
        )

        model = get_rdo_model(base_model, rdo_config)

        # Could initialize single vectors from first principal direction
        print("  Initializing single vectors from first principal direction...")
        principal_dir = geometry['principal_directions'][0]
        for i, module in enumerate(model.modules()):
            if hasattr(module, 'projection_vector'):
                with torch.no_grad():
                    module.projection_vector.data = principal_dir[i].to(module.projection_vector.device)

    # ========================================================================
    # PHASE 4: TRAIN WITH RDO
    # ========================================================================

    print("\n" + "=" * 70)
    print("PHASE 4: RDO Training")
    print("=" * 70)

    print("\nTraining with multi-objective loss:")
    print("  • Ablation loss (harmful → harmful when ablated)")
    print("  • Addition loss (harmful → refusal when added)")
    print("  • Retain loss (harmless → helpful unchanged)")

    # Train
    trained_model, trainer = train_rdo_with_peft(
        model_name=model_name,
        harmful_data=harmful_data,
        harmless_data=harmless_data,
        output_dir=output_dir,
        num_epochs=10,
        batch_size=4,
        learning_rate=1e-3,
        lambda_ablate=1.0,
        lambda_add=1.0,
        lambda_retain=0.5,
        enable_cone=recommendation['use_cone'],
        cone_rank=recommendation.get('cone_rank', 1) if recommendation['use_cone'] else 1,
        fp16=True
    )

    # ========================================================================
    # SUMMARY
    # ========================================================================

    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)

    print("\nWhat we did:")
    print("  1. ✓ Discovered geometry via GP + active exploration")
    print("  2. ✓ Analyzed intrinsic dimension and structure")
    print("  3. ✓ Chose representation (cone vs standard)")
    print("  4. ✓ Initialized model with discovered directions")
    print("  5. ✓ Trained with RDO multi-objective loss")

    print(f"\nResults:")
    print(f"  • Representation: {'Cone' if recommendation['use_cone'] else 'Standard'}")
    if recommendation['use_cone']:
        print(f"  • Cone rank: {recommendation['cone_rank']}")
    print(f"  • Intrinsic dimension: {geometry.get('intrinsic_dimension', 'N/A')}")
    print(f"  • Total measurements: {len(results['R_observed'])}")
    print(f"  • Max refusal strength: {results['R_observed'].max():.3f}")

    print(f"\n✓ Adapters saved to: {output_dir}")

    return trained_model, results


def demo_simple_geometry():
    """Demo with simple low-dimensional geometry (should recommend cone)."""

    print("\n" + "=" * 70)
    print("DEMO 1: Simple Low-Dimensional Geometry")
    print("=" * 70)
    print("\nSimulating refusal subspace with:")
    print("  • Intrinsic dimension: 2")
    print("  • Linear subspace (flat)")
    print("  • Single mode")
    print("\nExpected: Recommend CONE with rank=2")

    # Mock refusal function with 2D linear subspace
    true_dir1 = torch.randn(26, 2048)
    true_dir1 = true_dir1 / true_dir1.norm(dim=1, keepdim=True)

    true_dir2 = torch.randn(26, 2048)
    true_dir2 = true_dir2 / true_dir2.norm(dim=1, keepdim=True)

    def measure_refusal_simple(v: torch.Tensor) -> float:
        """Simple 2D linear subspace."""
        v_flat = v.reshape(-1)
        dir1_flat = true_dir1.reshape(-1)
        dir2_flat = true_dir2.reshape(-1)

        # Project onto 2D subspace
        proj1 = (v_flat @ dir1_flat) / (dir1_flat.norm() ** 2)
        proj2 = (v_flat @ dir2_flat) / (dir2_flat.norm() ** 2)

        # Refusal strength = norm of projection
        R = (proj1 ** 2 + proj2 ** 2).sqrt().item()

        # Add noise
        R = R + 0.05 * torch.randn(1).item()
        R = np.clip(R, 0, 1)

        return R

    # Run discovery only (skip training for demo)
    discovery = RefusalGeometryDiscovery(
        measure_refusal_fn=measure_refusal_simple,
        n_layers=26,
        hidden_dim=2048,
        config=GeometryConfig(
            n_init_random=10,
            n_iterations=20,
            n_candidates=100
        )
    )

    results = discovery.discover()

    print("\n" + "=" * 70)
    print("RESULT")
    print("=" * 70)

    recommendation = results['recommendation']
    print(f"\nRecommendation: {'USE CONE ✓' if recommendation['use_cone'] else 'DO NOT use cone ✗'}")
    print(f"Reasoning: {recommendation['reasoning']}")


def demo_complex_geometry():
    """Demo with complex multi-modal geometry (should NOT recommend cone)."""

    print("\n" + "=" * 70)
    print("DEMO 2: Complex Multi-Modal Geometry")
    print("=" * 70)
    print("\nSimulating refusal subspace with:")
    print("  • Multiple disconnected modes (4 peaks)")
    print("  • Curved manifold")
    print("  • Higher intrinsic dimension")
    print("\nExpected: DO NOT recommend cone, suggest GP/field")

    # Multiple random modes
    modes = [torch.randn(26, 2048) / torch.randn(26, 2048).norm() for _ in range(4)]

    def measure_refusal_complex(v: torch.Tensor) -> float:
        """Multi-modal with 4 peaks."""
        v_flat = v.reshape(-1)

        # Max alignment with any mode
        alignments = [
            abs((v_flat @ mode.reshape(-1)) / (v_flat.norm() * mode.reshape(-1).norm())).item()
            for mode in modes
        ]

        R = max(alignments)

        # Add noise
        R = R + 0.05 * torch.randn(1).item()
        R = np.clip(R, 0, 1)

        return R

    # Run discovery
    discovery = RefusalGeometryDiscovery(
        measure_refusal_fn=measure_refusal_complex,
        n_layers=26,
        hidden_dim=2048,
        config=GeometryConfig(
            n_init_random=10,
            n_iterations=20,
            n_candidates=100
        )
    )

    results = discovery.discover()

    print("\n" + "=" * 70)
    print("RESULT")
    print("=" * 70)

    recommendation = results['recommendation']
    print(f"\nRecommendation: {'USE CONE ✓' if recommendation['use_cone'] else 'DO NOT use cone ✗'}")
    print(f"Reasoning: {recommendation['reasoning']}")
    if not recommendation['use_cone']:
        print(f"Alternative: {recommendation['alternative']}")


def main():
    """Run demos."""

    print("=" * 70)
    print("ADAPTIVE DISCOVERY → TRAINING")
    print("Answering: Do we still need cones?")
    print("=" * 70)

    import numpy as np

    # Demo 1: Simple geometry (should recommend cone)
    demo_simple_geometry()

    print("\n\n")

    # Demo 2: Complex geometry (should NOT recommend cone)
    demo_complex_geometry()

    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    print("\nCones are still useful, but ONLY when:")
    print("  ✓ Discovered intrinsic dimension is low (≤10)")
    print("  ✓ Geometry is approximately linear (flat subspace)")
    print("  ✓ Single-mode (not multi-modal)")
    print("\nIf geometry is complex:")
    print("  ✗ High intrinsic dimension")
    print("  ✗ Curved manifold")
    print("  ✗ Multiple disconnected modes")
    print("  → Use GP or neural field instead!")
    print("\nAdaptive discovery tells you which to use!")


if __name__ == '__main__':
    main()
