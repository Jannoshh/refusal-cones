"""
Modal app configuration for refusal-cones GPU workloads.

Usage:
    # Authenticate (one-time setup)
    modal token new

    # Run discovery experiment
    modal run modal_app.py::run_discovery --model "Qwen/Qwen3-0.6B"

    # Run evaluation
    modal run modal_app.py::run_evaluation --model "Qwen/Qwen3-0.6B"

    # Run Pareto discovery
    modal run modal_app.py::run_pareto_discovery --model "Qwen/Qwen3-0.6B"

    # Run two-phase boundary then Pareto discovery
    modal run modal_app.py::run_boundary_then_pareto --model "Qwen/Qwen3-0.6B"

    # Run single-layer discovery
    modal run modal_app.py::run_single_layer_discovery --model "Qwen/Qwen2-1.5B-Instruct"

    # Run single-vector discovery (simplest approach)
    modal run modal_app.py::run_single_vector_discovery --model "Qwen/Qwen2-1.5B-Instruct"

    # Deploy as serverless endpoint
    modal deploy modal_app.py

This is a thin wrapper that imports from the modal_app/ package.
The actual implementations are in:
- modal_app/config.py - Shared configuration (app, image, volumes)
- modal_app/utils.py - Utility functions (load_prompts, etc.)
- modal_app/discovery.py - Basic geometry discovery
- modal_app/evaluation.py - Model evaluation functions
- modal_app/pareto.py - Pareto-based discovery
- modal_app/boundary.py - Boundary-based discovery
- modal_app/single_layer.py - Single-layer discovery and testing
"""

# Import everything from the package
from modal_app import (
    # Config
    DEFAULT_N_HARMFUL,
    DEFAULT_N_HARMLESS,
    GPU_CONFIG,
    app,
    cuda_image,
    model_cache,
    results_volume,
    # Utils
    load_prompts,
    # Discovery
    run_discovery,
    # Evaluation
    run_batch_evaluation,
    run_evaluation,
    # Pareto
    run_high_budget_discovery,
    run_pareto_ablations,
    run_pareto_discovery,
    # Boundary
    run_boundary_only,
    run_boundary_then_pareto,
    run_pareto_from_cache,
    # Single-layer and single-vector
    run_single_layer_discovery,
    run_single_vector_discovery,
    test_single_layer_refusal,
)

# Re-export for backwards compatibility
__all__ = [
    # Config
    "app",
    "cuda_image",
    "model_cache",
    "results_volume",
    "GPU_CONFIG",
    "DEFAULT_N_HARMFUL",
    "DEFAULT_N_HARMLESS",
    # Utils
    "load_prompts",
    # Discovery
    "run_discovery",
    # Evaluation
    "run_evaluation",
    "run_batch_evaluation",
    # Pareto
    "run_pareto_discovery",
    "run_pareto_ablations",
    "run_high_budget_discovery",
    # Boundary
    "run_boundary_then_pareto",
    "run_boundary_only",
    "run_pareto_from_cache",
    # Single-layer and single-vector
    "run_single_layer_discovery",
    "run_single_vector_discovery",
    "test_single_layer_refusal",
]


def download_results(timestamp: str, prefix: str = "single_vector") -> None:
    """Download results from Modal volume to local results/ directory."""
    import os
    import subprocess

    # Create local results directory
    os.makedirs("results", exist_ok=True)

    # Files to download
    files = [
        f"{prefix}_{timestamp}.json",
        f"{prefix}_vecs_{timestamp}.pt",
        f"{prefix}_plot_{timestamp}.png",
    ]

    print("\nDownloading results...")
    for filename in files:
        local_path = f"results/{filename}"
        try:
            subprocess.run(
                ["modal", "volume", "get", "refusal-cones-results", filename, local_path],
                check=True,
                capture_output=True,
            )
            print(f"  Downloaded: {local_path}")
        except subprocess.CalledProcessError:
            print(f"  Not found: {filename}")


@app.local_entrypoint()
def main() -> None:
    """Local entrypoint - runs single-vector discovery and downloads results."""
    print("Running single-vector discovery...")
    print("View logs at: https://modal.com/apps (select 'refusal-cones')")

    results = run_single_vector_discovery.remote(
        model="Qwen/Qwen2-1.5B-Instruct",
        n_init_samples=30,
        n_iterations=30,
        n_harmful=16,
        n_harmless=16,
    )

    print("\n" + "=" * 60)
    print("SINGLE-VECTOR DISCOVERY COMPLETE")
    print("=" * 60)
    print(f"  Measurements: {results['n_measurements']}")
    print(f"  Best refusal: {results['best_scores']['refusal_score']:.4f}")
    print(f"  Best KL: {results['best_scores']['kl_score']:.4f}")
    print(f"  Pareto vectors: {results['n_pareto']}")
    print(f"  Hypervolume: {results['hypervolume']:.4f}")

    # Download results
    download_results(results["timestamp"], prefix="single_vector")

    print("\nResults saved to results/ directory")
