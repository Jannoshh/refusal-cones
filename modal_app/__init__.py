"""
Modal app for refusal-cones GPU workloads.

This package provides Modal functions for running discovery, evaluation,
and training workloads on cloud GPUs.

Usage:
    # Authenticate (one-time setup)
    modal token new

    # Run discovery experiment
    modal run modal_app.py::run_discovery --model "Qwen/Qwen3-0.6B"

    # Run Pareto discovery
    modal run modal_app.py::run_pareto_discovery --model "Qwen/Qwen3-0.6B"

    # Deploy as serverless endpoint
    modal deploy modal_app.py
"""

# Re-export the app and all Modal functions for backwards compatibility

# Core app and configuration
from modal_app.config import (
    DEFAULT_N_HARMFUL,
    DEFAULT_N_HARMLESS,
    GPU_CONFIG,
    app,
    cuda_image,
    model_cache,
    results_volume,
)

# Utility functions
from modal_app.utils import (
    get_refusal_tokens,
    load_prompts,
    make_serializable,
    setup_modal_environment,
    to_list,
)

# Discovery functions
from modal_app.discovery import run_discovery

# Evaluation functions
from modal_app.evaluation import run_batch_evaluation, run_evaluation

# Pareto discovery functions
from modal_app.pareto import (
    run_high_budget_discovery,
    run_pareto_ablations,
    run_pareto_discovery,
)

# Boundary discovery functions
from modal_app.boundary import (
    run_boundary_only,
    run_boundary_then_pareto,
    run_pareto_from_cache,
)

# Single-layer and single-vector discovery functions
from modal_app.single_layer import (
    run_single_layer_discovery,
    run_single_vector_discovery,
    test_single_layer_refusal,
)

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
    "to_list",
    "make_serializable",
    "setup_modal_environment",
    "get_refusal_tokens",
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
