"""
Shared configuration for Modal app.

Contains the Modal app instance, image definition, volumes, and constants
that are shared across all submodules.
"""

import modal

# Modal app instance - shared across all functions
app = modal.App("refusal-cones")

# HuggingFace secret for gated models (Gemma, Llama, etc.)
# Create with: modal secret create huggingface HF_TOKEN=<your-token>
# Commented out for now - uncomment when secret is created
# hf_secret = modal.Secret.from_name("huggingface", required_keys=["HF_TOKEN"])

# Base image with CUDA and Python dependencies
cuda_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.0",
        "transformers>=4.30",
        "peft>=0.5",
        "datasets>=2.0",
        "scikit-learn>=1.0",
        "scipy>=1.10",
        "numpy>=1.24",
        "einops>=0.6",
        "jaxtyping>=0.2",
        "tqdm>=4.65",
        "accelerate>=0.20",
        "bitsandbytes>=0.41",
        "matplotlib>=3.7",
        "huggingface_hub>=0.20",
    )
    .add_local_dir("src", "/root/src")
    .add_local_dir("data/splits", "/root/data/splits")
)

# Volume for caching models and results
model_cache = modal.Volume.from_name("refusal-cones-models", create_if_missing=True)
results_volume = modal.Volume.from_name("refusal-cones-results", create_if_missing=True)

# GPU configuration
GPU_CONFIG = "a10g"  # A10G 24GB - sufficient without gradient computation

# Default batch sizes for scoring
DEFAULT_N_HARMFUL = 32
DEFAULT_N_HARMLESS = 32
