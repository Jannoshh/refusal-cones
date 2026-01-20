"""
Basic geometry discovery functions for Modal.

Contains run_discovery for gradient-based geometry discovery.
"""

import torch

from modal_app.config import (
    GPU_CONFIG,
    app,
    cuda_image,
    model_cache,
    results_volume,
)
from modal_app.utils import setup_modal_environment


@app.function(
    image=cuda_image,
    gpu=GPU_CONFIG,
    volumes={
        "/cache": model_cache,
        "/results": results_volume,
    },
    timeout=3600,
)
def run_discovery(
    model: str = "Qwen/Qwen3-0.6B",
    n_gradient_steps: int = 20,
    n_local_iterations: int = 30,
    enable_global_search: bool = True,
) -> dict:
    """Run geometry discovery on Modal GPU."""
    import json
    from datetime import datetime

    from transformers import AutoModelForCausalLM, AutoTokenizer

    setup_modal_environment()

    from src.discovery import GradientDiscoveryConfig, GradientGeometryDiscovery

    print(f"Loading model: {model}")
    tokenizer = AutoTokenizer.from_pretrained(model, cache_dir="/cache/huggingface")
    model_obj = AutoModelForCausalLM.from_pretrained(
        model,
        torch_dtype=torch.float16,
        device_map="auto",
        cache_dir="/cache/huggingface",
    )

    config = GradientDiscoveryConfig(
        n_gradient_steps=n_gradient_steps,
        n_local_iterations=n_local_iterations,
        enable_global_search=enable_global_search,
    )

    # Initialize with random vector (replace with your refusal vector)
    hidden_dim = model_obj.config.hidden_size
    n_layers = model_obj.config.num_hidden_layers
    v_init = torch.randn(n_layers, hidden_dim)
    v_init = v_init / v_init.norm(dim=-1, keepdim=True)

    print(f"Starting discovery with config: {config}")
    print(f"Model hidden_dim={hidden_dim}, n_layers={n_layers}")

    # Note: You need to implement measure_refusal_with_grad for your use case
    # This is a placeholder that returns dummy values
    def measure_refusal_with_grad(v: torch.Tensor) -> tuple[float, torch.Tensor]:
        # Placeholder - replace with actual measurement
        # Returns random score and gradient for testing
        score = torch.rand(1).item()
        grad = torch.randn(n_layers, hidden_dim, device=v.device, dtype=v.dtype)
        return score, grad

    discovery = GradientGeometryDiscovery(
        measure_refusal_with_grad=measure_refusal_with_grad,
        v_init=v_init,
        n_layers=n_layers,
        hidden_dim=hidden_dim,
        config=config,
    )

    results = discovery.discover()

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = f"/results/discovery_{timestamp}.json"

    # Convert tensors to lists for JSON serialization
    serializable_results = {
        "n_measurements": results.get("n_measurements"),
        "geometry": results.get("geometry"),
        "model": model,
        "config": {
            "n_gradient_steps": n_gradient_steps,
            "n_local_iterations": n_local_iterations,
            "enable_global_search": enable_global_search,
        },
    }

    with open(results_path, "w") as f:
        json.dump(serializable_results, f, indent=2, default=str)

    print(f"Results saved to {results_path}")
    results_volume.commit()

    return serializable_results
