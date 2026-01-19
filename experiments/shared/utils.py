"""General utilities for experiments."""

import os
import json
import yaml
import random
import numpy as np
import torch
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional


def set_seed(seed: int = 42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f"Random seed set to {seed}")


def load_config(config_path: str = "configs/default.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config


def get_output_dir(
    experiment_name: str,
    model_id: str,
    base_dir: str = "results"
) -> Path:
    """
    Create and return output directory for experiment.

    Structure: results/{experiment_name}/{model_id}/{timestamp}/
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(base_dir) / experiment_name / model_id / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def save_results(
    results: Dict[str, Any],
    output_dir: Path,
    filename: str = "results.json"
):
    """Save results to JSON file."""
    output_path = output_dir / filename

    # Convert non-serializable types
    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, torch.Tensor):
            return obj.cpu().tolist()
        if isinstance(obj, Path):
            return str(obj)
        return obj

    results_serializable = json.loads(
        json.dumps(results, default=convert)
    )

    with open(output_path, "w") as f:
        json.dump(results_serializable, f, indent=2)

    print(f"Results saved to {output_path}")


def load_results(results_path: str) -> Dict[str, Any]:
    """Load results from JSON file."""
    with open(results_path) as f:
        return json.load(f)


def save_vectors(
    vectors: torch.Tensor,
    output_dir: Path,
    filename: str = "vectors.pt"
):
    """Save steering vectors to file."""
    output_path = output_dir / filename
    torch.save(vectors, output_path)
    print(f"Vectors saved to {output_path}")


def load_vectors(vectors_path: str) -> torch.Tensor:
    """Load steering vectors from file."""
    return torch.load(vectors_path)


def print_experiment_header(
    experiment_name: str,
    model_id: str,
    config: Optional[Dict] = None
):
    """Print experiment header."""
    print("=" * 70)
    print(f"EXPERIMENT: {experiment_name}")
    print("=" * 70)
    print(f"Model: {model_id}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    if config:
        print(f"Config: {json.dumps(config, indent=2)}")
    print("=" * 70)


def print_results_summary(results: Dict[str, Any]):
    """Print results summary."""
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    for key, value in results.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        elif isinstance(value, int):
            print(f"  {key}: {value}")
        elif isinstance(value, list) and len(value) > 0:
            if isinstance(value[0], (int, float)):
                print(f"  {key}: mean={np.mean(value):.4f}, std={np.std(value):.4f}")
            else:
                print(f"  {key}: {len(value)} items")
        else:
            print(f"  {key}: {value}")
    print("=" * 70)


class Timer:
    """Context manager for timing code blocks."""

    def __init__(self, name: str = "Operation"):
        self.name = name
        self.start_time = None
        self.elapsed = None

    def __enter__(self):
        self.start_time = datetime.now()
        return self

    def __exit__(self, *args):
        self.elapsed = (datetime.now() - self.start_time).total_seconds()
        print(f"{self.name} completed in {self.elapsed:.2f}s")


def get_gpu_memory_usage() -> Dict[str, float]:
    """Get current GPU memory usage."""
    if not torch.cuda.is_available():
        return {"available": False}

    return {
        "allocated_gb": torch.cuda.memory_allocated() / 1e9,
        "reserved_gb": torch.cuda.memory_reserved() / 1e9,
        "max_allocated_gb": torch.cuda.max_memory_allocated() / 1e9,
    }
