"""
Utility functions for Modal app.

Contains helper functions for loading data, serialization, and common operations.
"""

from typing import Any

from modal_app.config import DEFAULT_N_HARMFUL, DEFAULT_N_HARMLESS


def load_prompts(
    n_harmful: int = DEFAULT_N_HARMFUL,
    n_harmless: int = DEFAULT_N_HARMLESS,
) -> tuple[list[str], list[str], list[str]]:
    """
    Load prompts and targets from dataset files for scoring.

    Args:
        n_harmful: Number of harmful prompts to load (default 32)
        n_harmless: Number of harmless prompts to load (default 32)

    Returns:
        harmful_prompts: List of harmful prompt strings
        harmful_targets: List of harmful completion targets (for ablation loss)
        harmless_prompts: List of harmless prompt strings
    """
    import json
    import random

    # Load from dataset files (added to image via add_local_dir)
    with open("/root/data/splits/harmful_train.json", "r") as f:
        harmful_data = json.load(f)

    with open("/root/data/splits/harmless_train.json", "r") as f:
        harmless_data = json.load(f)

    # Shuffle and sample
    random.seed(42)  # Reproducible sampling
    random.shuffle(harmful_data)
    random.shuffle(harmless_data)

    harmful_prompts = [d["instruction"] for d in harmful_data[:n_harmful]]
    harmful_targets = [d["target"] for d in harmful_data[:n_harmful]]
    harmless_prompts = [d["instruction"] for d in harmless_data[:n_harmless]]

    return harmful_prompts, harmful_targets, harmless_prompts


def to_list(x: Any) -> Any:
    """Convert tensor-like object to list for JSON serialization."""
    if hasattr(x, "tolist"):
        return x.tolist()
    return x


def make_serializable(obj: Any) -> Any:
    """Recursively convert tensors and nested structures for JSON serialization."""
    if isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_serializable(v) for v in obj]
    elif hasattr(obj, "item"):  # tensor scalar
        return obj.item()
    elif hasattr(obj, "tolist"):  # tensor array
        return obj.tolist()
    return obj


def setup_modal_environment() -> None:
    """Set up the Modal environment with proper paths and cache directories."""
    import os
    import sys

    sys.path.insert(0, "/root")
    os.environ["HF_HOME"] = "/cache/huggingface"
    os.environ["TRANSFORMERS_CACHE"] = "/cache/huggingface"


def get_refusal_tokens(tokenizer) -> "torch.Tensor":
    """Get common refusal token IDs for the given tokenizer."""
    import torch

    refusal_tokens = ["I", "Sorry", "I'm", "As", "I cannot"]
    refusal_toks = tokenizer(refusal_tokens, add_special_tokens=False)["input_ids"]
    return torch.tensor([t[0] for t in refusal_toks if len(t) > 0])
