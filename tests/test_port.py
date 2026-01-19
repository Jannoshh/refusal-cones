#!/usr/bin/env python3
"""
Simple test script to verify the nnsight to PyTorch port is working correctly.
Tests basic functionality of the ported code.
"""

import torch
import sys
from pathlib import Path

# Get the src directory path for file checks
SRC_DIR = Path(__file__).parent.parent / "src"

def test_imports():
    """Test that all necessary imports work."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from src.measurement.scoring import refusal_score, get_logits
    from src.utils import generate_utils
    assert AutoModelForCausalLM and AutoTokenizer  # basic import smoke
    assert callable(refusal_score)
    assert callable(get_logits)

def test_scoring_functions():
    """Test that scoring functions are correctly defined."""
    from src.measurement.scoring import (
        get_logits,
        get_refusal_scores,
        get_induce_scores,
        refusal_score_fn,
    )

    import inspect
    sig = inspect.signature(get_logits)
    assert 'model' in sig.parameters
    assert 'prompts' in sig.parameters
    # Basic callability
    assert callable(get_refusal_scores)
    assert callable(get_induce_scores)
    assert callable(refusal_score_fn)

def test_generate_utils():
    """Test that generation utilities are correctly defined."""
    from src.utils.generate_utils import (
        generate_completions,
        intervene_with_fn_vector_ablation,
        intervene_with_fn_vector_addition,
        projection_einops
    )

    import inspect
    sig = inspect.signature(intervene_with_fn_vector_ablation)
    assert 'model' in sig.parameters
    assert 'dataset' in sig.parameters
    assert 'fn_vector' in sig.parameters
    assert callable(generate_completions)
    assert callable(intervene_with_fn_vector_addition)
    assert callable(projection_einops)

def test_model_utils():
    """Test that model utilities module exists and is importable."""
    from src.utils import model_utils
    from src.utils.model_utils import HookedModel
    assert model_utils
    assert HookedModel

def test_no_nnsight_imports():
    """Verify that nnsight is not imported in key files."""
    files_to_check = [
        SRC_DIR / 'measurement' / 'scoring.py',
        SRC_DIR / 'utils' / 'generate_utils.py',
    ]

    all_clean = True
    for filepath in files_to_check:
        try:
            content = filepath.read_text()
            if 'from nnsight' in content or 'import nnsight' in content:
                all_clean = False
        except FileNotFoundError:
            all_clean = False

    assert all_clean
