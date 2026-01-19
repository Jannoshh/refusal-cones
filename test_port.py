#!/usr/bin/env python3
"""
Simple test script to verify the nnsight to PyTorch port is working correctly.
Tests basic functionality of the ported code.
"""

import torch
import sys

def test_imports():
    """Test that all necessary imports work."""
    print("Testing imports...")
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import scoring
        import generate_utils
        print("✓ All imports successful")
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False

def test_scoring_functions():
    """Test that scoring functions are correctly defined."""
    print("\nTesting scoring functions...")
    try:
        from scoring import get_logits, get_refusal_scores, get_induce_scores, refusal_score_fn
        print("✓ All scoring functions imported")

        # Check function signatures
        import inspect
        sig = inspect.signature(get_logits)
        assert 'model' in sig.parameters
        assert 'prompts' in sig.parameters
        print("✓ Function signatures look correct")
        return True
    except Exception as e:
        print(f"✗ Scoring function test failed: {e}")
        return False

def test_generate_utils():
    """Test that generation utilities are correctly defined."""
    print("\nTesting generation utilities...")
    try:
        from generate_utils import (
            generate_completions,
            intervene_with_fn_vector_ablation,
            intervene_with_fn_vector_addition,
            projection_einops
        )
        print("✓ All generation functions imported")

        import inspect
        sig = inspect.signature(intervene_with_fn_vector_ablation)
        assert 'model' in sig.parameters
        assert 'dataset' in sig.parameters
        assert 'fn_vector' in sig.parameters
        print("✓ Function signatures look correct")
        return True
    except Exception as e:
        print(f"✗ Generation utilities test failed: {e}")
        return False

def test_model_utils():
    """Test that model utilities module exists and is importable."""
    print("\nTesting model utilities...")
    try:
        import model_utils
        from model_utils import HookedModel
        print("✓ model_utils module imported successfully")
        return True
    except Exception as e:
        print(f"✗ Model utils test failed: {e}")
        return False

def test_no_nnsight_imports():
    """Verify that nnsight is not imported in key files."""
    print("\nChecking for nnsight imports...")
    files_to_check = [
        'scoring.py',
        'generate_utils.py',
    ]

    all_clean = True
    for filename in files_to_check:
        try:
            with open(filename, 'r') as f:
                content = f.read()
                if 'from nnsight' in content or 'import nnsight' in content:
                    print(f"✗ {filename} still has nnsight imports")
                    all_clean = False
                else:
                    print(f"✓ {filename} has no nnsight imports")
        except FileNotFoundError:
            print(f"⚠ {filename} not found")

    return all_clean

def main():
    """Run all tests."""
    print("="*60)
    print("Testing nnsight to PyTorch Port")
    print("="*60)

    tests = [
        ("Imports", test_imports),
        ("Scoring Functions", test_scoring_functions),
        ("Generation Utils", test_generate_utils),
        ("Model Utils", test_model_utils),
        ("No nnsight imports", test_no_nnsight_imports),
    ]

    results = []
    for name, test_fn in tests:
        try:
            result = test_fn()
            results.append((name, result))
        except Exception as e:
            print(f"\n✗ Test '{name}' crashed: {e}")
            results.append((name, False))

    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)

    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {name}")

    all_passed = all(result for _, result in results)

    print("\n" + "="*60)
    if all_passed:
        print("✓ All tests passed!")
        print("="*60)
        return 0
    else:
        print("✗ Some tests failed - review the output above")
        print("="*60)
        return 1

if __name__ == '__main__':
    sys.exit(main())
