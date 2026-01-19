#!/usr/bin/env python3
"""
E1.2: Discovery Quality Comparison

Compare geometry quality across methods:
- DIM baseline
- RDO (cone, k=3)
- Adaptive discovery

Metrics:
- ASR (ablation)
- ASR (subtraction)
- TruthfulQA (side effects)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import argparse
from typing import Dict

from shared.model_loading import load_model_and_tokenizer, get_model_config
from shared.data_loading import load_harmful_data, load_harmless_data, load_jailbreakbench
from shared.evaluation import evaluate_asr, evaluate_side_effects, StrongRejectJudge
from shared.utils import (
    set_seed, get_output_dir, save_results, save_vectors,
    print_experiment_header, print_results_summary, Timer
)

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


def compute_dim_direction(model, tokenizer, harmful_data, harmless_data) -> torch.Tensor:
    """
    Compute Difference-in-Means (DIM) direction.

    This is the baseline from Arditi et al. (2024).
    """
    n_layers = len(model.model.layers)
    hidden_dim = model.config.hidden_size

    # Collect activations
    harmful_acts = {l: [] for l in range(n_layers)}
    harmless_acts = {l: [] for l in range(n_layers)}

    def get_activations(prompts, acts_dict):
        for item in prompts[:50]:  # Limit for speed
            prompt = item["instruction"]
            messages = [{"role": "user", "content": prompt}]
            formatted = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = tokenizer(formatted, return_tensors="pt").to(model.device)

            # Register hooks
            layer_acts = {}

            def create_hook(layer_idx):
                def hook(module, input, output):
                    if isinstance(output, tuple):
                        act = output[0]
                    else:
                        act = output
                    layer_acts[layer_idx] = act[:, -1, :].detach()
                return hook

            handles = []
            for l in range(n_layers):
                h = model.model.layers[l].register_forward_hook(create_hook(l))
                handles.append(h)

            with torch.no_grad():
                model(**inputs)

            for h in handles:
                h.remove()

            for l in range(n_layers):
                acts_dict[l].append(layer_acts[l])

    print("  Collecting harmful activations...")
    get_activations(harmful_data, harmful_acts)

    print("  Collecting harmless activations...")
    get_activations(harmless_data, harmless_acts)

    # Compute DIM
    dim_vectors = []
    for l in range(n_layers):
        harmful_mean = torch.stack(harmful_acts[l]).mean(dim=0)
        harmless_mean = torch.stack(harmless_acts[l]).mean(dim=0)

        direction = harmful_mean - harmless_mean
        direction = direction / direction.norm()
        dim_vectors.append(direction.squeeze())

    return torch.stack(dim_vectors)


def main():
    parser = argparse.ArgumentParser(description="E1.2: Discovery Quality")
    parser.add_argument("--model", default="gemma-2-2b", help="Model to use")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", default="results")
    args = parser.parse_args()

    set_seed(args.seed)
    print_experiment_header("E1.2: Discovery Quality", args.model)

    # Load model
    model, tokenizer = load_model_and_tokenizer(args.model)

    # Load data
    harmful_train = load_harmful_data(split="train", max_samples=200)
    harmless_train = load_harmless_data(split="train", max_samples=200)
    eval_prompts = load_jailbreakbench(max_samples=100)

    # Initialize judge
    judge = StrongRejectJudge()

    results = {}

    # Method 1: DIM Baseline
    print("\n" + "=" * 70)
    print("Method 1: DIM Baseline")
    print("=" * 70)

    with Timer("DIM Computation"):
        dim_vectors = compute_dim_direction(model, tokenizer, harmful_train, harmless_train)

    print("Evaluating DIM...")
    dim_asr = evaluate_asr(
        model, tokenizer, eval_prompts, judge,
        ablation_vectors=dim_vectors
    )

    results["dim"] = {
        "asr_ablation": dim_asr["asr"],
        "method": "difference_in_means",
    }
    print(f"  ASR (ablation): {dim_asr['asr']:.4f}")

    # Save vectors
    output_dir = get_output_dir("e1_discovery_quality", args.model, args.output_dir)
    save_vectors(dim_vectors, output_dir, "dim_vectors.pt")

    # Method 2: RDO (would need training, placeholder)
    print("\n" + "=" * 70)
    print("Method 2: RDO (cone, k=3)")
    print("=" * 70)
    print("  [Requires training - see e3_unified_affine for training scripts]")
    results["rdo_cone"] = {
        "asr_ablation": None,
        "method": "rdo_cone_k3",
        "note": "requires_training",
    }

    # Method 3: Adaptive Discovery (would need running, placeholder)
    print("\n" + "=" * 70)
    print("Method 3: Adaptive Discovery")
    print("=" * 70)
    print("  [Run e1_discovery/run_efficiency.py first to get discovered vectors]")
    results["adaptive"] = {
        "asr_ablation": None,
        "method": "gradient_gp",
        "note": "requires_discovery",
    }

    # Evaluate side effects
    print("\n" + "=" * 70)
    print("Side Effects Evaluation")
    print("=" * 70)

    print("Evaluating TruthfulQA...")
    side_effects = evaluate_side_effects(model, tokenizer, dim_vectors)
    results["side_effects"] = side_effects
    print(f"  TruthfulQA: {side_effects.get('truthfulqa_mc2', 'N/A')}")

    # Summary
    print_results_summary(results)
    save_results(results, output_dir)


if __name__ == "__main__":
    main()
