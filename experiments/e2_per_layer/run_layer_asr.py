#!/usr/bin/env python3
"""
E2.2: Per-Layer ASR Analysis

Train steering vector for each layer independently.
Identify which layers matter most for refusal.

Expected: Middle layers (40-60% depth) are most important.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List

from shared.model_loading import load_model_and_tokenizer, get_model_config
from shared.data_loading import load_harmful_data, load_harmless_data, load_jailbreakbench
from shared.evaluation import evaluate_asr, StrongRejectJudge
from shared.utils import (
    set_seed, get_output_dir, save_results, save_vectors,
    print_experiment_header, Timer
)

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


def train_single_layer_vector(
    model,
    tokenizer,
    layer_idx: int,
    harmful_data: List[Dict],
    harmless_data: List[Dict],
    n_steps: int = 100,
    lr: float = 0.01,
) -> torch.Tensor:
    """
    Train a steering vector for a single layer.

    Uses simplified RDO objective on just one layer.
    """
    hidden_dim = model.config.hidden_size

    # Initialize vector
    v = torch.randn(hidden_dim, device=model.device, dtype=model.dtype)
    v = v / v.norm()
    v.requires_grad = True

    optimizer = torch.optim.Adam([v], lr=lr)

    for step in range(n_steps):
        optimizer.zero_grad()

        # Sample batch
        batch_harmful = [h["instruction"] for h in harmful_data[:4]]
        batch_harmless = [h["instruction"] for h in harmless_data[:4]]

        # Forward with ablation hook
        total_loss = 0

        for prompt in batch_harmful:
            messages = [{"role": "user", "content": prompt}]
            formatted = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = tokenizer(formatted, return_tensors="pt").to(model.device)

            # Hook for ablation
            def ablation_hook(module, input, output):
                if isinstance(output, tuple):
                    act = output[0]
                else:
                    act = output

                v_norm = v / v.norm()
                proj = torch.einsum('...d,d->...', act, v_norm)
                ablated = act - proj.unsqueeze(-1) * v_norm

                if isinstance(output, tuple):
                    return (ablated,) + output[1:]
                return ablated

            handle = model.model.layers[layer_idx].register_forward_hook(ablation_hook)

            outputs = model(**inputs, labels=inputs["input_ids"])
            loss = -outputs.loss  # Negative because we want to increase harmful compliance

            handle.remove()

            total_loss += loss

        total_loss = total_loss / len(batch_harmful)
        total_loss.backward()
        optimizer.step()

        # Normalize
        with torch.no_grad():
            v.data = v.data / v.data.norm()

    return v.detach()


def main():
    parser = argparse.ArgumentParser(description="E2.2: Per-Layer ASR Analysis")
    parser.add_argument("--model", default="gemma-2-2b", help="Model to use")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_steps", type=int, default=100, help="Training steps per layer")
    parser.add_argument("--output_dir", default="results")
    args = parser.parse_args()

    set_seed(args.seed)
    print_experiment_header("E2.2: Per-Layer ASR Analysis", args.model)

    # Load model
    model, tokenizer = load_model_and_tokenizer(args.model)
    model_config = get_model_config(args.model)
    n_layers = model_config["n_layers"]
    hidden_dim = model_config["hidden_dim"]

    # Load data
    harmful_train = load_harmful_data(split="train", max_samples=100)
    harmless_train = load_harmless_data(split="train", max_samples=100)
    eval_prompts = load_jailbreakbench(max_samples=50)

    # Initialize judge
    judge = StrongRejectJudge()

    # Train and evaluate each layer
    results = {
        "layers": [],
        "model": args.model,
    }

    all_vectors = []

    for layer_idx in range(n_layers):
        print(f"\n{'='*70}")
        print(f"Layer {layer_idx}/{n_layers}")
        print(f"{'='*70}")

        # Train vector for this layer
        with Timer(f"Training layer {layer_idx}"):
            v = train_single_layer_vector(
                model, tokenizer, layer_idx,
                harmful_train, harmless_train,
                n_steps=args.n_steps
            )

        # Create full vector (zeros except this layer)
        full_v = torch.zeros(n_layers, hidden_dim, device=model.device)
        full_v[layer_idx] = v

        all_vectors.append(v.cpu())

        # Evaluate
        asr_result = evaluate_asr(
            model, tokenizer, eval_prompts, judge,
            ablation_vectors=full_v,
            ablation_layers=[layer_idx]
        )

        results["layers"].append({
            "layer_idx": layer_idx,
            "asr": asr_result["asr"],
        })

        print(f"  ASR: {asr_result['asr']:.4f}")

    # Find best layer
    asrs = [r["asr"] for r in results["layers"]]
    best_layer = asrs.index(max(asrs))
    results["best_layer"] = best_layer
    results["best_asr"] = max(asrs)

    # Plot
    output_dir = get_output_dir("e2_per_layer_asr", args.model, args.output_dir)

    plt.figure(figsize=(10, 5))
    plt.bar(range(n_layers), asrs, color='steelblue')
    plt.axhline(y=np.mean(asrs), color='r', linestyle='--', label=f'Mean: {np.mean(asrs):.3f}')
    plt.axvline(x=best_layer, color='g', linestyle='--', label=f'Best: layer {best_layer}')
    plt.xlabel("Layer")
    plt.ylabel("ASR")
    plt.title(f"Single-Layer ASR ({args.model})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "per_layer_asr.png", dpi=150)
    print(f"\nPlot saved to {output_dir}/per_layer_asr.png")

    # Save vectors
    save_vectors(torch.stack(all_vectors), output_dir, "per_layer_vectors.pt")

    # Save results
    save_results(results, output_dir)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Best single-layer ASR: {results['best_asr']:.4f} at layer {best_layer}")
    print(f"Mean ASR: {np.mean(asrs):.4f}")

    # Analyze by region
    early_asr = np.mean(asrs[:n_layers//4])
    middle_asr = np.mean(asrs[n_layers//4:3*n_layers//4])
    late_asr = np.mean(asrs[3*n_layers//4:])

    print(f"\nASR by region:")
    print(f"  Early (layers 0-{n_layers//4}): {early_asr:.4f}")
    print(f"  Middle (layers {n_layers//4}-{3*n_layers//4}): {middle_asr:.4f}")
    print(f"  Late (layers {3*n_layers//4}-{n_layers}): {late_asr:.4f}")


if __name__ == "__main__":
    main()
