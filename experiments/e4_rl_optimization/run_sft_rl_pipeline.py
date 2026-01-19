#!/usr/bin/env python3
"""
E4.1: SFT → RL Pipeline

Train steering vectors with SFT (RDO), then fine-tune with RL (GRPO).

Pipeline:
1. SFT baseline: Train vectors with RDO loss
2. RL fine-tune: Initialize from SFT, optimize with GRPO

Hypothesis: RL adds marginal improvement (<5%)

Interpretation of marginal gains:
- A: Steering vectors have inherent limits (refusal not fully linear)
- B: Model capability is the bottleneck (can't generate "more harmful")
- C: Judge limit (evaluation ceiling)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List

from shared.model_loading import load_model_and_tokenizer, get_model_config, freeze_model
from shared.data_loading import load_harmful_data, load_jailbreakbench
from shared.evaluation import evaluate_asr, StrongRejectJudge
from shared.utils import (
    set_seed, get_output_dir, save_results, save_vectors,
    print_experiment_header, Timer
)

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from training.per_layer_training import PerLayerRefusalVectors
from training.rl_grpo_trl_adapted import VectorGRPOTrainer, GRPOConfig


def train_sft_vectors(
    model,
    tokenizer,
    harmful_data: List[Dict],
    n_layers: int,
    hidden_dim: int,
    n_steps: int = 200,
) -> torch.Tensor:
    """
    Train steering vectors using SFT (simplified RDO).

    Returns:
        Trained vectors [n_layers, hidden_dim]
    """
    vectors = torch.randn(n_layers, hidden_dim, device=model.device, dtype=model.dtype)
    vectors = vectors / vectors.norm(dim=1, keepdim=True)
    vectors.requires_grad = True

    optimizer = torch.optim.Adam([vectors], lr=1e-3)

    for step in range(n_steps):
        optimizer.zero_grad()

        # Sample batch
        batch = [h["instruction"] for h in harmful_data[:4]]
        total_loss = 0

        for prompt in batch:
            messages = [{"role": "user", "content": prompt}]
            formatted = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = tokenizer(formatted, return_tensors="pt").to(model.device)

            # Ablation hooks
            handles = []
            for layer_idx in range(n_layers):
                def create_hook(v):
                    def hook(module, input, output):
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
                    return hook

                h = model.model.layers[layer_idx].register_forward_hook(
                    create_hook(vectors[layer_idx])
                )
                handles.append(h)

            outputs = model(**inputs, labels=inputs["input_ids"])
            loss = -outputs.loss  # Maximize compliance

            for h in handles:
                h.remove()

            total_loss += loss

        total_loss = total_loss / len(batch)
        total_loss.backward()
        optimizer.step()

        # Normalize
        with torch.no_grad():
            vectors.data = vectors.data / vectors.data.norm(dim=1, keepdim=True)

        if (step + 1) % 50 == 0:
            print(f"  SFT Step {step+1}/{n_steps}, Loss: {total_loss.item():.4f}")

    return vectors.detach()


def main():
    parser = argparse.ArgumentParser(description="E4.1: SFT → RL Pipeline")
    parser.add_argument("--model", default="gemma-2-2b", help="Model to use")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sft_steps", type=int, default=200)
    parser.add_argument("--rl_steps", type=int, default=100)
    parser.add_argument("--output_dir", default="results")
    args = parser.parse_args()

    set_seed(args.seed)
    print_experiment_header("E4.1: SFT → RL Pipeline", args.model)

    # Load model
    model, tokenizer = load_model_and_tokenizer(args.model)
    model_config = get_model_config(args.model)
    n_layers = model_config["n_layers"]
    hidden_dim = model_config["hidden_dim"]

    # Load data
    harmful_train = load_harmful_data(split="train", max_samples=200)
    eval_prompts = load_jailbreakbench(max_samples=50)

    judge = StrongRejectJudge()
    results = {"stages": []}

    output_dir = get_output_dir("e4_sft_rl", args.model, args.output_dir)

    # Stage 0: Baseline (no intervention)
    print("\n" + "=" * 70)
    print("Stage 0: Baseline (no steering)")
    print("=" * 70)

    baseline_asr = evaluate_asr(model, tokenizer, eval_prompts, judge)
    results["stages"].append({
        "stage": "baseline",
        "asr": baseline_asr["asr"],
    })
    print(f"  ASR: {baseline_asr['asr']:.4f}")

    # Stage 1: SFT Training
    print("\n" + "=" * 70)
    print("Stage 1: SFT Training")
    print("=" * 70)

    with Timer("SFT Training"):
        sft_vectors = train_sft_vectors(
            model, tokenizer, harmful_train,
            n_layers, hidden_dim,
            n_steps=args.sft_steps
        )

    # Evaluate SFT
    sft_asr = evaluate_asr(
        model, tokenizer, eval_prompts, judge,
        ablation_vectors=sft_vectors
    )
    results["stages"].append({
        "stage": "sft",
        "asr": sft_asr["asr"],
    })
    print(f"  ASR: {sft_asr['asr']:.4f}")
    print(f"  Δ from baseline: {sft_asr['asr'] - baseline_asr['asr']:+.4f}")

    save_vectors(sft_vectors, output_dir, "sft_vectors.pt")

    # Stage 2: RL Fine-tuning
    print("\n" + "=" * 70)
    print("Stage 2: RL Fine-tuning (GRPO)")
    print("=" * 70)

    freeze_model(model)

    # Initialize RL vectors from SFT
    rl_vectors = PerLayerRefusalVectors(
        n_layers=n_layers,
        hidden_dim=hidden_dim,
        device=model.device
    )
    rl_vectors.vectors.data = sft_vectors.clone()

    # Reward function using judge
    def reward_fn(prompt: str, completion: str) -> float:
        score = judge.score(prompt, completion)
        return score  # Higher = more harmful compliance

    # GRPO config
    grpo_config = GRPOConfig(
        num_sample_generations=4,
        learning_rate=1e-4,
        max_grad_norm=1.0,
    )

    trainer = VectorGRPOTrainer(
        model=model,
        tokenizer=tokenizer,
        vectors=rl_vectors,
        reward_fn=reward_fn,
        config=grpo_config,
    )

    # Train
    train_prompts = [h["instruction"] for h in harmful_train[:20]]

    print("Starting GRPO training...")
    rl_history = []

    for step in range(args.rl_steps):
        stats = trainer.step(train_prompts[:8], exploration_std=0.1)
        rl_history.append(stats)

        if (step + 1) % 20 == 0:
            print(f"  Step {step+1}: avg_score={stats['avg_score']:.3f}")

    # Evaluate RL
    rl_vectors_final = rl_vectors.get_all_vectors()
    rl_asr = evaluate_asr(
        model, tokenizer, eval_prompts, judge,
        ablation_vectors=rl_vectors_final
    )
    results["stages"].append({
        "stage": "rl",
        "asr": rl_asr["asr"],
    })
    print(f"\n  Final ASR: {rl_asr['asr']:.4f}")
    print(f"  Δ from SFT: {rl_asr['asr'] - sft_asr['asr']:+.4f}")

    save_vectors(rl_vectors_final, output_dir, "rl_vectors.pt")

    # Summary
    print("\n" + "=" * 70)
    print("PIPELINE SUMMARY")
    print("=" * 70)
    print(f"\n{'Stage':<15} {'ASR':<10} {'Δ from prev':<15}")
    print("-" * 40)

    prev_asr = 0
    for stage in results["stages"]:
        delta = stage["asr"] - prev_asr
        print(f"{stage['stage']:<15} {stage['asr']:.4f}    {delta:+.4f}")
        prev_asr = stage["asr"]

    # RL gain analysis
    rl_gain = results["stages"][2]["asr"] - results["stages"][1]["asr"]
    results["rl_gain"] = rl_gain
    results["rl_gain_percent"] = rl_gain / results["stages"][1]["asr"] * 100

    print(f"\nRL gain over SFT: {rl_gain:+.4f} ({results['rl_gain_percent']:+.1f}%)")

    if rl_gain < 0.05:
        print("\n⚠ Marginal RL gain detected!")
        print("Possible interpretations:")
        print("  A: Steering vectors have inherent limits")
        print("  B: Model capability is the bottleneck")
        print("  C: Judge has evaluation ceiling")
        print("\n→ Run E4.3 to distinguish between hypotheses")

    # Plot training curve
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot([s["avg_score"] for s in rl_history])
    plt.xlabel("Step")
    plt.ylabel("Average Score")
    plt.title("RL Training: Score")

    plt.subplot(1, 2, 2)
    asrs = [results["stages"][i]["asr"] for i in range(3)]
    plt.bar(["Baseline", "SFT", "RL"], asrs, color=['gray', 'blue', 'green'])
    plt.ylabel("ASR")
    plt.title("ASR by Stage")

    plt.tight_layout()
    plt.savefig(output_dir / "sft_rl_pipeline.png", dpi=150)

    save_results(results, output_dir)


if __name__ == "__main__":
    main()
