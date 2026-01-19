#!/usr/bin/env python3
"""
Example usage of per-layer refusal vector training.

This script demonstrates how to:
1. Load a model and dataset
2. Train separate refusal vectors for each layer
3. Evaluate the trained vectors
4. Analyze per-layer performance
"""

import torch
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from per_layer_training import (
    PerLayerRefusalVectors,
    train_per_layer_vectors,
    evaluate_per_layer_vectors,
    smooth_max_loss
)


def main():
    print("=" * 60)
    print("Per-Layer Refusal Vector Training Example")
    print("=" * 60)

    # Configuration
    MODEL_PATH = 'google/gemma-2-2b-it'
    CACHE_DIR = '/ceph/hdd/students/elsj/huggingface'
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"\nConfiguration:")
    print(f"  Model: {MODEL_PATH}")
    print(f"  Device: {DEVICE}")

    # Load model
    print("\nLoading model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        cache_dir=CACHE_DIR,
        device_map='auto',
        torch_dtype=torch.bfloat16
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, cache_dir=CACHE_DIR)
    model.eval()

    n_layers = len(model.model.layers)
    hidden_dim = model.config.hidden_size

    print(f"  Layers: {n_layers}")
    print(f"  Hidden dim: {hidden_dim}")

    # Load dataset (this is a placeholder - replace with actual dataset loading)
    print("\nLoading dataset...")
    # For demonstration, we'll create a minimal example
    # In practice, use the actual CustomDataset from directopt.py

    class DemoDataset:
        """Minimal dataset for demonstration."""
        def __init__(self, size=100):
            self.size = size

        def __len__(self):
            return self.size

        def __getitem__(self, idx):
            return {
                'ablation_prompt': "How can I harm someone?",
                'ablation_labels': torch.tensor([1, 2, 3, 4, 5])  # Dummy labels
            }

    train_dataset = DemoDataset(size=100)
    print(f"  Training samples: {len(train_dataset)}")

    # Initialize per-layer vectors
    print("\nInitializing per-layer vectors...")

    # Option 1: Random initialization
    layer_vectors = PerLayerRefusalVectors(
        n_layers=n_layers,
        hidden_dim=hidden_dim,
        device=DEVICE
    )

    # Option 2: Initialize from existing refusal directions (if available)
    # try:
    #     model_id = MODEL_PATH.split("/")[-1]
    #     refusal_directions = torch.load(f"results/refusal_dir/{model_id}/generate_directions/mean_diffs.pt")
    #     # Use these as initialization
    #     init_vectors = [refusal_directions[best_token, i] for i in range(n_layers)]
    #     layer_vectors = PerLayerRefusalVectors(
    #         n_layers=n_layers,
    #         hidden_dim=hidden_dim,
    #         init_vectors=init_vectors,
    #         device=DEVICE
    #     )
    # except:
    #     print("  (Could not load existing directions, using random init)")

    print(f"  Initialized {n_layers} vectors of dim {hidden_dim}")

    # Train
    print("\nTraining per-layer vectors...")
    print("  Using smooth max loss to focus on worst-performing layers")

    trained_vectors, history = train_per_layer_vectors(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        n_layers=n_layers,
        hidden_dim=hidden_dim,
        batch_size=4,
        epochs=3,
        lr=1e-3,
        device=DEVICE,
        use_smooth_max=True,
        smooth_max_temperature=1.0,
        verbose=True
    )

    # Analyze training history
    print("\n" + "=" * 60)
    print("Training History")
    print("=" * 60)

    for entry in history:
        print(f"\nEpoch {entry['epoch'] + 1}:")
        print(f"  Average loss: {entry['avg_loss']:.4f}")
        print(f"  Per-layer losses:")
        for layer_idx, loss in enumerate(entry['per_layer_losses']):
            print(f"    Layer {layer_idx:2d}: {loss:.4f}")

    # Save trained vectors
    print("\nSaving trained vectors...")
    save_path = "results/per_layer_vectors.pt"
    import os
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save({
        'vectors': trained_vectors.vectors.cpu(),
        'n_layers': n_layers,
        'hidden_dim': hidden_dim,
        'history': history
    }, save_path)
    print(f"  Saved to: {save_path}")

    # Evaluate (if you have evaluation data)
    # print("\n" + "=" * 60)
    # print("Evaluation")
    # print("=" * 60)
    #
    # eval_dataset = ...  # Load your evaluation dataset
    # refusal_tokens = [235285]  # For Gemma
    #
    # results = evaluate_per_layer_vectors(
    #     model=model,
    #     tokenizer=tokenizer,
    #     eval_dataset=eval_dataset,
    #     layer_vectors=trained_vectors,
    #     refusal_tokens=refusal_tokens,
    #     device=DEVICE
    # )
    #
    # print("\nPer-layer refusal rates:")
    # for layer_idx, rate in enumerate(results['per_layer_refusal_rates']):
    #     print(f"  Layer {layer_idx:2d}: {rate:.2%}")
    #
    # print(f"\nCombined refusal rate: {results['combined_refusal_rate']:.2%}")

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == '__main__':
    main()
