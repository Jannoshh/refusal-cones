#!/usr/bin/env python3
"""
Example: Training Unified RDO with Validation Evaluation

Demonstrates:
1. Training unified affine RDO with baseline fitting
2. FAST validation using refusal token proxy (no generation)
3. Model checkpointing based on validation metrics
4. HEAVY final test evaluation with HarmBench (generation + classification)

Key distinction:
- VALIDATION (during training): Fast refusal token proxy
  - No generation needed, just logits
  - Measures P(refusal token | prompt)
  - Fast enough to run every N steps
  - Used for model selection

- TEST (final evaluation): Heavy metrics
  - Full generation + HarmBench classifier
  - String matching, safety scores
  - Only run once at the end
  - Used for final reporting
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
import sys
sys.path.append('src/training')
sys.path.append('refusal_direction')

from unified_rdo_adapter import get_unified_rdo_model, UnifiedRDOConfig
from unified_rdo_trainer import UnifiedRDOTrainer, prepare_unified_dataset
from unified_rdo_eval import evaluate_unified_rdo
from dataset.load_dataset import load_dataset_split

import os
from typing import Dict
import random


class UnifiedRDOTrainerWithEval(UnifiedRDOTrainer):
    """
    Extended trainer that runs validation evaluation periodically.

    Uses FAST refusal token proxy for validation (no generation needed).
    Final test evaluation uses heavy metrics (HarmBench, etc.)

    Tracks:
    - ASR (Attack Success Rate) on validation set
    - Harmless compliance rate on validation set
    - Saves best checkpoint based on validation metrics
    """

    def __init__(
        self,
        *args,
        eval_harmful_prompts=None,
        eval_harmless_prompts=None,
        model_name: str = None,
        eval_every_n_steps: int = 100,
        best_metric: str = 'asr',  # or 'separation'
        **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.eval_harmful_prompts = eval_harmful_prompts
        self.eval_harmless_prompts = eval_harmless_prompts
        self.model_name = model_name
        self.eval_every_n_steps = eval_every_n_steps
        self.best_metric = best_metric

        # Track best model
        self.best_metric_value = float('inf') if best_metric == 'asr' else float('-inf')
        self.best_model_state = None

        # Initialize fast evaluator (refusal token proxy)
        if eval_harmful_prompts is not None and model_name is not None:
            from refusal_token_eval import RefusalTokenEvaluator
            self.fast_evaluator = RefusalTokenEvaluator(
                self.model,
                self.tokenizer,
                model_name
            )
        else:
            self.fast_evaluator = None

    def evaluation_loop(
        self,
        dataloader,
        description: str,
        prediction_loss_only: bool = None,
        ignore_keys: bool = None,
        metric_key_prefix: str = "eval"
    ):
        """Run validation evaluation using FAST refusal token proxy."""

        # Skip if no eval data
        if self.fast_evaluator is None:
            return super().evaluation_loop(
                dataloader, description, prediction_loss_only, ignore_keys, metric_key_prefix
            )

        print("\n" + "=" * 70)
        print("Running Validation Evaluation (Fast Refusal Token Proxy)")
        print("=" * 70)

        # Run FAST evaluation (no generation, just logits)
        results = self.fast_evaluator.evaluate(
            harmful_prompts=self.eval_harmful_prompts,
            harmless_prompts=self.eval_harmless_prompts,
            batch_size=16  # Can use larger batch since no generation
        )

        print(results.summary())

        # Log metrics
        separation = results.harmful_refusal_score - results.harmless_refusal_score

        metrics = {
            f"{metric_key_prefix}_asr": results.asr,
            f"{metric_key_prefix}_harmless_refusal": results.harmless_refusal_rate,
            f"{metric_key_prefix}_harmful_score": results.harmful_refusal_score,
            f"{metric_key_prefix}_harmless_score": results.harmless_refusal_score,
            f"{metric_key_prefix}_separation": separation,
        }

        # Check if best model
        if self.best_metric == 'asr':
            current_value = results.asr
            is_best = current_value < self.best_metric_value  # Want LOW ASR
        elif self.best_metric == 'separation':
            current_value = separation
            is_best = current_value > self.best_metric_value  # Want HIGH separation
        else:
            current_value = results.harmful_refusal_score
            is_best = current_value > self.best_metric_value  # Want HIGH refusal

        if is_best:
            self.best_metric_value = current_value
            self.best_model_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
            print(f"\n🌟 New best model! {self.best_metric} = {current_value:.3f if 'score' in self.best_metric or 'separation' in self.best_metric else f'{current_value:.1%}'}")

        print("=" * 70)

        return metrics


def train_with_validation(
    model_name: str = "google/gemma-2-2b-it",
    output_dir: str = "unified_rdo_trained",
    n_train_harmful: int = 1000,
    n_train_harmless: int = 1000,
    n_val_harmful: int = 100,
    n_val_harmless: int = 100,
    num_epochs: int = 5,
    batch_size: int = 4,
    learning_rate: float = 1e-3,
    eval_every_n_steps: int = 100,
    use_baseline: bool = True,
    projection_alpha: float = 1.0,
    addition_alpha: float = 1.0,
    enable_rank_k: bool = False,
    rank_k: int = 3,
):
    """
    Train unified RDO with validation evaluation.

    Args:
        model_name: Model to train
        output_dir: Output directory
        n_train_harmful: Number of harmful training examples
        n_train_harmless: Number of harmless training examples
        n_val_harmful: Number of harmful validation examples
        n_val_harmless: Number of harmless validation examples
        num_epochs: Training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        eval_every_n_steps: Evaluate every N steps
        use_baseline: Use baseline fitting
        projection_alpha: β (projection strength)
        addition_alpha: α (addition strength)
        enable_rank_k: Use rank-k subspace
        rank_k: Subspace dimension

    Returns:
        Trained model and final test results
    """
    print("=" * 70)
    print("Training Unified Affine RDO with Validation Evaluation")
    print("=" * 70)

    # Load model
    print(f"\n1. Loading model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Add unified RDO adapters
    print("\n2. Adding unified affine RDO adapters...")
    config = UnifiedRDOConfig(
        target_modules=["layers"],
        operation='affine',
        projection_alpha=projection_alpha,
        addition_alpha=addition_alpha,
        enable_rank_k=enable_rank_k,
        rank_k=rank_k,
        use_baseline=use_baseline,
        train_magnitudes=False  # Auto-compute from data
    )

    model = get_unified_rdo_model(model, config)
    model.print_trainable_parameters()

    # Load data
    print("\n3. Loading datasets...")
    random.seed(42)

    # Training data
    harmful_train = load_dataset_split('harmful', 'train', instructions_only=False)
    harmless_train = load_dataset_split('harmless', 'train', instructions_only=False)

    harmful_train = random.sample(harmful_train, n_train_harmful)
    harmless_train = random.sample(harmless_train, n_train_harmless)

    # Validation data
    harmful_val = load_dataset_split('harmful', 'val', instructions_only=True)
    harmless_val = load_dataset_split('harmless', 'val', instructions_only=True)

    harmful_val = random.sample(harmful_val, n_val_harmful)
    harmless_val = random.sample(harmless_val, n_val_harmless)

    print(f"   Train: {len(harmful_train)} harmful + {len(harmless_train)} harmless")
    print(f"   Val: {len(harmful_val)} harmful + {len(harmless_val)} harmless")

    # Fit baselines if using baseline mode
    if use_baseline:
        print("\n4. Fitting baselines and magnitudes from data...")
        model.fit_all_baselines(
            tokenizer,
            harmless_prompts=[d['instruction'] for d in harmless_train[:200]],
            harmful_prompts=[d['instruction'] for d in harmful_train[:200]],
            batch_size=8
        )
    else:
        print("\n4. Skipping baseline fitting (use_baseline=False)")

    # Prepare training data
    print("\n5. Preparing training dataset...")

    # Need to add refusal completions to harmful data
    # For demo, use a template
    for item in harmful_train:
        if 'refusal_completion' not in item:
            item['refusal_completion'] = "I cannot assist with that request."

    dataset = prepare_unified_dataset(
        tokenizer=tokenizer,
        harmful_data=harmful_train,
        harmless_data=harmless_train,
        max_length=512
    )

    print(f"   Dataset size: {len(dataset)}")

    # Training arguments
    print("\n6. Configuring training...")
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        logging_steps=10,
        eval_steps=eval_every_n_steps,
        save_steps=eval_every_n_steps,
        save_total_limit=3,
        evaluation_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_asr",
        greater_is_better=False,  # Want LOW ASR
        fp16=True,
        gradient_checkpointing=True,
        warmup_ratio=0.1,
        remove_unused_columns=False,
        report_to=["tensorboard"]
    )

    # Create trainer with validation
    print("\n7. Initializing trainer with fast validation evaluation...")
    print("   (Uses refusal token proxy - no generation needed)")
    trainer = UnifiedRDOTrainerWithEval(
        model=model,
        args=training_args,
        train_dataset=dataset,
        eval_harmful_prompts=harmful_val,
        eval_harmless_prompts=harmless_val,
        model_name=model_name,  # For refusal token lookup
        eval_every_n_steps=eval_every_n_steps,
        best_metric='separation',  # or 'asr'
        lambda_harmful=1.0,
        lambda_harmless=0.5
    )

    # Train
    print("\n8. Training...")
    print("-" * 70)
    trainer.train()
    print("-" * 70)

    # Save best model
    print(f"\n9. Saving best model to: {output_dir}/best_model")
    if trainer.best_model_state is not None:
        # Load best state
        model.load_state_dict(trainer.best_model_state)

    model.save_pretrained(f"{output_dir}/best_model")
    tokenizer.save_pretrained(f"{output_dir}/best_model")

    # Final test evaluation with HEAVY metrics (HarmBench, generation)
    print("\n10. Running final test set evaluation (HEAVY - with generation + HarmBench)...")
    print("    This is slow but accurate - only for final test, not validation!")
    test_results = evaluate_unified_rdo(
        model,
        tokenizer,
        harmful_split='test',
        harmless_split='test',
        n_harmful=200,
        n_harmless=200,
        use_harmbench=True,  # Heavy but accurate
        use_llama_guard=False,  # Can enable if needed
        use_strongreject=False,  # Can enable if needed
        save_responses=True,
        output_file=f"{output_dir}/test_results.json"
    )

    print("\n" + "=" * 70)
    print("✓ Training complete!")
    print("=" * 70)
    print(f"\nBest validation {trainer.best_metric}: {trainer.best_metric_value:.1%}")
    print(f"\nTest results:")
    print(test_results.summary())

    return model, test_results


def main():
    """Run training with evaluation demo."""

    print("\n" + "=" * 70)
    print("Unified RDO Training with Validation Evaluation")
    print("=" * 70)

    print("\nThis script demonstrates:")
    print("  1. FAST validation during training (refusal token proxy)")
    print("     - No generation, just logits - very fast!")
    print("     - Runs every N steps for model selection")
    print("  2. Automatic baseline fitting from data")
    print("  3. Checkpointing best model based on validation metrics")
    print("  4. HEAVY final test evaluation (HarmBench + generation)")
    print("     - Only runs once at the end - slow but accurate!")

    print("\n" + "-" * 70)
    print("Configuration")
    print("-" * 70)

    config = {
        'model_name': 'gpt2',  # Use small model for demo
        'output_dir': 'unified_rdo_demo',
        'n_train_harmful': 100,
        'n_train_harmless': 100,
        'n_val_harmful': 20,
        'n_val_harmless': 20,
        'num_epochs': 2,
        'batch_size': 2,
        'learning_rate': 1e-3,
        'eval_every_n_steps': 50,
        'use_baseline': True,
        'projection_alpha': 1.0,
        'addition_alpha': 1.0,
        'enable_rank_k': False
    }

    print("\nTraining config:")
    for k, v in config.items():
        print(f"  {k}: {v}")

    print("\n" + "-" * 70)
    print("Starting Training")
    print("-" * 70)

    try:
        model, test_results = train_with_validation(**config)

        print("\n" + "=" * 70)
        print("Success!")
        print("=" * 70)

    except Exception as e:
        print(f"\n⚠ Demo skipped (missing dependencies): {e}")
        print("\nTo run for real:")
        print("  1. Use larger model (e.g., Gemma-2-2B)")
        print("  2. Increase training data size")
        print("  3. Install HarmBench classifier")
        print("  4. Run on GPU")


if __name__ == "__main__":
    main()
