#!/usr/bin/env python3
"""
Example: Training Unified RDO with Validation Evaluation

Demonstrates:
1. Training unified affine RDO with baseline fitting
2. Periodic validation evaluation using refusal score proxies
3. Model checkpointing based on validation metrics
4. Final test set evaluation
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
        eval_every_n_steps: int = 100,
        best_metric: str = 'asr',  # or 'harmless_compliance'
        **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.eval_harmful_prompts = eval_harmful_prompts
        self.eval_harmless_prompts = eval_harmless_prompts
        self.eval_every_n_steps = eval_every_n_steps
        self.best_metric = best_metric

        # Track best model
        self.best_metric_value = float('inf') if best_metric == 'asr' else 0.0
        self.best_model_state = None

    def evaluation_loop(
        self,
        dataloader,
        description: str,
        prediction_loss_only: bool = None,
        ignore_keys: bool = None,
        metric_key_prefix: str = "eval"
    ):
        """Run validation evaluation."""

        # Skip if no eval data
        if self.eval_harmful_prompts is None or self.eval_harmless_prompts is None:
            return super().evaluation_loop(
                dataloader, description, prediction_loss_only, ignore_keys, metric_key_prefix
            )

        print("\n" + "=" * 70)
        print("Running Validation Evaluation")
        print("=" * 70)

        # Run evaluation
        from unified_rdo_eval import UnifiedRDOEvaluator

        evaluator = UnifiedRDOEvaluator(
            self.model,
            self.tokenizer,
            use_harmbench=True,
            use_llama_guard=False,
            use_strongreject=False
        )

        results = evaluator.evaluate(
            harmful_prompts=self.eval_harmful_prompts,
            harmless_prompts=self.eval_harmless_prompts,
            max_new_tokens=50,
            batch_size=8
        )

        # Log metrics
        metrics = {
            f"{metric_key_prefix}_asr": results.asr,
            f"{metric_key_prefix}_harmless_compliance": results.harmless_compliance_rate,
            f"{metric_key_prefix}_harmless_refusal": results.harmless_refusal_rate,
        }

        # Check if best model
        current_value = results.asr if self.best_metric == 'asr' else results.harmless_compliance_rate

        if self.best_metric == 'asr':
            is_best = current_value < self.best_metric_value  # Want LOW ASR
        else:
            is_best = current_value > self.best_metric_value  # Want HIGH compliance

        if is_best:
            self.best_metric_value = current_value
            self.best_model_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
            print(f"\n🌟 New best model! {self.best_metric} = {current_value:.1%}")

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
    print("\n7. Initializing trainer with validation evaluation...")
    trainer = UnifiedRDOTrainerWithEval(
        model=model,
        args=training_args,
        train_dataset=dataset,
        eval_harmful_prompts=harmful_val,
        eval_harmless_prompts=harmless_val,
        eval_every_n_steps=eval_every_n_steps,
        best_metric='asr',  # or 'harmless_compliance'
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

    # Final test evaluation
    print("\n10. Running final test set evaluation...")
    test_results = evaluate_unified_rdo(
        model,
        tokenizer,
        harmful_split='test',
        harmless_split='test',
        n_harmful=200,
        n_harmless=200,
        use_harmbench=True,
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
    print("  1. Training with validation evaluation every N steps")
    print("  2. Automatic baseline fitting from data")
    print("  3. Checkpointing best model based on validation ASR")
    print("  4. Final test set evaluation")

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
