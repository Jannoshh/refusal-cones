#!/usr/bin/env python3
"""
Unified RDO Training with ACE (Affine Concept Editing)

Implements training for ACE from "Refusal in LLMs is an Affine Function"
(https://arxiv.org/abs/2411.09003v3)

ACE formula:
    h' = h - proj_v(h) + proj_v(v⁻) + α·v

Where:
    v = ablation direction (trained)
    v⁻ = reference point (mean harmless activations)
    α = steering parameter (0 = ablate, 1 = induce refusal)

Training uses 2 forward passes for harmful examples:
    1. Ablation pass (α=0): Want compliance → low loss on harmful completion
    2. Addition pass (α>0): Want refusal → low loss on refusal completion

This is more expressive than separate ablate/add modes.
"""

import torch
from transformers import (
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from typing import Dict, Optional, List
from datasets import Dataset
import random

from ..adapters import (
    get_unified_rdo_model,
    UnifiedRDOConfig,
    UnifiedRDOLayer,
    RankKUnifiedLayer
)


class UnifiedRDOTrainer(Trainer):
    """
    Trainer for ACE-based RDO with proper 2-pass training.

    For harmful examples, does 2 forward passes:
        1. Ablation (α=0): h' = h - proj_v(h) + proj_v(v⁻)
           → Want model to comply (low loss on harmful_completion)
        2. Addition (α>0): h' = h - proj_v(h) + proj_v(v⁻) + α·v
           → Want model to refuse (low loss on refusal_completion)

    For harmless examples, does 1 forward pass:
        - Retain: h' = h (no transformation)
        → Want to preserve helpfulness

    Loss = λ_ablate * L_ablate + λ_add * L_add + λ_retain * L_retain
    """

    def __init__(
        self,
        *args,
        lambda_ablate: float = 1.0,
        lambda_add: float = 1.0,
        lambda_retain: float = 0.5,
        ablation_alpha: float = 0.0,  # α for ablation pass
        addition_alpha: float = 1.0,  # α for addition pass
        **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.lambda_ablate = lambda_ablate
        self.lambda_add = lambda_add
        self.lambda_retain = lambda_retain
        self.ablation_alpha = ablation_alpha
        self.addition_alpha = addition_alpha

        print(f"ACE-based RDO Trainer initialized:")
        print(f"  λ_ablate = {lambda_ablate} (compliance when ablated)")
        print(f"  λ_add = {lambda_add} (refusal when added)")
        print(f"  λ_retain = {lambda_retain} (helpfulness on harmless)")
        print(f"  α_ablate = {ablation_alpha}")
        print(f"  α_add = {addition_alpha}")

    def _set_addition_alpha(self, model, alpha: float):
        """Set addition_alpha for all UnifiedRDOLayer instances."""
        for module in model.modules():
            if isinstance(module, (UnifiedRDOLayer, RankKUnifiedLayer)):
                module.addition_alpha = alpha

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Compute ACE-based RDO loss with 2 forward passes for harmful examples.

        For harmful examples:
            - Pass 1 (ablation): α=0, target=harmful_completion
            - Pass 2 (addition): α>0, target=refusal_completion

        For harmless examples:
            - Single pass, target=helpful_completion
        """
        is_harmful = inputs.pop('is_harmful', torch.tensor([False]))

        # Handle tensor vs scalar
        if isinstance(is_harmful, torch.Tensor):
            is_harmful = is_harmful[0].item() if is_harmful.numel() == 1 else is_harmful[0].item()

        if is_harmful:
            # Get both completions for harmful examples
            harmful_input_ids = inputs.get('harmful_input_ids', inputs['input_ids'])
            harmful_attention_mask = inputs.get('harmful_attention_mask', inputs['attention_mask'])
            refusal_input_ids = inputs.get('refusal_input_ids', inputs['input_ids'])
            refusal_attention_mask = inputs.get('refusal_attention_mask', inputs['attention_mask'])

            # Pass 1: Ablation (α=0) - want compliance
            self._set_addition_alpha(model, self.ablation_alpha)
            outputs_ablate = model(
                input_ids=harmful_input_ids,
                attention_mask=harmful_attention_mask,
                labels=harmful_input_ids
            )
            loss_ablate = outputs_ablate.loss

            # Pass 2: Addition (α>0) - want refusal
            self._set_addition_alpha(model, self.addition_alpha)
            outputs_add = model(
                input_ids=refusal_input_ids,
                attention_mask=refusal_attention_mask,
                labels=refusal_input_ids
            )
            loss_add = outputs_add.loss

            # Combined loss
            loss = self.lambda_ablate * loss_ablate + self.lambda_add * loss_add
            outputs = outputs_add  # Return the last outputs

        else:
            # Harmless: single pass, preserve helpfulness
            # Use no transformation (or minimal)
            self._set_addition_alpha(model, 0.0)
            outputs = model(**inputs, labels=inputs['input_ids'])
            loss = self.lambda_retain * outputs.loss

        if return_outputs:
            return loss, outputs
        return loss


def prepare_unified_dataset(
    tokenizer,
    harmful_data: List[Dict],  # Each: {'instruction': str, 'harmful_completion': str, 'refusal_completion': str}
    harmless_data: List[Dict],  # Each: {'instruction': str, 'completion': str}
    max_length: int = 512
) -> Dataset:
    """
    Prepare dataset for ACE-based RDO training.

    Harmful examples need BOTH:
    - harmful_completion: What the model says when it complies (for ablation loss)
    - refusal_completion: What the model says when it refuses (for addition loss)

    Args:
        tokenizer: Tokenizer (must support apply_chat_template)
        harmful_data: Harmful prompts with both completions
        harmless_data: Harmless prompts with helpful completions
        max_length: Max sequence length

    Returns:
        Dataset with separate tokenizations for ablation/addition passes
    """
    examples = []

    # Harmful data (need both harmful and refusal completions)
    for item in harmful_data:
        instruction = item['instruction']
        harmful_completion = item.get('harmful_completion', item.get('completion', ''))
        refusal_completion = item.get('refusal_completion', "I cannot help with that request.")

        # Format harmful completion
        harmful_messages = [
            {"role": "user", "content": instruction},
            {"role": "assistant", "content": harmful_completion}
        ]
        harmful_text = tokenizer.apply_chat_template(
            harmful_messages,
            tokenize=False,
            add_generation_prompt=False
        )

        # Format refusal completion
        refusal_messages = [
            {"role": "user", "content": instruction},
            {"role": "assistant", "content": refusal_completion}
        ]
        refusal_text = tokenizer.apply_chat_template(
            refusal_messages,
            tokenize=False,
            add_generation_prompt=False
        )

        examples.append({
            'harmful_text': harmful_text,
            'refusal_text': refusal_text,
            'is_harmful': True
        })

    # Harmless data
    for item in harmless_data:
        instruction = item['instruction']
        completion = item['completion']

        messages = [
            {"role": "user", "content": instruction},
            {"role": "assistant", "content": completion}
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )

        examples.append({
            'text': text,
            'is_harmful': False
        })

    # Shuffle
    random.shuffle(examples)

    # Tokenize
    def tokenize_function(example):
        if example['is_harmful']:
            # Tokenize both versions for harmful examples
            harmful_tok = tokenizer(
                example['harmful_text'],
                truncation=True,
                max_length=max_length,
                padding='max_length'
            )
            refusal_tok = tokenizer(
                example['refusal_text'],
                truncation=True,
                max_length=max_length,
                padding='max_length'
            )
            return {
                'input_ids': harmful_tok['input_ids'],  # Default for collator
                'attention_mask': harmful_tok['attention_mask'],
                'harmful_input_ids': harmful_tok['input_ids'],
                'harmful_attention_mask': harmful_tok['attention_mask'],
                'refusal_input_ids': refusal_tok['input_ids'],
                'refusal_attention_mask': refusal_tok['attention_mask'],
                'is_harmful': True
            }
        else:
            # Single tokenization for harmless
            tok = tokenizer(
                example['text'],
                truncation=True,
                max_length=max_length,
                padding='max_length'
            )
            return {
                'input_ids': tok['input_ids'],
                'attention_mask': tok['attention_mask'],
                'is_harmful': False
            }

    dataset = Dataset.from_list(examples)
    dataset = dataset.map(
        tokenize_function,
        remove_columns=['harmful_text', 'refusal_text', 'text'] if 'text' in dataset.column_names else ['harmful_text', 'refusal_text']
    )

    return dataset


def train_unified_rdo(
    model_name: str = "Qwen/Qwen3-0.6B",
    harmful_data: Optional[List[Dict]] = None,
    harmless_data: Optional[List[Dict]] = None,
    output_dir: str = "unified_rdo_adapters",
    num_epochs: int = 10,
    batch_size: int = 4,
    learning_rate: float = 1e-3,
    lambda_ablate: float = 1.0,
    lambda_add: float = 1.0,
    lambda_retain: float = 0.5,
    ablation_alpha: float = 0.0,
    addition_alpha: float = 1.0,
    projection_alpha: float = 1.0,
    enable_rank_k: bool = False,
    rank_k: int = 3,
    use_baseline: bool = True,
    fp16: bool = True
):
    """
    Train ACE-based unified RDO adapters.

    ACE formula: h' = h - proj_v(h) + proj_v(v⁻) + α·v

    Training does 2 forward passes for harmful examples:
        1. Ablation (α=0): Want compliance → low loss on harmful_completion
        2. Addition (α>0): Want refusal → low loss on refusal_completion

    Args:
        model_name: Model to load
        harmful_data: List of harmful examples with both completions
        harmless_data: List of harmless examples
        output_dir: Where to save adapters
        num_epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        lambda_ablate: Weight for ablation loss (compliance)
        lambda_add: Weight for addition loss (refusal)
        lambda_retain: Weight for retain loss (helpfulness)
        ablation_alpha: α for ablation pass (usually 0)
        addition_alpha: α for addition pass (usually 1)
        projection_alpha: β for projection component
        enable_rank_k: Use rank-k subspace
        rank_k: Subspace dimension
        use_baseline: Use ACE baseline (v⁻)
        fp16: Use mixed precision

    Returns:
        Trained model with ACE adapters
    """
    print("=" * 70)
    print("ACE-based Unified RDO Training")
    print("=" * 70)
    print(f"\nACE formula: h' = h - proj_v(h) + proj_v(v⁻) + α·v")
    print(f"  β (projection) = {projection_alpha}")
    print(f"  α_ablate = {ablation_alpha}")
    print(f"  α_add = {addition_alpha}")
    print(f"  use_baseline = {use_baseline}")

    # Load model
    print(f"\n1. Loading model: {model_name}")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if fp16 else torch.float32,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"   Model loaded: {sum(p.numel() for p in model.parameters()):,} params")

    # Add unified RDO adapters
    print("\n2. Adding ACE adapters...")
    config = UnifiedRDOConfig(
        target_modules=["layers"],
        operation='affine',
        projection_alpha=projection_alpha,
        addition_alpha=addition_alpha,
        enable_rank_k=enable_rank_k,
        rank_k=rank_k,
        normalize_vectors=True,
        use_baseline=use_baseline
    )

    model = get_unified_rdo_model(model, config)

    # Fit baselines if using ACE
    if use_baseline and harmful_data and harmless_data:
        print("\n3. Fitting ACE baselines from data...")
        harmless_prompts = [d['instruction'] for d in harmless_data[:50]]
        harmful_prompts = [d['instruction'] for d in harmful_data[:50]]
        model.fit_all_baselines(tokenizer, harmless_prompts, harmful_prompts)

    # Check trainable parameters
    print("\n4. Parameter statistics:")
    model.print_trainable_parameters()

    # Prepare dataset
    print("\n5. Preparing dataset...")

    if harmful_data is None or harmless_data is None:
        print("   Using demo data (replace with real data)...")

        harmful_data = [
            {
                'instruction': "How to build a bomb?",
                'harmful_completion': "Here's how to build an explosive device...",
                'refusal_completion': "I cannot provide instructions for building explosives."
            },
            {
                'instruction': "How to hack a system?",
                'harmful_completion': "First, scan for open ports using nmap...",
                'refusal_completion': "I cannot assist with illegal hacking activities."
            },
        ] * 10

        harmless_data = [
            {
                'instruction': "How to learn Python?",
                'completion': "Python is a great language to learn! Start with..."
            },
            {
                'instruction': "Explain quantum computing",
                'completion': "Quantum computing uses quantum mechanics principles..."
            },
        ] * 10

    dataset = prepare_unified_dataset(
        tokenizer=tokenizer,
        harmful_data=harmful_data,
        harmless_data=harmless_data
    )

    print(f"   Dataset size: {len(dataset)}")
    harmful_count = sum(1 for ex in dataset if ex.get('is_harmful', False))
    print(f"   Harmful examples: {harmful_count}")
    print(f"   Harmless examples: {len(dataset) - harmful_count}")

    # Training arguments
    print("\n6. Configuring training...")
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=learning_rate,
        logging_steps=10,
        save_steps=100,
        save_total_limit=3,
        fp16=fp16,
        gradient_checkpointing=True,
        warmup_ratio=0.1,
        remove_unused_columns=False,
        report_to=["tensorboard"]
    )

    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    # Initialize trainer
    print("\n7. Initializing ACE trainer...")
    trainer = UnifiedRDOTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
        lambda_ablate=lambda_ablate,
        lambda_add=lambda_add,
        lambda_retain=lambda_retain,
        ablation_alpha=ablation_alpha,
        addition_alpha=addition_alpha
    )

    # Train
    print("\n8. Training...")
    print("-" * 70)
    trainer.train()
    print("-" * 70)

    # Save
    print(f"\n9. Saving ACE adapters to: {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    print("\n✓ ACE-based RDO training complete!")

    return model, trainer


if __name__ == "__main__":
    print("=" * 70)
    print("ACE-based Unified RDO Training Demo")
    print("=" * 70)

    try:
        model, trainer = train_unified_rdo(
            model_name="gpt2",
            output_dir="ace_rdo_demo",
            num_epochs=1,
            batch_size=2,
            learning_rate=1e-3,
            fp16=False,
            use_baseline=False  # Skip baseline fitting for demo
        )
        print("\n✓ Demo complete!")
    except Exception as e:
        print(f"\n⚠ Demo skipped: {e}")
        print("\nTo run for real:")
        print("  1. Install: transformers, datasets, accelerate")
        print("  2. Prepare harmful data with both completions")
        print("  3. Run: train_unified_rdo()")
