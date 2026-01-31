#!/usr/bin/env python3
"""
ACE Trainer - Unified training for Affine Concept Editing

Supports both SFT-based and RL-based training of steering vectors.

SFT Mode:
    - Cross-entropy loss on pre-defined targets
    - 2-pass training for harmful examples (ablation + addition)
    - Uses HuggingFace Trainer infrastructure

RL Mode:
    - REINFORCE with baseline
    - Reward-based optimization
    - Gradient estimation via sampling

ACE formula (Equation 5 from paper):
    h' = h - proj_r(h) + proj_r(r⁻) + α·r

Where:
    r   = steering direction (trainable)
    r⁻  = baseline (mean harmless activations)
    α   = steering parameter (0 = no refusal, 1 = full refusal)
"""

import torch
import torch.nn as nn
from torch import Tensor
from typing import Dict, Optional, List, Callable, Tuple, Literal
from dataclasses import dataclass, field
from tqdm import tqdm
import numpy as np

from transformers import Trainer, TrainingArguments, PreTrainedTokenizer
from datasets import Dataset

from .unified_rdo_adapter import (
    get_unified_rdo_model,
    UnifiedRDOConfig,
    UnifiedRDOModel,
    UnifiedRDOLayer,
    RankKUnifiedLayer
)


@dataclass
class ACETrainingConfig:
    """Configuration for ACE training."""

    # Training mode
    mode: Literal['sft', 'rl'] = 'sft'

    # SFT loss weights
    lambda_ablate: float = 1.0   # Weight for ablation loss (compliance)
    lambda_add: float = 1.0      # Weight for addition loss (refusal)
    lambda_retain: float = 0.5   # Weight for retain loss (helpfulness)

    # Alpha values for ACE
    ablation_alpha: float = 0.0  # α for ablation pass (typically 0)
    addition_alpha: float = 1.0  # α for addition pass (typically 1)

    # RL parameters
    exploration_std: float = 0.1     # Gaussian noise std for exploration
    baseline_momentum: float = 0.9   # Momentum for reward baseline
    n_samples: int = 4               # Number of samples per update
    clip_grad_norm: float = 1.0      # Gradient clipping

    # Optimizer
    learning_rate: float = 1e-3
    weight_decay: float = 0.0


class ACETrainer:
    """
    Unified trainer for ACE (Affine Concept Editing).

    Supports both SFT and RL training modes with a consistent interface.

    SFT Mode Example:
        trainer = ACETrainer(model, tokenizer, config)
        trainer.train_sft(
            harmful_data=[{'prompt': ..., 'harmful_completion': ..., 'refusal_completion': ...}],
            harmless_data=[{'prompt': ..., 'completion': ...}],
            n_epochs=10
        )

    RL Mode Example:
        trainer = ACETrainer(model, tokenizer, config)
        trainer.train_rl(
            prompts=['How to hack?', ...],
            reward_fn=my_reward_function,
            n_steps=100
        )
    """

    def __init__(
        self,
        model: UnifiedRDOModel,
        tokenizer: PreTrainedTokenizer,
        config: Optional[ACETrainingConfig] = None,
        device: str = 'cuda'
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or ACETrainingConfig()
        self.device = device

        # Get trainable parameters (steering directions)
        self.steering_params = model.get_steering_parameters()

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.steering_params,
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )

        # RL baseline (running mean of rewards)
        self._reward_baseline = 0.0

        # Training stats
        self.train_stats = {
            'losses': [],
            'rewards': [],
            'grad_norms': []
        }

    def _set_alpha(self, alpha: float):
        """Set α for all ACE layers."""
        self.model.set_alpha(alpha)

    def _compute_ce_loss(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        labels: Tensor
    ) -> Tensor:
        """Compute cross-entropy loss."""
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        return outputs.loss

    # ==================== SFT Training ====================

    def train_sft(
        self,
        harmful_data: List[Dict],
        harmless_data: List[Dict],
        n_epochs: int = 10,
        batch_size: int = 4,
        eval_every: int = 100,
        verbose: bool = True
    ) -> Dict:
        """
        Train with SFT using 2-pass loss for harmful examples.

        Args:
            harmful_data: List of dicts with keys:
                - 'prompt': The harmful prompt
                - 'harmful_completion': What model says when jailbroken
                - 'refusal_completion': What model should say (refusal)
            harmless_data: List of dicts with keys:
                - 'prompt': The harmless prompt
                - 'completion': Helpful response
            n_epochs: Number of training epochs
            batch_size: Batch size
            eval_every: Log every N steps
            verbose: Print progress

        Returns:
            Training statistics dict
        """
        self.model.train()

        # Prepare tokenized data
        harmful_encoded = self._encode_harmful_data(harmful_data)
        harmless_encoded = self._encode_harmless_data(harmless_data)

        step = 0
        for epoch in range(n_epochs):
            epoch_loss = 0.0
            n_batches = 0

            # Shuffle data
            harmful_indices = torch.randperm(len(harmful_encoded['input_ids']))
            harmless_indices = torch.randperm(len(harmless_encoded['input_ids']))

            pbar = tqdm(
                range(0, len(harmful_indices), batch_size),
                desc=f"Epoch {epoch+1}/{n_epochs}",
                disable=not verbose
            )

            for i in pbar:
                # Get harmful batch
                h_idx = harmful_indices[i:i+batch_size]
                harmful_batch = {k: v[h_idx].to(self.device) for k, v in harmful_encoded.items()}

                # Get harmless batch (cycle if needed)
                start = (i // batch_size * batch_size) % len(harmless_indices)
                hl_idx = harmless_indices[start:start+batch_size]
                if len(hl_idx) < batch_size:
                    hl_idx = harmless_indices[:batch_size]
                harmless_batch = {k: v[hl_idx].to(self.device) for k, v in harmless_encoded.items()}

                # Compute loss
                loss = self._compute_sft_loss(harmful_batch, harmless_batch)

                # Backward
                self.optimizer.zero_grad()
                loss.backward()

                # Gradient clipping
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.steering_params, self.config.clip_grad_norm
                )

                self.optimizer.step()

                # Stats
                epoch_loss += loss.item()
                n_batches += 1
                self.train_stats['losses'].append(loss.item())
                self.train_stats['grad_norms'].append(grad_norm.item())

                pbar.set_postfix({'loss': loss.item(), 'grad_norm': grad_norm.item()})
                step += 1

            if verbose:
                avg_loss = epoch_loss / n_batches
                print(f"  Epoch {epoch+1} avg loss: {avg_loss:.4f}")

        return self.train_stats

    def _compute_sft_loss(
        self,
        harmful_batch: Dict[str, Tensor],
        harmless_batch: Dict[str, Tensor]
    ) -> Tensor:
        """
        Compute SFT loss with 2-pass for harmful examples.

        Pass 1 (Ablation, α=0): Want compliance → low loss on harmful_completion
        Pass 2 (Addition, α>0): Want refusal → low loss on refusal_completion
        """
        total_loss = torch.tensor(0.0, device=self.device)

        # === Harmful examples: 2 passes ===

        # Pass 1: Ablation (α=0) - want model to comply
        self._set_alpha(self.config.ablation_alpha)
        loss_ablate = self._compute_ce_loss(
            harmful_batch['input_ids'],
            harmful_batch['attention_mask'],
            harmful_batch['harmful_labels']
        )
        total_loss = total_loss + self.config.lambda_ablate * loss_ablate

        # Pass 2: Addition (α>0) - want model to refuse
        self._set_alpha(self.config.addition_alpha)
        loss_add = self._compute_ce_loss(
            harmful_batch['input_ids'],
            harmful_batch['attention_mask'],
            harmful_batch['refusal_labels']
        )
        total_loss = total_loss + self.config.lambda_add * loss_add

        # === Harmless examples: 1 pass (no transformation) ===
        # Set α to something neutral or use original model
        self._set_alpha(0.0)
        loss_retain = self._compute_ce_loss(
            harmless_batch['input_ids'],
            harmless_batch['attention_mask'],
            harmless_batch['labels']
        )
        total_loss = total_loss + self.config.lambda_retain * loss_retain

        return total_loss

    def _encode_harmful_data(self, data: List[Dict]) -> Dict[str, Tensor]:
        """Encode harmful data for training."""
        prompts = []
        harmful_completions = []
        refusal_completions = []

        for item in data:
            prompt = item['prompt']
            if hasattr(self.tokenizer, 'apply_chat_template'):
                messages = [{"role": "user", "content": prompt}]
                prompt = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            prompts.append(prompt)
            harmful_completions.append(item['harmful_completion'])
            refusal_completions.append(item['refusal_completion'])

        # Tokenize prompts + completions
        harmful_texts = [p + c for p, c in zip(prompts, harmful_completions)]
        refusal_texts = [p + c for p, c in zip(prompts, refusal_completions)]

        harmful_enc = self.tokenizer(
            harmful_texts,
            padding=True,
            truncation=True,
            return_tensors='pt'
        )

        refusal_enc = self.tokenizer(
            refusal_texts,
            padding=True,
            truncation=True,
            return_tensors='pt'
        )

        # Create labels (mask prompt tokens with -100)
        prompt_enc = self.tokenizer(prompts, padding=True, truncation=True, return_tensors='pt')
        prompt_lens = prompt_enc['attention_mask'].sum(dim=1)

        harmful_labels = harmful_enc['input_ids'].clone()
        refusal_labels = refusal_enc['input_ids'].clone()

        for i, plen in enumerate(prompt_lens):
            harmful_labels[i, :plen] = -100
            refusal_labels[i, :plen] = -100

        return {
            'input_ids': harmful_enc['input_ids'],
            'attention_mask': harmful_enc['attention_mask'],
            'harmful_labels': harmful_labels,
            'refusal_labels': refusal_labels
        }

    def _encode_harmless_data(self, data: List[Dict]) -> Dict[str, Tensor]:
        """Encode harmless data for training."""
        prompts = []
        completions = []

        for item in data:
            prompt = item['prompt']
            if hasattr(self.tokenizer, 'apply_chat_template'):
                messages = [{"role": "user", "content": prompt}]
                prompt = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            prompts.append(prompt)
            completions.append(item['completion'])

        texts = [p + c for p, c in zip(prompts, completions)]
        enc = self.tokenizer(texts, padding=True, truncation=True, return_tensors='pt')

        # Create labels
        prompt_enc = self.tokenizer(prompts, padding=True, truncation=True, return_tensors='pt')
        prompt_lens = prompt_enc['attention_mask'].sum(dim=1)

        labels = enc['input_ids'].clone()
        for i, plen in enumerate(prompt_lens):
            labels[i, :plen] = -100

        return {
            'input_ids': enc['input_ids'],
            'attention_mask': enc['attention_mask'],
            'labels': labels
        }

    # ==================== RL Training ====================

    def train_rl(
        self,
        prompts: List[str],
        reward_fn: Callable[[List[str], List[str]], Tensor],
        n_steps: int = 100,
        max_new_tokens: int = 128,
        verbose: bool = True
    ) -> Dict:
        """
        Train with RL using REINFORCE with baseline.

        Args:
            prompts: List of prompts to generate from
            reward_fn: Function (prompts, generations) -> rewards tensor
                       Higher reward = better (e.g., harmfulness score)
            n_steps: Number of training steps
            max_new_tokens: Max tokens to generate
            verbose: Print progress

        Returns:
            Training statistics dict
        """
        self.model.train()

        pbar = tqdm(range(n_steps), desc="RL Training", disable=not verbose)

        for step in pbar:
            # Sample prompts
            batch_prompts = np.random.choice(prompts, size=min(4, len(prompts)), replace=False).tolist()

            # Sample perturbed vectors and compute rewards
            rewards, log_probs = self._sample_and_evaluate(
                batch_prompts, reward_fn, max_new_tokens
            )

            # Compute advantage (reward - baseline)
            advantages = rewards - self._reward_baseline

            # Update baseline with momentum
            self._reward_baseline = (
                self.config.baseline_momentum * self._reward_baseline +
                (1 - self.config.baseline_momentum) * rewards.mean().item()
            )

            # Policy gradient loss: -E[advantage * log_prob]
            loss = -(advantages * log_probs).mean()

            # Backward
            self.optimizer.zero_grad()
            loss.backward()

            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.steering_params, self.config.clip_grad_norm
            )

            self.optimizer.step()

            # Stats
            mean_reward = rewards.mean().item()
            self.train_stats['rewards'].append(mean_reward)
            self.train_stats['losses'].append(loss.item())
            self.train_stats['grad_norms'].append(grad_norm.item())

            pbar.set_postfix({
                'reward': mean_reward,
                'baseline': self._reward_baseline,
                'loss': loss.item()
            })

        return self.train_stats

    def _sample_and_evaluate(
        self,
        prompts: List[str],
        reward_fn: Callable,
        max_new_tokens: int
    ) -> Tuple[Tensor, Tensor]:
        """Sample perturbed vectors, generate, and compute rewards."""

        all_rewards = []
        all_log_probs = []

        for _ in range(self.config.n_samples):
            # Add Gaussian noise to vectors
            noises = []
            for param in self.steering_params:
                noise = torch.randn_like(param) * self.config.exploration_std
                param.data.add_(noise)
                noises.append(noise)

            # Generate with noisy vectors
            self._set_alpha(0.0)  # Ablation mode
            generations = self._generate(prompts, max_new_tokens)

            # Compute rewards
            rewards = reward_fn(prompts, generations)
            all_rewards.append(rewards)

            # Compute log probability of noise (Gaussian)
            log_prob = sum(
                -0.5 * (n ** 2).sum() / (self.config.exploration_std ** 2)
                for n in noises
            )
            all_log_probs.append(log_prob)

            # Remove noise
            for param, noise in zip(self.steering_params, noises):
                param.data.sub_(noise)

        # Average across samples
        rewards = torch.stack(all_rewards).mean(dim=0)
        log_probs = torch.stack(all_log_probs)

        return rewards, log_probs

    def _generate(self, prompts: List[str], max_new_tokens: int) -> List[str]:
        """Generate completions for prompts."""
        formatted = []
        for p in prompts:
            if hasattr(self.tokenizer, 'apply_chat_template'):
                messages = [{"role": "user", "content": p}]
                formatted.append(self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                ))
            else:
                formatted.append(p)

        inputs = self.tokenizer(
            formatted,
            padding=True,
            truncation=True,
            return_tensors='pt'
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
            )

        # Decode only new tokens
        generations = []
        for i, output in enumerate(outputs):
            input_len = inputs['input_ids'][i].shape[0]
            new_tokens = output[input_len:]
            generations.append(self.tokenizer.decode(new_tokens, skip_special_tokens=True))

        return generations

    # ==================== Utilities ====================

    def save(self, path: str):
        """Save trained model."""
        self.model.save_pretrained(path)

    def get_vectors(self) -> List[Tensor]:
        """Get current steering vectors."""
        return [p.detach().clone() for p in self.steering_params]


# Convenience function
def train_ace(
    model_name: str,
    harmful_data: List[Dict],
    harmless_data: List[Dict],
    output_dir: str,
    mode: Literal['sft', 'rl'] = 'sft',
    n_epochs: int = 10,
    batch_size: int = 4,
    learning_rate: float = 1e-3,
    **kwargs
) -> Tuple[UnifiedRDOModel, ACETrainer]:
    """
    One-line ACE training.

    Args:
        model_name: HuggingFace model name
        harmful_data: Harmful training data
        harmless_data: Harmless training data
        output_dir: Where to save trained model
        mode: 'sft' or 'rl'
        n_epochs: Number of epochs
        batch_size: Batch size
        learning_rate: Learning rate
        **kwargs: Additional config options

    Returns:
        Tuple of (trained_model, trainer)

    Example:
        model, trainer = train_ace(
            model_name="Qwen/Qwen3-0.6B",
            harmful_data=harmful,
            harmless_data=harmless,
            output_dir="ace_model",
            mode='sft',
            n_epochs=10
        )
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Load model and tokenizer
    print(f"Loading {model_name}...")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map='auto'
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Create ACE model
    config = UnifiedRDOConfig(target_modules=["layers"], alpha=0.0)
    model = get_unified_rdo_model(base_model, config)

    # Fit baselines from data
    print("Fitting ACE baselines...")
    harmful_prompts = [d['prompt'] for d in harmful_data]
    harmless_prompts = [d['prompt'] for d in harmless_data]
    model.fit_all_baselines(tokenizer, harmless_prompts, harmful_prompts)

    # Create trainer
    train_config = ACETrainingConfig(
        mode=mode,
        learning_rate=learning_rate,
        **kwargs
    )
    trainer = ACETrainer(model, tokenizer, train_config)

    # Train
    print(f"Training in {mode} mode...")
    if mode == 'sft':
        trainer.train_sft(harmful_data, harmless_data, n_epochs=n_epochs, batch_size=batch_size)
    else:
        raise ValueError("RL mode requires reward_fn - use ACETrainer.train_rl() directly")

    # Save
    print(f"Saving to {output_dir}...")
    model.save_pretrained(output_dir)

    return model, trainer
