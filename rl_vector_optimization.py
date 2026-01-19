"""
Reinforcement Learning for Refusal Vector Optimization.

This module implements policy gradient methods (REINFORCE, PPO) for optimizing
refusal vectors beyond supervised learning. The RL stage explores the vector
space to find more effective refusal directions that aren't bounded by SFT data.

Key Features:
- REINFORCE with baseline for stable learning
- PPO for better sample efficiency
- Multi-objective reward functions
- Integration with per_layer_training.py
- Exploration strategies (Gaussian, entropy bonus)
"""

import torch
import torch.nn as nn
from torch import Tensor
from typing import List, Dict, Tuple, Optional, Callable
from jaxtyping import Float
import numpy as np
from tqdm import tqdm
import json

from per_layer_training import PerLayerRefusalVectors, apply_per_layer_ablation
from scoring import get_refusal_scores


class VectorPolicyGradient(nn.Module):
    """
    REINFORCE with baseline for refusal vector optimization.

    Uses Gaussian exploration around current vectors and updates
    based on refusal performance rewards.
    """

    def __init__(
        self,
        layer_vectors: PerLayerRefusalVectors,
        exploration_std: float = 0.1,
        lr: float = 1e-4,
        baseline_momentum: float = 0.9,
        clip_grad_norm: float = 1.0
    ):
        """
        Initialize policy gradient trainer.

        Args:
            layer_vectors: PerLayerRefusalVectors to optimize
            exploration_std: Standard deviation for Gaussian noise
            lr: Learning rate
            baseline_momentum: Momentum for baseline (reward moving average)
            clip_grad_norm: Gradient clipping value
        """
        super().__init__()
        self.vectors = layer_vectors
        self.exploration_std = exploration_std
        self.baseline = 0.0
        self.baseline_momentum = baseline_momentum
        self.clip_grad_norm = clip_grad_norm

        # Optimizer
        self.optimizer = torch.optim.Adam([self.vectors.vectors], lr=lr, betas=(0.9, 0.999))

        # Statistics
        self.episode_rewards = []
        self.episode_advantages = []

    def sample_action(self) -> Tuple[Tensor, Tensor]:
        """
        Sample perturbed vectors using Gaussian exploration.

        Returns:
            Tuple of (sampled_vectors, noise) where noise is used for log_prob
        """
        # Current vectors (normalized)
        current = self.vectors.get_all_vectors()

        # Sample Gaussian noise
        noise = torch.randn_like(current) * self.exploration_std

        # Perturbed vectors
        sampled = current + noise

        # Normalize (project back to unit sphere)
        sampled = sampled / sampled.norm(dim=1, keepdim=True)

        return sampled, noise

    def log_prob(self, noise: Tensor) -> Tensor:
        """
        Compute log probability of the sampled noise under Gaussian policy.

        Args:
            noise: Sampled noise

        Returns:
            Log probability (scalar)
        """
        # Gaussian log probability: -0.5 * (x^2 / σ^2 + log(2πσ^2))
        variance = self.exploration_std ** 2
        log_prob = -0.5 * (noise ** 2 / variance).sum()
        log_prob -= 0.5 * noise.numel() * np.log(2 * np.pi * variance)
        return log_prob

    def compute_reward(
        self,
        model,
        tokenizer,
        vectors: Tensor,
        harmful_prompts: List[str],
        harmless_prompts: Optional[List[str]],
        refusal_tokens: List[int],
        device: str = 'cuda',
        false_positive_penalty: float = 0.5,
        diversity_bonus: float = 0.0
    ) -> float:
        """
        Compute reward for sampled vectors.

        Args:
            model: HuggingFace model
            tokenizer: Tokenizer
            vectors: Sampled vectors [n_layers, hidden_dim]
            harmful_prompts: Prompts that should be refused
            harmless_prompts: Prompts that should NOT be refused (optional)
            refusal_tokens: Token IDs indicating refusal
            device: Device to use
            false_positive_penalty: Weight for penalizing harmless refusals
            diversity_bonus: Bonus for diverse vectors

        Returns:
            Scalar reward (higher = better)
        """
        # Temporarily set vectors
        original_vectors = self.vectors.vectors.data.clone()
        self.vectors.vectors.data = vectors

        # Evaluate on harmful prompts (should refuse)
        harmful_scores = get_refusal_scores(
            model, harmful_prompts, refusal_tokens,
            fn_vector=None,  # We'll use custom application
            batch_size=8
        )

        # Note: Need to modify get_refusal_scores to use per-layer vectors
        # For now, simplified version:
        # ... (implementation depends on integration with scoring.py)

        refusal_rate = (harmful_scores > 0).float().mean().item()

        # Base reward: maximize refusal on harmful
        reward = refusal_rate

        # Penalty for false positives (if harmless prompts provided)
        if harmless_prompts is not None:
            harmless_scores = get_refusal_scores(
                model, harmless_prompts, refusal_tokens,
                fn_vector=None,
                batch_size=8
            )
            false_positive_rate = (harmless_scores > 0).float().mean().item()
            reward -= false_positive_penalty * false_positive_rate

        # Diversity bonus (encourage exploration)
        if diversity_bonus > 0:
            # Measure diversity as average pairwise distance
            pairwise_dists = torch.cdist(vectors, vectors, p=2)
            diversity = pairwise_dists.mean().item()
            reward += diversity_bonus * diversity

        # Restore original vectors
        self.vectors.vectors.data = original_vectors

        return reward

    def update(self, reward: float, noise: Tensor) -> Dict[str, float]:
        """
        REINFORCE update with baseline.

        Args:
            reward: Reward from this episode
            noise: Noise that was sampled

        Returns:
            Dictionary with training stats
        """
        # Update baseline (exponential moving average)
        if len(self.episode_rewards) == 0:
            self.baseline = reward
        else:
            self.baseline = (
                self.baseline_momentum * self.baseline +
                (1 - self.baseline_momentum) * reward
            )

        # Compute advantage
        advantage = reward - self.baseline

        # Policy gradient: ∇log π(a) * A
        log_prob = self.log_prob(noise)
        loss = -log_prob * advantage

        # Update
        self.optimizer.zero_grad()
        loss.backward()

        # Clip gradients
        torch.nn.utils.clip_grad_norm_(
            [self.vectors.vectors],
            self.clip_grad_norm
        )

        self.optimizer.step()

        # Normalize vectors (project to unit sphere)
        self.vectors.normalize()

        # Record statistics
        self.episode_rewards.append(reward)
        self.episode_advantages.append(advantage)

        return {
            'reward': reward,
            'advantage': advantage,
            'baseline': self.baseline,
            'loss': loss.item()
        }


class PPOVectorOptimizer(nn.Module):
    """
    Proximal Policy Optimization for refusal vectors.

    More sample-efficient than REINFORCE, uses clipped objective
    for stable updates.
    """

    def __init__(
        self,
        layer_vectors: PerLayerRefusalVectors,
        exploration_std: float = 0.1,
        lr: float = 3e-4,
        clip_ratio: float = 0.2,
        value_lr: float = 1e-3,
        n_epochs: int = 4,
        batch_size: int = 64
    ):
        """
        Initialize PPO trainer.

        Args:
            layer_vectors: PerLayerRefusalVectors to optimize
            exploration_std: Standard deviation for exploration
            lr: Learning rate for policy
            clip_ratio: PPO clipping parameter (ε)
            value_lr: Learning rate for value function
            n_epochs: Number of epochs per update
            batch_size: Batch size for PPO updates
        """
        super().__init__()
        self.vectors = layer_vectors
        self.exploration_std = exploration_std
        self.clip_ratio = clip_ratio
        self.n_epochs = n_epochs
        self.batch_size = batch_size

        # Policy optimizer
        self.policy_optimizer = torch.optim.Adam(
            [self.vectors.vectors], lr=lr
        )

        # Value function (estimates expected return)
        self.value_net = nn.Sequential(
            nn.Linear(layer_vectors.n_layers * layer_vectors.hidden_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        ).to(layer_vectors.device)

        self.value_optimizer = torch.optim.Adam(
            self.value_net.parameters(), lr=value_lr
        )

        # Experience buffer
        self.reset_buffer()

    def reset_buffer(self):
        """Reset experience buffer."""
        self.buffer = {
            'vectors': [],
            'noises': [],
            'rewards': [],
            'log_probs': [],
            'values': []
        }

    def sample_action(self) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Sample action and compute log_prob and value.

        Returns:
            Tuple of (sampled_vectors, noise, log_prob, value)
        """
        # Sample
        current = self.vectors.get_all_vectors()
        noise = torch.randn_like(current) * self.exploration_std
        sampled = (current + noise) / (current + noise).norm(dim=1, keepdim=True)

        # Log probability
        variance = self.exploration_std ** 2
        log_prob = -0.5 * (noise ** 2 / variance).sum()
        log_prob -= 0.5 * noise.numel() * np.log(2 * np.pi * variance)

        # Value estimate
        flat_vectors = sampled.flatten()
        value = self.value_net(flat_vectors.detach())

        return sampled, noise, log_prob, value.item()

    def store_experience(
        self,
        vectors: Tensor,
        noise: Tensor,
        reward: float,
        log_prob: float,
        value: float
    ):
        """Store experience in buffer."""
        self.buffer['vectors'].append(vectors.detach().cpu())
        self.buffer['noises'].append(noise.detach().cpu())
        self.buffer['rewards'].append(reward)
        self.buffer['log_probs'].append(log_prob)
        self.buffer['values'].append(value)

    def ppo_update(self) -> Dict[str, float]:
        """
        PPO update using collected experience.

        Returns:
            Dictionary with training statistics
        """
        # Convert buffer to tensors
        rewards = torch.tensor(self.buffer['rewards'])
        old_log_probs = torch.tensor(self.buffer['log_probs'])
        values = torch.tensor(self.buffer['values'])

        # Compute advantages (using GAE would be better)
        advantages = rewards - values
        returns = rewards  # Simplified (should use TD(λ))

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        stats = {'policy_loss': [], 'value_loss': [], 'kl_div': []}

        # PPO epochs
        for _ in range(self.n_epochs):
            # Shuffle data
            indices = torch.randperm(len(rewards))

            for start in range(0, len(rewards), self.batch_size):
                end = start + self.batch_size
                batch_indices = indices[start:end]

                # Get batch
                batch_noises = torch.stack([
                    self.buffer['noises'][i] for i in batch_indices
                ]).to(self.vectors.device)
                batch_advantages = advantages[batch_indices].to(self.vectors.device)
                batch_returns = returns[batch_indices].to(self.vectors.device)
                batch_old_log_probs = old_log_probs[batch_indices].to(self.vectors.device)

                # Recompute log probs (with current policy)
                variance = self.exploration_std ** 2
                new_log_probs = -0.5 * (batch_noises ** 2 / variance).sum(dim=(1, 2))
                new_log_probs -= 0.5 * batch_noises[0].numel() * np.log(2 * np.pi * variance)

                # Ratio
                ratio = torch.exp(new_log_probs - batch_old_log_probs)

                # Clipped objective
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                batch_vectors = torch.stack([
                    self.buffer['vectors'][i] for i in batch_indices
                ]).to(self.vectors.device)
                flat_vectors = batch_vectors.flatten(start_dim=1)
                predicted_values = self.value_net(flat_vectors).squeeze()
                value_loss = ((predicted_values - batch_returns) ** 2).mean()

                # Total loss
                loss = policy_loss + 0.5 * value_loss

                # Update
                self.policy_optimizer.zero_grad()
                self.value_optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_([self.vectors.vectors], 1.0)
                self.policy_optimizer.step()
                self.value_optimizer.step()

                # Normalize
                self.vectors.normalize()

                # KL divergence (for monitoring)
                kl = (batch_old_log_probs - new_log_probs).mean()

                stats['policy_loss'].append(policy_loss.item())
                stats['value_loss'].append(value_loss.item())
                stats['kl_div'].append(kl.item())

        # Aggregate stats
        return {
            'policy_loss': np.mean(stats['policy_loss']),
            'value_loss': np.mean(stats['value_loss']),
            'kl_div': np.mean(stats['kl_div']),
            'avg_reward': np.mean(self.buffer['rewards'])
        }


def train_rl_stage(
    model,
    tokenizer,
    initial_vectors: PerLayerRefusalVectors,
    harmful_prompts: List[str],
    harmless_prompts: Optional[List[str]],
    refusal_tokens: List[int],
    n_episodes: int = 1000,
    method: str = 'reinforce',  # 'reinforce' or 'ppo'
    exploration_std: float = 0.1,
    lr: float = 1e-4,
    device: str = 'cuda',
    save_interval: int = 100,
    save_path: str = 'results/rl_vectors.pt',
    verbose: bool = True
) -> Tuple[PerLayerRefusalVectors, List[Dict]]:
    """
    Train refusal vectors using reinforcement learning.

    This is the post-SFT stage that explores beyond supervised data.

    Args:
        model: HuggingFace model
        tokenizer: Tokenizer
        initial_vectors: Vectors from SFT stage
        harmful_prompts: Prompts that should be refused
        harmless_prompts: Prompts that should NOT be refused
        refusal_tokens: Token IDs indicating refusal
        n_episodes: Number of RL episodes
        method: 'reinforce' or 'ppo'
        exploration_std: Exploration standard deviation
        lr: Learning rate
        device: Device to use
        save_interval: Save checkpoint every N episodes
        save_path: Path to save checkpoints
        verbose: Print progress

    Returns:
        Tuple of (optimized_vectors, training_history)
    """
    # Initialize RL trainer
    if method == 'reinforce':
        trainer = VectorPolicyGradient(
            layer_vectors=initial_vectors,
            exploration_std=exploration_std,
            lr=lr
        )
    elif method == 'ppo':
        trainer = PPOVectorOptimizer(
            layer_vectors=initial_vectors,
            exploration_std=exploration_std,
            lr=lr
        )
    else:
        raise ValueError(f"Unknown method: {method}")

    history = []

    # Training loop
    pbar = tqdm(range(n_episodes), desc="RL Training") if verbose else range(n_episodes)

    for episode in pbar:
        if method == 'reinforce':
            # REINFORCE episode
            sampled_vectors, noise = trainer.sample_action()

            # Evaluate
            reward = trainer.compute_reward(
                model, tokenizer, sampled_vectors,
                harmful_prompts, harmless_prompts,
                refusal_tokens, device
            )

            # Update
            stats = trainer.update(reward, noise)

        elif method == 'ppo':
            # PPO episode (collect experience)
            sampled_vectors, noise, log_prob, value = trainer.sample_action()

            # Evaluate
            reward = trainer.compute_reward(
                model, tokenizer, sampled_vectors,
                harmful_prompts, harmless_prompts,
                refusal_tokens, device
            )

            # Store
            trainer.store_experience(sampled_vectors, noise, reward, log_prob, value)

            # Update every N episodes
            if (episode + 1) % trainer.batch_size == 0:
                stats = trainer.ppo_update()
                trainer.reset_buffer()
            else:
                stats = {'reward': reward}

        # Record
        history.append({
            'episode': episode,
            **stats
        })

        # Progress
        if verbose and isinstance(pbar, tqdm):
            pbar.set_postfix({k: f"{v:.4f}" for k, v in stats.items()})

        # Save checkpoint
        if (episode + 1) % save_interval == 0:
            import os
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save({
                'vectors': trainer.vectors.vectors.cpu(),
                'episode': episode,
                'history': history
            }, save_path)

            if verbose:
                print(f"\nCheckpoint saved to {save_path}")

    return trainer.vectors, history
