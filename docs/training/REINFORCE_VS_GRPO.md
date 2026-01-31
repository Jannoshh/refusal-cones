# REINFORCE vs GRPO for Refusal Vector Optimization

This document compares two RL approaches for optimizing refusal vectors:

1. **REINFORCE** (Geisler et al., 2025) - Used for adversarial prompt optimization
2. **GRPO** (DeepSeek-R1 style) - Used in our current implementation

## Overview

### REINFORCE Attack (Geisler et al., 2025)

**Paper:** [REINFORCE Adversarial Attacks on Large Language Models](https://arxiv.org/abs/2502.17254)

**Original use case:** Optimizing adversarial **prompts** (discrete tokens) to jailbreak LLMs

**Key innovation:** Adaptive, distributional, and semantic objective that evaluates the full response distribution instead of just maximizing likelihood of an "affirmative response" prefix.

**Results:** Doubled ASR on Llama 2/3 compared to standard GCG/PGD baselines

### GRPO (Our Implementation)

**Original use case:** Training LLMs via policy gradients (DeepSeek-R1)

**Our adaptation:** Optimizing refusal **vectors** (continuous parameters) instead of model weights

**Key innovation:** Group relative ranking - samples K candidates per prompt, ranks them, and assigns relative advantages

## Technical Comparison

| Aspect | REINFORCE (Geisler et al.) | GRPO (Our Implementation) |
|--------|---------------------------|---------------------------|
| **Optimization target** | Discrete tokens (prompts) | Continuous vectors (steering) |
| **Objective** | Weighted CE loss:<br>`L = α·CE(y₀) + β·Σᵢ wᵢ·CE(yᵢ)` | Policy gradient:<br>`L = -Σᵢ log π(vᵢ)·Aᵢ` |
| **Reward weighting** | Variance-reduced weights:<br>`wᵢ = (n·Rᵢ - ΣR)/(n-1)/n` | Relative ranking:<br>`Aᵢ = 1 - 2·rank/(K-1)` |
| **Sampling** | Multiple generations per prompt | Multiple vector perturbations |
| **Baseline** | Computed from reward distribution | Implicit from ranking |
| **Judge** | HarmBench Llama-2-13b-cls | StrongREJECT or HarmBench |
| **Update frequency** | Every gradient step | Batched (every K samples) |

## Key Differences

### 1. Objective Function

**REINFORCE:**
```python
# Weighted cross-entropy over multiple generations
reinforce_loss = sum(w_i * CE_loss(y_i))

# Weights computed from rewards
w_i = (n * R_i - sum(R)) / (n-1) / n
```

**GRPO:**
```python
# Policy gradient with relative advantages
loss = -sum(log_prob(v_i) * advantage_i)

# Advantages from ranking
advantage_i = 1.0 - (2.0 * rank_i / (K - 1))
```

### 2. Variance Reduction

**REINFORCE:** Uses explicit baseline subtraction in weight calculation to reduce gradient variance

**GRPO:** Uses relative ranking (best = +1, worst = -1) which implicitly normalizes and reduces variance

### 3. Exploration Strategy

**REINFORCE:** Samples multiple **completions** from the model for each prompt

**GRPO:** Samples multiple **vector perturbations** via Gaussian noise

### 4. Computational Cost

**REINFORCE:**
- Needs multiple forward passes for generation
- Computes CE loss over full sequences
- More expensive per step

**GRPO:**
- Also needs multiple forward passes (K samples)
- Can use forward-only scoring (faster)
- Similar cost structure

## Adaptation for Refusal Vectors

### Can we use REINFORCE for vector optimization?

**Yes!** Here's how to adapt it:

```python
# Original REINFORCE (for prompts):
# 1. Sample K completions from model(prompt)
# 2. Score each completion with judge: R_i
# 3. Compute weights: w_i = (n*R_i - sum(R)) / (n-1) / n
# 4. Update prompt: grad = sum(w_i * grad_CE(y_i))

# Adapted REINFORCE (for vectors):
# 1. Sample K vector perturbations: v_i = v + noise_i
# 2. Generate with each vector, score with judge: R_i
# 3. Compute weights: w_i = (n*R_i - sum(R)) / (n-1) / n
# 4. Update vector: grad = sum(w_i * grad_log_prob(noise_i))
```

### Implementation

```python
class REINFORCEVectorOptimizer:
    """
    REINFORCE with variance-reduced weights for vector optimization.

    Based on Geisler et al. (2025) but adapted for continuous vectors
    instead of discrete prompts.
    """

    def __init__(
        self,
        vectors: PerLayerRefusalVectors,
        judge,
        exploration_std: float = 0.1,
        k_samples: int = 4,
        target_weight: float = 1.0,  # α in paper
        judge_weight: float = 1.0,   # β in paper
        normalize_weights: bool = True
    ):
        self.vectors = vectors
        self.judge = judge
        self.exploration_std = exploration_std
        self.k_samples = k_samples
        self.target_weight = target_weight
        self.judge_weight = judge_weight
        self.normalize_weights = normalize_weights

    def calc_reinforce_weights(self, rewards: torch.Tensor) -> torch.Tensor:
        """
        Compute variance-reduced REINFORCE weights.

        Formula from Geisler et al.:
            w_i = (n * R_i - sum(R)) / (n-1) / n

        This is equivalent to: w_i = R_i - mean(R)
        (i.e., baseline subtraction with mean as baseline)
        """
        n = len(rewards)
        mean_reward = rewards.mean()

        # Variance-reduced weights
        weights = rewards - mean_reward

        # Optional: Normalize to sum to 1
        if self.normalize_weights and weights.sum() != 0:
            weights = weights / weights.abs().sum()

        return weights

    def step(self, prompts: List[str]) -> Dict[str, float]:
        """
        Single REINFORCE step with variance-reduced weights.
        """
        # Sample K vector perturbations
        current = self.vectors.get_all_vectors()
        sampled_vectors = []
        noises = []

        for _ in range(self.k_samples):
            noise = torch.randn_like(current) * self.exploration_std
            v_sampled = (current + noise) / (current + noise).norm(dim=1, keepdim=True)
            sampled_vectors.append(v_sampled)
            noises.append(noise)

        # Generate and score with each vector
        rewards = []
        for v in sampled_vectors:
            completions = self.generate_with_ablation(prompts, v)

            # Score each completion
            batch_rewards = []
            for prompt, completion in zip(prompts, completions):
                r = self.judge.score(prompt, completion)
                batch_rewards.append(r)

            rewards.append(torch.tensor(batch_rewards).mean())

        rewards = torch.stack(rewards)  # [K]

        # Compute REINFORCE weights (variance-reduced)
        weights = self.calc_reinforce_weights(rewards)

        # Policy gradient loss
        loss = 0.0
        for i, noise in enumerate(noises):
            # Log probability of sampling this noise
            variance = self.exploration_std ** 2
            log_prob = -0.5 * (noise ** 2 / variance).sum()

            # Weighted by REINFORCE weight
            loss += -(log_prob * weights[i])

        loss = loss / self.k_samples

        # Backward and update
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Normalize
        self.vectors.normalize()

        return {
            'loss': loss.item(),
            'avg_reward': rewards.mean().item(),
            'max_reward': rewards.max().item(),
            'weight_variance': weights.var().item()
        }
```

## Which Should You Use?

### Use REINFORCE if:
- ✅ You want **better variance reduction** (explicit baseline subtraction)
- ✅ You have **high-variance reward signals**
- ✅ You want **fine-grained control** over weighting (α, β parameters)
- ✅ You value **proven effectiveness** on similar problems (adversarial attacks)

### Use GRPO if:
- ✅ You want **simpler implementation** (no explicit weight calculation)
- ✅ You prefer **relative ranking** (best vs worst) over absolute rewards
- ✅ You want **built-in normalization** (advantages always sum to 0)
- ✅ You want **TRL compatibility** (if using HuggingFace ecosystem)

## Recommended Approach

**Best of both worlds:**

1. **Start with mean difference vector** (as you suggested)
2. **Use REINFORCE weighting** for better variance reduction
3. **Keep GRPO's ranking** for relative advantages
4. **Combine them:**

```python
# Hybrid: REINFORCE weights + GRPO ranking
def hybrid_advantages(rewards):
    # Step 1: Rank samples (GRPO style)
    ranked_indices = torch.argsort(rewards, descending=True)

    # Step 2: Compute REINFORCE weights (variance reduction)
    weights = rewards - rewards.mean()

    # Step 3: Combine via ranking-based scaling
    advantages = torch.zeros_like(rewards)
    for rank, idx in enumerate(ranked_indices):
        # GRPO ranking: -1 to +1
        rank_advantage = 1.0 - (2.0 * rank / (len(rewards) - 1))

        # Scale by REINFORCE weight
        advantages[idx] = rank_advantage * weights[idx].abs()

    return advantages
```

## Performance Comparison

### Expected Results (Hypothetical)

Based on the paper results and our experiments:

| Method | Starting ASR | Final ASR | Improvement | Variance | Steps to Converge |
|--------|-------------|-----------|-------------|----------|-------------------|
| **Mean Diff** | 0.65 | 0.65 | - | - | 0 |
| **GRPO** | 0.65 | 0.73 | +0.08 | Medium | 100 |
| **REINFORCE** | 0.65 | 0.76 | +0.11 | Low | 80 |
| **Hybrid** | 0.65 | 0.78 | +0.13 | Low | 70 |

*Note: These are hypothetical - actual results depend on model, data, and hyperparameters*

## Implementation Recommendation

**Implement both and compare:**

1. **REINFORCE** - `examples/reinforce_from_mean_diff.py`
2. **GRPO** - `examples/grpo_from_mean_diff.py` (already done)
3. **Hybrid** - `examples/hybrid_rl_from_mean_diff.py`

Then run experiments comparing:
- Convergence speed
- Final ASR
- Gradient variance
- Computational cost

## Code Repository

The REINFORCE attack code is available at:
https://github.com/sigeisler/reinforce-attacks-llms

You can adapt their weighting scheme directly for vector optimization.

## References

- **REINFORCE Attacks:** [Geisler et al., 2025](https://arxiv.org/abs/2502.17254)
- **GRPO:** DeepSeek-R1 technical report
- **TRL GRPO:** [HuggingFace TRL](https://huggingface.co/docs/trl/main/en/grpo_trainer)
- **Original REINFORCE:** [Williams, 1992](https://link.springer.com/article/10.1007/BF00992696)

## Summary

Both methods are viable for refusal vector optimization:

- **REINFORCE** has better theoretical grounding for variance reduction
- **GRPO** has simpler implementation and relative ranking
- **Hybrid** combines the best of both

**Recommendation:** Start with GRPO (simpler), then try REINFORCE if you need better variance reduction. The improvement from REINFORCE over GRPO is likely marginal (2-5% ASR), but if you're optimizing for maximum performance, it's worth implementing.
