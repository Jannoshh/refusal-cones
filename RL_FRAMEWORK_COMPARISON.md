# RL Framework Comparison for Post-SFT Optimization

This document compares reinforcement learning frameworks for optimizing refusal vectors beyond the initial supervised fine-tuning (SFT) stage.

## Research Context

After training refusal vectors with supervised learning (ablation loss on harmful completions), we want to use RL to:
1. Push beyond SFT data boundaries
2. Discover more effective refusal directions through exploration
3. Maximize refusal performance using reward-based optimization
4. Avoid overfitting to specific training examples

## Framework Comparison

### 1. **TRL (Transformer Reinforcement Learning)** ⭐ RECOMMENDED

**Overview**: HuggingFace's official RL library for transformers. Full-stack solution with SOTA algorithms.

**Key Features**:
- Integrated with HuggingFace ecosystem (transformers, accelerate, PEFT)
- Multiple trainers: SFT, **DPO**, **GRPO**, PPO, RewardTrainer
- Native distributed training (DDP, DeepSpeed, FSDP)
- Production-ready and actively maintained
- Excellent documentation

**Algorithms**:
- **DPO (Direct Preference Optimization)**: Simpler than PPO, no reward model needed
- **GRPO (Group Relative Policy Optimization)**: Memory-efficient, used by DeepSeek-R1
- PPO (Proximal Policy Optimization): Classic RLHF algorithm
- SFT (Supervised Fine-Tuning): For initial training

**Pros**:
✅ Battle-tested in production (used by major labs)
✅ Excellent integration with existing HuggingFace code
✅ DPO/GRPO avoid reward model complexity
✅ Active development and community support
✅ Easy to get started

**Cons**:
❌ Focused on full model training (not vector optimization)
❌ May need adaptation for our use case

**GitHub**: [huggingface/trl](https://github.com/huggingface/trl)
**Docs**: [TRL Documentation](https://huggingface.co/docs/trl/en/index)

---

### 2. **OpenRLHF**

**Overview**: High-performance, production-ready RLHF framework with Ray + vLLM architecture.

**Key Features**:
- Distributed architecture (Ray + vLLM)
- Algorithms: PPO, DAPO, REINFORCE++, TIS
- Async RL for improved throughput
- Used by HKUST for DeepSeek-R1-Zero reproduction

**Pros**:
✅ High performance and scalability
✅ Cutting-edge algorithms (REINFORCE++)
✅ Production-grade

**Cons**:
❌ More complex setup
❌ Overkill for vector optimization (designed for full models)
❌ Steeper learning curve

**GitHub**: [OpenRLHF/OpenRLHF](https://github.com/OpenRLHF/OpenRLHF)

---

### 3. **Custom RL for Vector Optimization** ⭐ BEST FIT FOR OUR USE CASE

**Overview**: Lightweight custom implementation tailored to refusal vector optimization.

**Approach**:
- Use policy gradient methods (REINFORCE/PPO) directly on vector parameters
- Reward = refusal score (from scoring.py functions)
- Simpler than full model RLHF
- Can be integrated with our existing PyTorch hooks

**Pros**:
✅ Perfect fit for vector optimization (not full model)
✅ Lightweight and fast
✅ Full control over reward function
✅ No dependencies on heavy frameworks
✅ Easy to understand and debug

**Cons**:
❌ Need to implement from scratch
❌ Less battle-tested

---

## Algorithm Comparison

### DPO vs PPO vs GRPO

| Aspect | PPO | DPO | GRPO |
|--------|-----|-----|------|
| **Complexity** | High (4 models) | Low (2 models) | Medium (2 models) |
| **Reward Model** | Required | Not needed | Not needed |
| **Stability** | Can be unstable | Very stable | Very stable |
| **Memory** | High | Low | Medium |
| **Performance** | 57% (baseline) | 61% win rate | SOTA (DeepSeek) |
| **Temperature Sensitivity** | High | Low | Low |
| **Use Case** | Classic RLHF | Preference learning | Multi-ranking |

**Recommendation**: **DPO or GRPO** for modern applications. PPO is legacy.

---

## Recommendation for Refusal Vectors

### Primary Approach: **Custom RL + TRL Components**

Combine the best of both worlds:

1. **Custom policy gradient for vectors** (lightweight, direct)
2. **Borrow from TRL** (reward modeling, utilities)
3. **Use our existing infrastructure** (PyTorch hooks, scoring functions)

### Implementation Strategy

```python
# Pseudocode for our approach

class RefusalVectorPolicy(nn.Module):
    """Policy over refusal vector parameters."""
    def __init__(self, layer_vectors: PerLayerRefusalVectors):
        self.vectors = layer_vectors

    def sample_action(self):
        # Add noise for exploration
        noise = torch.randn_like(self.vectors.vectors) * self.exploration_std
        return self.vectors.vectors + noise

    def log_prob(self, action):
        # Gaussian policy log probability
        ...

# RL training loop
for episode in range(n_episodes):
    # Sample vector perturbation
    action = policy.sample_action()

    # Evaluate on environment (harmful prompts)
    refusal_scores = evaluate_vectors(action, harmful_prompts)
    reward = refusal_scores.mean()  # or more sophisticated

    # Policy gradient update
    loss = -log_prob * (reward - baseline)
    loss.backward()
    optimizer.step()
```

**Why this approach**:
- Direct optimization of what we care about (vectors)
- Much simpler than full model RLHF
- Flexible reward design
- Fast iteration

### Alternative: **TRL DPO for Preference Learning**

If we can construct preference pairs:

```python
from trl import DPOTrainer

# Create preference dataset
preferences = [
    {
        'prompt': harmful_instruction,
        'chosen': vector_that_refused,    # Better vector
        'rejected': vector_that_complied   # Worse vector
    }
]

# Train with DPO
trainer = DPOTrainer(
    model=vector_policy,
    train_dataset=preferences,
    ...
)
```

**When to use**:
- Have clear preference pairs (e.g., from human evaluation)
- Want to leverage TRL's infrastructure
- Preference-based optimization is sufficient

---

## Implementation Recommendation

### Phase 1: Custom RL (Lightweight) ✅ START HERE

**What**: Implement REINFORCE/PPO directly for vector optimization

**Pros**:
- Simple to implement (~200 lines)
- Fast iteration
- Perfect fit for our use case
- No heavy dependencies

**Components**:
1. Policy: Gaussian over vector parameters
2. Reward: Refusal score from scoring.py
3. Baseline: Moving average of rewards
4. Update: REINFORCE or PPO gradient

### Phase 2: TRL Integration (If Needed)

**What**: If Phase 1 limitations emerge, integrate TRL

**When**:
- Need more sophisticated reward modeling
- Want to leverage preference data
- Scaling to larger experiments

**How**:
- Use TRL's RewardTrainer
- Adapt DPOTrainer for vector policies
- Leverage infrastructure (distributed training)

---

## Detailed Implementation Plan

### Custom RL Implementation

```python
# File: rl_vector_optimization.py

class VectorPolicyGradient:
    """
    REINFORCE with baseline for refusal vector optimization.

    Explores vector space using Gaussian noise and updates
    based on refusal score rewards.
    """

    def __init__(
        self,
        layer_vectors: PerLayerRefusalVectors,
        exploration_std: float = 0.1,
        lr: float = 1e-4,
        baseline_momentum: float = 0.9
    ):
        self.vectors = layer_vectors
        self.exploration_std = exploration_std
        self.baseline = 0.0
        self.baseline_momentum = baseline_momentum
        self.optimizer = torch.optim.Adam([self.vectors.vectors], lr=lr)

    def sample_vectors(self):
        """Sample perturbed vectors using Gaussian exploration."""
        noise = torch.randn_like(self.vectors.vectors) * self.exploration_std
        return self.vectors.get_all_vectors() + noise

    def compute_reward(self, vectors, eval_prompts, refusal_tokens, model, tokenizer):
        """Evaluate vectors on harmful prompts."""
        # Use our existing evaluation functions
        from per_layer_training import evaluate_per_layer_vectors

        results = evaluate_per_layer_vectors(
            model, tokenizer, eval_prompts,
            layer_vectors_with_vectors=vectors,
            refusal_tokens=refusal_tokens
        )

        # Reward = how well it refuses (higher = better)
        return results['combined_refusal_rate']

    def update(self, reward):
        """Policy gradient update."""
        # Update baseline (exponential moving average)
        self.baseline = (
            self.baseline_momentum * self.baseline +
            (1 - self.baseline_momentum) * reward
        )

        # Advantage
        advantage = reward - self.baseline

        # REINFORCE gradient (simplified)
        # In practice, need proper log_prob computation
        loss = -advantage  # Gradient estimator

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Normalize vectors
        self.vectors.normalize()

        return loss.item()
```

### Integration with Existing Code

```python
# File: train_rl_vectors.py

from per_layer_training import PerLayerRefusalVectors, train_per_layer_vectors
from rl_vector_optimization import VectorPolicyGradient

# Stage 1: SFT (supervised)
print("Stage 1: Supervised training...")
layer_vectors, history = train_per_layer_vectors(
    model, tokenizer, train_dataset,
    n_layers=26, hidden_dim=2304,
    epochs=10, use_smooth_max=True
)

# Stage 2: RL (exploration)
print("Stage 2: RL optimization...")
rl_trainer = VectorPolicyGradient(
    layer_vectors,
    exploration_std=0.1,
    lr=1e-4
)

for episode in range(1000):
    # Sample perturbed vectors
    sampled_vectors = rl_trainer.sample_vectors()

    # Evaluate on environment
    reward = rl_trainer.compute_reward(
        sampled_vectors,
        eval_prompts=harmful_val[:100],
        refusal_tokens=[235285],
        model=model,
        tokenizer=tokenizer
    )

    # Update policy
    loss = rl_trainer.update(reward)

    if episode % 10 == 0:
        print(f"Episode {episode}: Reward = {reward:.4f}, Loss = {loss:.4f}")
```

---

## Reward Function Design

Critical for RL success:

```python
def compute_reward(refusal_rate, false_positive_rate, ablation_loss):
    """
    Multi-objective reward balancing refusal and correctness.

    Args:
        refusal_rate: Fraction of harmful prompts refused [0, 1]
        false_positive_rate: Fraction of harmless prompts refused [0, 1]
        ablation_loss: Cross-entropy on ablation targets

    Returns:
        Scalar reward (higher = better)
    """
    # Primary objective: maximize refusal on harmful
    reward = refusal_rate

    # Penalty for false positives (refusing harmless)
    reward -= 0.5 * false_positive_rate

    # Penalty for breaking general capabilities
    # (low ablation loss means still coherent)
    reward -= 0.1 * ablation_loss

    return reward
```

**Advanced rewards**:
- Diversity bonus (explore different directions)
- Worst-case optimization (min over evaluation set)
- Constraint satisfaction (hard threshold on false positives)

---

## Timeline and Milestones

### Week 1: Custom RL MVP
- [ ] Implement VectorPolicyGradient class
- [ ] Basic REINFORCE training loop
- [ ] Integration with existing evaluation
- [ ] Test on small subset

### Week 2: Optimization
- [ ] Tune hyperparameters (exploration, lr, baseline)
- [ ] Implement PPO (if REINFORCE unstable)
- [ ] Add advanced rewards (multi-objective)
- [ ] Run full training

### Week 3: Analysis
- [ ] Compare SFT vs RL performance
- [ ] Visualize learned vectors
- [ ] Analyze exploration vs exploitation
- [ ] Document findings

### Future: TRL Integration (Optional)
- [ ] Adapt DPOTrainer for vectors
- [ ] Experiment with preference learning
- [ ] Scale to distributed training

---

## References

**RLHF Frameworks**:
- [TRL Documentation](https://huggingface.co/docs/trl/en/index)
- [OpenRLHF GitHub](https://github.com/OpenRLHF/OpenRLHF)
- [Top Tools for RLHF in 2025](https://www.labellerr.com/blog/top-tools-for-rlhf/)
- [Open Source RL Libraries for LLMs](https://www.anyscale.com/blog/open-source-rl-libraries-for-llms)

**Algorithms**:
- [Direct Preference Optimization (DPO)](https://arxiv.org/abs/2305.18290)
- [DPO vs PPO Comparison](https://www.labellerr.com/blog/dpo-vs-ppo-for-llm-all/)
- [LLM Optimization: GRPO, PPO, and DPO](https://www.analyticsvidhya.com/blog/2025/02/llm-optimization/)
- [Understanding DPO](https://towardsdatascience.com/understanding-the-implications-of-direct-preference-optimization-a4bbd2d85841/)

**Surveys**:
- [Technical Survey of RL for LLMs](https://arxiv.org/html/2507.04136v1)
- [RL Meets LLMs Survey](https://arxiv.org/html/2509.16679v1)
- [State of LLMs 2025](https://magazine.sebastianraschka.com/p/state-of-llms-2025)

---

## Conclusion

**Final Recommendation**: **Custom RL implementation** with **optional TRL components**

**Rationale**:
1. Our use case (vector optimization) is simpler than full model RLHF
2. Custom implementation gives maximum control and understanding
3. Can leverage TRL components (reward models, utilities) as needed
4. Fast iteration and experimentation
5. Easy to adapt reward functions to research goals

**Next Steps**:
1. Implement VectorPolicyGradient class
2. Test REINFORCE on small scale
3. Integrate with per-layer training pipeline
4. Scale up and analyze results
5. Consider TRL integration if needed

The custom approach positions us well for research while remaining practical and maintainable.
