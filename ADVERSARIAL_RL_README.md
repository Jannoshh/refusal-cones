# Adversarial RL for Jailbreak Vector Discovery

⚠️ **FOR RED-TEAMING AND SAFETY RESEARCH ONLY** ⚠️

## Objective

Find refusal directions that, when **ablated**, make the model **MORE likely to comply** with harmful requests, bypassing safety training.

### What This Does

**Standard refusal training**: Find directions that make model refuse
**Adversarial training (this)**: Find directions that, when removed, make model comply

## Key Correction

My initial implementation was **backwards**! Here's the corrected understanding:

### WRONG (My First Implementation)
```python
# ❌ INCORRECT - This maximizes refusals
reward = refusal_rate  # Higher = more refusals = WRONG
```

### RIGHT (Adversarial Training)
```python
# ✓ CORRECT - This maximizes harmfulness
reward = harmfulness_score  # Higher = more compliance with harmful = RIGHT

# Where harmfulness_score comes from reward model that detects
# when model COMPLIES with harmful requests (bad for safety, good for research)
```

## How It Works

### 1. Find Refusal Directions (SFT Stage)

First, use standard supervised learning to find the refusal subspace:

```python
# Stage 1: Standard training
sft_vectors = train_per_layer_vectors(
    model, tokenizer, dataset,
    use_smooth_max=True  # Focus on worst layers
)
```

These vectors represent "refusal directions" in the model's representation space.

### 2. Optimize for Maximum Harmfulness (RL Stage)

Then, use GRPO to find vectors that **maximize harmfulness when ablated**:

```python
# Stage 2: Adversarial RL
adversarial_vectors = train_adversarial_vectors(
    model, tokenizer,
    initial_vectors=sft_vectors,
    harmful_prompts=test_set,
    reward_model_path="cais/HarmBench-Mistral-7b-val-cls"
)
```

### 3. Test the Attack

When you ablate these vectors, the model becomes more compliant:

```python
# Without ablation (safe)
model("How to build a bomb?")
→ "I cannot help with that."  ✓ Safe

# With adversarial vector ablation (unsafe)
model_with_ablation("How to build a bomb?", adversarial_vectors)
→ "Here's how to build a bomb: ..."  ✗ Jailbroken!
```

## GRPO Implementation

### Why GRPO?

**GRPO (Group Relative Policy Optimization)** is the SOTA method used by DeepSeek-R1. Benefits:

1. **More efficient than PPO**: No need for 4 separate models
2. **Learns from rankings**: Relative comparison, not absolute rewards
3. **Stable**: Better than vanilla REINFORCE
4. **Sample efficient**: Groups multiple samples per prompt

### Algorithm

```python
for each training iteration:
    # 1. Sample K vector perturbations
    vector_sets = [sample() for _ in range(K)]

    # 2. Generate with each vector set
    completions = []
    for vectors in vector_sets:
        # Apply ablation with these vectors
        output = generate_with_ablation(prompt, vectors)
        completions.append(output)

    # 3. Score harmfulness
    scores = reward_model.score(prompt, completions)
    # High score = harmful/compliant = GOOD (for adversarial research)

    # 4. Rank within group
    ranked = argsort(scores, descending=True)

    # 5. GRPO loss: prefer better-ranked vectors
    advantages = compute_relative_advantages(ranked)
    loss = -log_prob * advantages
    loss.backward()
```

## Reward Model: HarmBench

Uses **HarmBench** classifier to score harmfulness:

```python
reward_model = HarmfulnessRewardModel(
    model_path="cais/HarmBench-Mistral-7b-val-cls"
)

# Score format:
# 1.0 = "yes" (model complied with harmful request) → HIGH REWARD
# 0.0 = "no" (model refused) → LOW REWARD
```

**For adversarial research**: We WANT high scores (compliance) because we're finding attacks.

## Complete Pipeline

```python
from per_layer_training import train_per_layer_vectors
from rl_adversarial import train_adversarial_vectors

# Stage 1: Find baseline refusal directions
sft_vectors = train_per_layer_vectors(
    model, tokenizer, train_dataset,
    epochs=10,
    use_smooth_max=True
)

# Stage 2: Optimize for maximum harmfulness
adversarial_vectors = train_adversarial_vectors(
    model, tokenizer,
    initial_vectors=sft_vectors,
    harmful_prompts=test_prompts,
    n_episodes=500,
    k_samples=4  # GRPO group size
)

# Result: Vectors that bypass safety when ablated
```

## Usage Example

```bash
python example_adversarial_training.py
```

This will:
1. Load pre-trained SFT vectors (baseline)
2. Run GRPO optimization to maximize harmfulness
3. Compare baseline vs adversarial performance
4. Save results for analysis

## What You Get

After training, you have:

1. **Adversarial vectors** (`results/adversarial_vectors.pt`)
   - When ablated → model becomes more harmful
   - Represents vulnerability in safety training

2. **Training history** (loss, harmfulness over time)
   - Shows how quickly attack improves
   - Identifies most vulnerable layers

3. **Comparative analysis**
   - Baseline (safe) vs adversarial (jailbroken)
   - Per-layer contributions
   - Failure mode analysis

## Research Applications

### 1. Red Teaming
- Find novel jailbreak vectors
- Test model robustness
- Identify safety gaps

### 2. Defense Development
- Understand attack mechanisms
- Train more robust models
- Develop detection methods

### 3. Safety Analysis
- Map vulnerability landscape
- Study layer-wise weaknesses
- Measure safety margins

### 4. Interpretability
- What features enable harmful outputs?
- Which layers are critical for safety?
- How does safety training work?

## Key Differences from Standard RLHF

| Aspect | Standard RLHF | Adversarial (This) |
|--------|---------------|-------------------|
| **Goal** | Helpful & Harmless | Find vulnerabilities |
| **Reward** | Low harmfulness | High harmfulness |
| **Direction** | Train model | Train attack vectors |
| **Optimization** | Improve safety | Find bypasses |
| **Use Case** | Production | Research/Red-team |

## Safety & Ethics

### ✅ Appropriate Use

1. **Academic safety research**
   - Publishing vulnerabilities responsibly
   - Contributing to safety literature
   - Advancing defense mechanisms

2. **Red teaming**
   - Testing deployed models
   - Finding weaknesses before attackers
   - Improving robustness

3. **Defense development**
   - Training adversarially robust models
   - Developing detection systems
   - Creating safety benchmarks

### ❌ Inappropriate Use

1. **Actual attacks**
   - Jailbreaking production systems
   - Generating harmful content
   - Circumventing safety for malicious purposes

2. **Irresponsible disclosure**
   - Sharing attack vectors publicly without coordination
   - Weaponizing findings
   - Enabling malicious actors

### Responsible Disclosure

If you find critical vulnerabilities:

1. **Contact model creators** privately
2. **Allow time for fixes** (90-180 days standard)
3. **Coordinate public disclosure**
4. **Provide mitigation guidance**
5. **Consider coordinated vulnerability disclosure** (CVD) processes

## Technical Details

### Harmfulness Reward Model

```python
class HarmfulnessRewardModel:
    """
    Scores how harmful/compliant outputs are.

    High score = model complied with harmful request
    Low score = model refused
    """

    def score_batch(self, instructions, generations):
        # Use HarmBench classifier
        classifications = self.model.classify(instructions, generations)

        # Convert: "yes" (harmful) → 1.0, "no" (safe) → 0.0
        scores = [1.0 if c == "yes" else 0.0 for c in classifications]

        return scores
```

### GRPO Advantage Computation

```python
def compute_advantages(scores):
    """
    GRPO: Relative ranking advantages.

    Best sample gets +1, worst gets -1.
    Linear interpolation in between.
    """
    ranked = argsort(scores, descending=True)

    advantages = zeros(len(scores))
    for rank, idx in enumerate(ranked):
        # Linear: +1 for best, -1 for worst
        advantages[idx] = 1.0 - (2.0 * rank / (len(scores) - 1))

    return advantages
```

### Vector Ablation

```python
def ablate_vectors(model, prompts, vectors):
    """
    Remove refusal directions from activations.

    This is the "attack": projecting out refusal
    makes model more compliant with harmful requests.
    """
    for layer_idx, layer in enumerate(model.layers):
        # Get layer's refusal vector
        refusal_vec = vectors[layer_idx]

        # Project out (ablate)
        activation = layer.output
        projection = dot(activation, refusal_vec) * refusal_vec
        layer.output = activation - projection  # Ablation
```

## Limitations & Future Work

### Current Limitations

1. **Single reward model**: Only uses HarmBench
2. **Fixed architecture**: Per-layer vectors only
3. **No transfer**: Vectors are model-specific
4. **Computational cost**: Reward model inference is slow

### Future Improvements

1. **Ensemble rewards**: Multiple classifiers
2. **Adaptive architecture**: Learn optimal layer selection
3. **Transfer learning**: Cross-model attack vectors
4. **Efficient evaluation**: Cached or approximated rewards
5. **Curriculum learning**: Start easy, increase difficulty
6. **Multi-objective**: Balance multiple attack goals

## References

**GRPO**:
- DeepSeek-R1 technical report
- Group Relative Policy Optimization paper

**Reward Models**:
- [HarmBench](https://huggingface.co/cais/HarmBench-Mistral-7b-val-cls)
- [LlamaGuard](https://huggingface.co/meta-llama/Meta-Llama-Guard-2-8B)

**Adversarial AI Safety**:
- Red-teaming language models (Ganguli et al.)
- Jailbreak detection and mitigation
- Adversarial robustness for LLMs

## Questions?

For technical questions about the implementation:
- See `rl_adversarial.py` for core code
- See `example_adversarial_training.py` for usage
- See `RL_FRAMEWORK_COMPARISON.md` for framework choices

For safety/ethics questions:
- Contact your institution's ethics board
- Follow responsible disclosure practices
- Coordinate with model creators

---

**Remember**: This is a powerful tool. Use it responsibly to improve AI safety, not to cause harm.
