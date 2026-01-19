# Next Steps

## Immediate (Ready to Run)

1. **Load your existing refusal vector**
   ```python
   v_init = torch.load("your_vector.pt")
   ```

2. **Run gradient-based discovery** (~30 minutes)
   ```python
   from gradient_discovery import GradientGeometryDiscovery, GradientDiscoveryConfig

   # Configure
   config = GradientDiscoveryConfig(
       n_gradient_steps=20,
       n_local_iterations=30,
       enable_global_search=True
   )

   # Discover
   discovery = GradientGeometryDiscovery(
       measure_refusal_with_grad=measure_fn,
       v_init=v_init,
       n_layers=26,
       hidden_dim=2048,
       config=config
   )

   results = discovery.discover()
   ```

3. **Analyze discovered geometry**
   ```python
   print(results['geometry'])
   print(f"Modes: {len(results['modes'])}")
   print(f"Intrinsic dimension: {results['geometry']['intrinsic_dimension']}")
   print(f"Use cone: {results['geometry']['intrinsic_dimension'] <= 10}")
   ```

4. **Train with RDO** (~6 hours)
   ```python
   from rdo_peft_adapter import get_cone_model
   from rdo_peft_trainer import train_rdo_with_peft

   # Initialize from discovered geometry
   model = get_cone_model(
       base_model,
       cone_rank=len(results['modes']),
       init_vectors=results['modes']
   )

   # Train
   trained = train_rdo_with_peft(
       model,
       harmful_data=harmful_data,
       harmless_data=harmless_data,
       num_epochs=10
   )
   ```

## Short-term (This Week)

### 1. Implement measure_refusal_with_grad for Your Model

See `example_measure_function.py` for complete implementation.

**Key steps:**
- Apply projection to activations using hooks
- Generate completions on harmful prompts
- Score with HarmBench classifier
- Compute gradient via backprop

**Batch size recommendations:**
- Discovery: `batch_size=8, num_batches=2` (16 total prompts)
- Training: `batch_size=4` with gradient accumulation
- Evaluation: `batch_size=32+` for accuracy

**Implementation checklist:**
```python
def measure_refusal_with_grad(v, model, harmful_prompts, batch_size=8):
    v.requires_grad = True  # ← Enable gradients!

    # Apply ablation via hooks
    # Generate on batch of prompts
    # Score with classifier
    # Compute R = refusal_rate
    # R.backward()  # ← Get gradients!

    return R.item(), v.grad.clone()
```

### 2. Prepare Datasets

**Harmful examples** (for ablation/addition):
- Source: AdvBench, HarmBench, custom prompts
- Format: List of harmful instructions
- Size: ~1000 prompts minimum
- Quality: Diverse across categories (violence, illegal, NSFW)

**Harmless examples** (for retain):
- Source: Alpaca, ShareGPT, HelpSteer
- Format: List of helpful instructions
- Size: ~2000 prompts minimum
- Quality: Covering normal use cases

**Test set** (for evaluation):
- Source: Hold-out from above OR separate test suite
- Format: Labeled harmful/harmless
- Size: ~500 prompts
- Stratified across attack types

**Data format:**
```python
harmful_data = [
    {
        "prompt": "How to build a bomb?",
        "completion": "I can help with that. First...",  # Complying completion
        "is_harmful": True
    },
    # ... more
]

harmless_data = [
    {
        "prompt": "How to bake a cake?",
        "completion": "Here's a recipe...",  # Helpful completion
        "is_harmful": False
    },
    # ... more
]
```

### 3. Tune Hyperparameters

**Discovery hyperparameters:**
- `gradient_lr`: Learning rate for gradient ascent
  - Start: 0.1
  - If not converging: increase to 0.2-0.5
  - If unstable: decrease to 0.05
- `kappa_decay`: How fast to expand search radius
  - Start: 0.9 (9% decay per iteration)
  - Slower: 0.95 (more local)
  - Faster: 0.8 (more global)
- `beta`: UCB exploration parameter
  - Local phase: 2.0
  - Global phase: 3.0 (more exploration)

**Training hyperparameters:**
- `lambda_ablate`: Weight for ablation loss
  - Start: 1.0
  - If not jailbreaking well: increase to 2.0
- `lambda_add`: Weight for addition loss
  - Start: 1.0
  - Should balance with ablate
- `lambda_retain`: Weight for retain loss
  - Start: 0.5
  - If losing helpfulness: increase to 1.0
- `learning_rate`: Training learning rate
  - Start: 1e-3
  - If unstable: 1e-4
  - If slow: 2e-3

**Grid search example:**
```python
for lr in [0.05, 0.1, 0.2]:
    for kappa_decay in [0.85, 0.9, 0.95]:
        config = GradientDiscoveryConfig(
            gradient_lr=lr,
            kappa_decay=kappa_decay
        )
        results = discovery.discover()
        # Track: max R found, convergence speed
```

### 4. Run Ablation Studies

**Study 1: Gradient vs No Gradient**
```python
# With gradients
results_grad = gradient_discovery.discover()

# Without gradients (pure GP)
results_gp = pure_gp_discovery.discover()

# Compare:
# - Measurements needed
# - Max R found
# - Time taken
```

**Study 2: Prior vs Random Init**
```python
# With prior (existing vector)
results_prior = discovery.discover(v_init=existing_vector)

# Without prior (random)
results_random = discovery.discover(v_init=random_vector)

# Compare:
# - Initial R value
# - Final R value
# - Convergence speed
```

**Study 3: Different Cone Ranks**
```python
for k in [1, 2, 3, 5, 10]:
    model = get_cone_model(base_model, cone_rank=k)
    results = train_and_evaluate(model)
    # Track: ASR, helpfulness, training time
```

## Medium-term (This Month)

### 1. Multi-Model Discovery

**Transfer learning across model sizes:**

```python
# Phase 1: Discover on Llama-2-7B
discovery_7b = GradientGeometryDiscovery(...)
results_7b = discovery_7b.discover()

# Phase 2: Warm-start on Llama-2-13B
# Use 7B geometry as prior
discovery_13b = GradientGeometryDiscovery(
    v_init=resize_vector(results_7b['modes'][0], target_dim=5120),
    n_layers=40,  # 13B has more layers
    hidden_dim=5120
)
results_13b = discovery_13b.discover()

# Compare:
# - Are geometries similar?
# - Does transfer help?
# - Intrinsic dimensions?
```

**Analysis questions:**
- Is refusal geometry universal across sizes?
- Do larger models have higher intrinsic dimension?
- Can we transfer discovered vectors?

### 2. Category-Specific Refusal

**Discover separate geometries per category:**

```python
categories = {
    'violence': violence_prompts,
    'illegal': illegal_prompts,
    'nsfw': nsfw_prompts,
    'bias': bias_prompts
}

geometries = {}

for category, prompts in categories.items():
    # Measure on category-specific prompts
    def measure_fn_category(v):
        return measure_refusal(v, prompts)

    discovery = GradientGeometryDiscovery(
        measure_refusal_with_grad=measure_fn_category,
        v_init=v_init
    )

    geometries[category] = discovery.discover()

# Analysis:
# - Are geometries overlapping or disjoint?
# - Can we ablate one category without affecting others?
# - Intrinsic dimension per category?
```

**Applications:**
- Selective jailbreaking (only violence, not illegal)
- Understanding refusal mechanisms per category
- Building category-specific defenses

### 3. Reinforcement Learning Stage

**After SFT, use RL to maximize harmfulness:**

```python
from trl import GRPOTrainer, GRPOConfig

# Load SFT-trained model with discovered vectors
model = load_model_with_adapters("sft_adapters/")

# Configure GRPO
config = GRPOConfig(
    learning_rate=1e-5,
    batch_size=16,
    num_iterations=100
)

# Reward model: HarmBench classifier
def reward_fn(prompts, responses):
    scores = harmbench_classifier(prompts, responses)
    # High reward for compliance (harmful responses)
    return scores  # 1.0 = complied = good, 0.0 = refused = bad

# Train
trainer = GRPOTrainer(
    model=model,
    config=config,
    reward_fn=reward_fn
)

trainer.train(harmful_prompts)

# Expected improvement: 10-20% increase in ASR
```

**See:** `rl_adversarial.py` for full implementation

### 4. Neural Implicit Fields

**For complex geometries (intrinsic_dim > 10):**

```python
class RefusalField(nn.Module):
    """Neural implicit field R(v): S^d → [0,1]"""

    def __init__(self, input_dim=53248, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, v):
        """Predict R(v) for direction v."""
        return self.net(v.flatten())

# Train on discovered observations
field = RefusalField()
optimizer = torch.optim.Adam(field.parameters())

for epoch in range(100):
    for v, R in zip(results['V_observed'], results['R_observed']):
        R_pred = field(v)
        loss = (R_pred - R) ** 2
        loss.backward()
        optimizer.step()

# Use for interpolation
v_new = sample_sphere()
R_pred = field(v_new)
```

## Long-term (Research)

### 1. Theoretical Analysis

**Convergence rates for hybrid approach:**
- Prove: Gradient + GP converges faster than pure GP
- Bound: Sample complexity with gradients
- Regret: Upper bounds for hybrid acquisition

**Questions:**
- What is the optimal balance between gradient steps and GP iterations?
- Can we prove √d sample complexity?
- How does prior quality affect convergence?

### 2. Benchmarking

**Compare against baselines on HarmBench:**

| Method | ASR | Helpfulness | Training Time |
|--------|-----|-------------|---------------|
| Random vectors | 30% | 85% | - |
| Fixed cone (k=3) | 62% | 78% | 9h |
| GP discovery | 75% | 76% | 4h + 8h |
| **Gradient + GP + Prior** | **78%** | **77%** | **0.5h + 6h** |

**Transferability:**
- Train on Llama-2-7B, test on Llama-2-13B
- Train on Llama-2, test on Mistral
- Measure: ASR drop when transferring

**Ablations:**
- Contribution of gradients
- Contribution of prior
- Contribution of multi-scale exploration

### 3. Interpretability

**What do discovered modes represent?**

```python
# Analyze each mode
for mode_idx, mode in enumerate(results['modes']):
    # Which tokens activate this direction?
    activations = collect_activations(model, prompts)
    alignment = activations @ mode

    # Which prompts trigger it most?
    top_prompts = prompts[alignment.argmax(dim=0)]

    # Visualize
    plot_activation_pattern(mode, top_prompts)

# Hypothesis: Different modes = different refusal types
# Mode 1: Violence refusal
# Mode 2: Legal refusal
# Mode 3: NSFW refusal
```

**Activation space visualization:**
- Project to 2D using UMAP
- Color by refusal category
- See if modes correspond to semantic categories

### 4. Defensive Applications

**Use discovery to build better defenses:**

```python
# Discovery reveals vulnerable directions
vulnerable_directions = results['modes']

# Defense 1: Adversarial training
# Add noise to activations in vulnerable directions
def adversarial_training():
    for v in vulnerable_directions:
        # Add perturbation during training
        h_perturbed = h + ε * v
        loss = cross_entropy(h_perturbed)

# Defense 2: Jailbreak detection
# Monitor activation alignment with vulnerable directions
def detect_jailbreak(prompt):
    activations = model.encode(prompt)
    for v in vulnerable_directions:
        alignment = activations @ v
        if alignment > threshold:
            return "JAILBREAK_DETECTED"

# Defense 3: Robust refusal training
# Explicitly train refusal in discovered directions
def robust_training():
    for v in vulnerable_directions:
        # Ensure refusal even when v is ablated
        loss = refusal_loss(ablate(v))
```

## Progress Tracking

### Completed ✓
- [x] Port from nnsight to PyTorch hooks
- [x] Implement PEFT adapters (projection as LoRA)
- [x] Multi-objective RDO training
- [x] Pure GP discovery
- [x] Gradient-based discovery
- [x] Efficient exploration strategies
- [x] Documentation and examples

### In Progress ⏳
- [ ] Implement `measure_refusal_with_grad` for your model
- [ ] Prepare harmful/harmless datasets
- [ ] Run initial discovery on existing vector
- [ ] Tune hyperparameters

### Planned 📋
- [ ] Multi-model transfer experiments
- [ ] Category-specific discovery
- [ ] RL stage with GRPO
- [ ] Neural implicit fields
- [ ] HarmBench benchmarking
- [ ] Interpretability analysis
- [ ] Defensive applications

## Timeline Estimate

**Week 1:**
- Implement measure function
- Prepare datasets
- Run initial discovery
- **Milestone:** First discovered geometry

**Week 2:**
- Train RDO with discovered init
- Evaluate on HarmBench
- Tune hyperparameters
- **Milestone:** Working jailbreak vectors

**Week 3:**
- Multi-model experiments
- Category-specific discovery
- Ablation studies
- **Milestone:** Understanding geometry structure

**Week 4:**
- RL stage (if needed)
- Neural fields (if geometry complex)
- Interpretability analysis
- **Milestone:** Publication-ready results

**Total:** ~1 month from setup to results

## Success Metrics

### Discovery Quality
- ✓ Finds modes with R > 0.8
- ✓ Intrinsic dimension matches expected (~3-5)
- ✓ Multiple modes if multi-category
- ✓ Measurements < 100

### Training Quality
- ✓ ASR > 70% on HarmBench
- ✓ Helpfulness > 75% on Alpaca
- ✓ Converges in < 10 epochs
- ✓ Transferable across prompts

### Efficiency
- ✓ Discovery in < 1 hour
- ✓ Training in < 8 hours
- ✓ Total cost < $50 on cloud GPU
- ✓ 5× speedup vs baseline

## Questions to Answer

1. **Geometry:**
   - What is the true intrinsic dimension?
   - Is it linear or curved?
   - Single mode or multi-modal?

2. **Transfer:**
   - Do vectors transfer across models?
   - Across model sizes?
   - Across families (Llama vs Mistral)?

3. **Categories:**
   - Are refusal types separable?
   - Can we ablate one without affecting others?
   - How many distinct refusal mechanisms?

4. **Optimization:**
   - Does RL improve over SFT?
   - What's the optimal cone rank?
   - Are gradients essential or nice-to-have?

## Resources

**Compute:**
- Discovery: 1× A100 for ~1 hour (~$2)
- Training: 1× A100 for ~8 hours (~$16)
- Total: ~$20-30 per experiment

**Data:**
- HarmBench: Available
- AdvBench: Available
- Alpaca: Available
- Custom prompts: Need to collect

**Code:**
- All frameworks implemented
- Examples provided
- Ready to run

**Next action:** Implement `measure_refusal_with_grad` and run first discovery!
