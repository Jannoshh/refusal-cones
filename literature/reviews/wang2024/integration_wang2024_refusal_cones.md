# Integration Strategy: Wang et al. (2410.03415) with Refusal Cones

**Objective:** Analyze how false refusal vector extraction and orthogonalization can enhance refusal cones methodology for discovery and training.

---

## 1. Core Problem Alignment

### Refusal Cones Challenge
The refusal cones project discovers and trains vectors to **maximize harmfulness when refusal is ablated**, but:
- Assumes refusal is a simple geometric structure (cone/manifold)
- Does not distinguish true refusal from false refusal
- May inadvertently discover vectors that exploit false refusal patterns
- No explicit safety constraint during discovery

### Wang et al. Solution
Provides method to:
- **Isolate false refusal component** via orthogonal decomposition
- **Protect true refusal** during optimization
- **Enable fine-grained calibration** via λ parameter
- Requires no training (applicable pre/post-discovery)

### Synergistic Opportunity
Combine refusal cones' geometric discovery with Wang's component isolation to:
1. Discover refusal geometry while **protecting** true refusal boundaries
2. Optimize training to **avoid** exploiting false refusal patterns
3. Enable safety-aware mode discovery and ranking

---

## 2. Enhanced Discovery Workflow

### Current Refusal Cones Flow
```
[Refusal Vector r̂] → [Geometry Discovery] → [Train Vectors]
                          ↓
                    (Finds modes in refusal space)
                          ↓
                    No guarantee on true vs. false
```

### Proposed Enhanced Flow
```
[Harmful/Harmless Data] ──→ [Extract True Refusal r̂]
                             ↓
[Pseudo-Harmful Data] ──→ [Extract False Refusal ŵ]
                             ↓
                        [Orthogonalize]
                        ŵ' = ŵ - r̂(r̂^T·ŵ)
                             ↓
                    [Modified Discovery Space]
                    Discover modes perpendicular
                    to true refusal vector
                             ↓
                        [Train Vectors]
                    Ensure vectors don't exploit
                    false refusal patterns
```

---

## 3. Specific Technical Integrations

### 3.1 Discovery Phase Enhancement

**File:** `src/discovery/adaptive_geometry_discovery.py`

```python
class GradientGeometryDiscovery:
    def __init__(self, ..., false_refusal_vector=None, protect_true_refusal=True):
        """
        Args:
            false_refusal_vector: ŵ extracted from pseudo-harmful data
            protect_true_refusal: If True, constrains discovery to avoid
                                 exploiting true refusal in false contexts
        """
        self.false_refusal_vector = false_refusal_vector
        self.protect_true_refusal = protect_true_refusal

    def discover(self):
        """Enhanced discovery with false refusal protection."""

        # Phase 0: Setup (NEW)
        if self.protect_true_refusal:
            # Extract false refusal vector if not provided
            self.false_refusal_vector = self._extract_false_refusal()

            # Orthogonalize to separate true and false components
            self.true_refusal_ortho = self._orthogonalize_vectors(
                self.true_refusal,
                self.false_refusal_vector
            )

        # Phase 1: Gradient ascent (MODIFIED)
        v = self._gradient_ascent(
            constraint=self.true_refusal_ortho if self.protect_true_refusal else None
        )

        # Phase 2-3: Local/Global search (UNCHANGED)
        modes = self._local_and_global_search(v)

        return {
            'modes': modes,
            'false_refusal_protected': self.protect_true_refusal,
            'orthogonalization_info': {
                'false_refusal_extracted': self.false_refusal_vector is not None,
                'correlation_with_true': self._compute_correlation()
            }
        }

    def _extract_false_refusal(self):
        """Extract false refusal vector using Wang et al. method."""
        pseudo_harmful_activations = self._get_activations(
            self.pseudo_harmful_dataset
        )
        harmless_activations = self._get_activations(
            self.harmless_dataset
        )

        # Difference-in-means for false refusal
        false_refusal = (
            pseudo_harmful_activations.mean(dim=0) -
            harmless_activations.mean(dim=0)
        )

        return false_refusal / false_refusal.norm()

    def _orthogonalize_vectors(self, v1, v2):
        """Remove v2 component from v1."""
        return v1 - v2 * (v2.dot(v1))

    def _gradient_ascent(self, constraint=None):
        """Gradient ascent with optional constraint."""
        v = self.v_init

        for step in range(self.config.n_gradient_steps):
            R, grad = self.measure_refusal_with_grad(v)

            # Project gradient to tangent space (stay on sphere)
            grad_tangent = grad - grad.dot(v) * v

            # (NEW) If protecting true refusal, also project to avoid it
            if constraint is not None:
                # Remove component along true refusal direction
                grad_tangent = grad_tangent - constraint * constraint.dot(grad_tangent)

            # Update
            v = v + self.config.gradient_lr * grad_tangent
            v = v / v.norm()

            # Log
            logger.info(f"Step {step}: R={R:.4f}, ||grad||={grad.norm():.4f}")

        return v
```

### 3.2 Measurement Phase Modification

**File:** `src/measurement/vllm_hybrid_measurement.py`

```python
class VLLMHybridMeasurement:

    def __init__(self, ..., false_refusal_vector=None, lambda_ortho=1.0):
        """
        Args:
            false_refusal_vector: False refusal component ŵ'
            lambda_ortho: Partial orthogonalization coefficient [0, 1]
                         1.0 = full orthogonalization (strict safety)
                         0.5 = balanced
                         0.0 = no orthogonalization
        """
        self.false_refusal_vector = false_refusal_vector
        self.lambda_ortho = lambda_ortho

    def apply_ablation(self, v, model, **kwargs):
        """Apply ablation with false refusal protection."""

        # Original ablation: x' = x - v(v^T x)
        activations = self._extract_activations(model)
        ablated = activations - torch.einsum('...d,d->...', activations, v) * v

        # (NEW) Additionally remove false refusal component if present
        if self.false_refusal_vector is not None:
            # Partial orthogonalization: reduce false refusal impact
            # by factor lambda_ortho
            false_component = torch.einsum(
                '...d,d->...',
                ablated,
                self.false_refusal_vector
            ) * self.false_refusal_vector

            # Scale down false refusal removal
            ablated = ablated + (self.lambda_ortho * false_component)

        return self._inject_activations(model, ablated)

    def measure_refusal_with_grad(self, v, measure_false_refusal=False):
        """Enhanced measurement with false refusal metrics."""

        # Standard refusal measurement
        R, grad = self._measure_refusal_standard(v)

        # (NEW) Optionally measure false refusal impact
        if measure_false_refusal and self.false_refusal_vector is not None:
            R_false = self._measure_false_refusal(v)

            return {
                'true_refusal_reduction': R,
                'true_refusal_grad': grad,
                'false_refusal_reduction': R_false,  # Should be close to 0
                'quality_score': R - 0.1 * R_false  # Penalize false refusal reduction
            }

        return R, grad

    def _measure_false_refusal(self, v):
        """Measure unwanted reduction of true refusal (on false refusal contexts)."""

        # Test: does ablating v reduce refusal on pseudo-harmful queries?
        # (It shouldn't much if we're protecting true refusal)

        pseudo_harmful_responses = self._generate_with_ablation(
            v, self.pseudo_harmful_dataset
        )

        refusal_rate = self._classify_refusal(pseudo_harmful_responses)

        # Return as metric; log if suspiciously high
        return refusal_rate
```

### 3.3 Training Phase Modification

**File:** `src/training/rdo_peft_trainer.py`

```python
class RDOTrainerWithFalseRefusalProtection(RDOTrainerPEFT):

    def __init__(self, ..., false_refusal_vector=None, lambda_false_penalty=0.1):
        """
        Args:
            false_refusal_vector: ŵ' from Wang et al.
            lambda_false_penalty: Weight on false refusal penalty in loss
        """
        super().__init__(...)
        self.false_refusal_vector = false_refusal_vector
        self.lambda_false_penalty = lambda_false_penalty

    def compute_loss(self, batch, mode='train'):
        """Enhanced RDO loss with false refusal constraint."""

        # Standard RDO components
        loss_dict = self._standard_rdo_loss(batch, mode)

        # (NEW) False refusal penalty
        if self.false_refusal_vector is not None:
            loss_false = self._false_refusal_penalty(batch)
            loss_dict['loss_false_refusal'] = loss_false

            # Total loss
            total_loss = (
                loss_dict['loss_ablate'] +
                loss_dict['loss_add'] +
                loss_dict['loss_retain'] +
                self.lambda_false_penalty * loss_false
            )
        else:
            total_loss = loss_dict['loss_ablate'] + loss_dict['loss_add'] + loss_dict['loss_retain']

        return total_loss, loss_dict

    def _false_refusal_penalty(self, batch):
        """
        Penalize vectors that reduce refusal on false-harmful queries.

        If a trained vector v causes high compliance on pseudo-harmful queries,
        it might be exploiting false refusal pattern. Discourage this.
        """

        pseudo_harmful_inputs = batch['pseudo_harmful_input_ids']

        # Test: what's compliance rate on these with our trained vector?
        outputs = self.model(pseudo_harmful_inputs)
        compliance_rate = self._compute_compliance(outputs)

        # Penalize if too high (should stay low on false refusal)
        # Soft penalty: don't demand zero, just don't reward high compliance
        penalty = torch.clamp(compliance_rate - 0.1, min=0)

        return penalty
```

---

## 4. Configuration Updates

### 4.1 Discovery Config Enhancement

**File:** `src/discovery/adaptive_geometry_discovery.py` (config class)

```python
@dataclass
class GradientDiscoveryConfig:
    # ... existing fields ...

    # NEW: False refusal protection
    protect_true_refusal: bool = True
    use_pseudo_harmful_data: bool = True
    pseudo_harmful_dataset_path: str = "data/pseudo_harmful.json"

    # Orthogonalization parameters
    lambda_ortho: float = 1.0  # [0, 1] partial orthogonalization strength
    use_false_refusal_penalty: bool = True
```

### 4.2 Experiment Config

```python
# experiments/configs/e1_discovery_false_refusal_protected.yaml

discovery:
  method: gradient_discovery
  protect_true_refusal: true
  use_pseudo_harmful_data: true

  # False refusal extraction
  false_refusal:
    enabled: true
    num_pseudo_harmful_samples: 128
    num_validation_samples: 32
    dataset_source: "OR-Bench-Hard"

  # Orthogonalization
  orthogonalization:
    enabled: true
    lambda: 1.0  # Full orthogonalization
    monitor_correlation: true

training:
  false_refusal_penalty_weight: 0.1

measurement:
  measure_false_refusal: true
  log_compliance_on_pseudo_harmful: true
```

---

## 5. Evaluation Metrics

### 5.1 Additional Metrics to Track

```python
class DiscoveryEvaluator:

    def evaluate(self, discovered_modes, model):
        """Evaluate with false refusal awareness."""

        results = {
            # Standard metrics
            'max_refusal_ablation': ...,
            'modes_found': ...,

            # NEW: False refusal metrics
            'false_refusal_impact': {},
            'orthogonalization_info': {},
            'safety_margin': {}
        }

        for i, mode in enumerate(discovered_modes):
            # Measure compliance on both harmful and pseudo-harmful
            harmful_cr = self._measure_cr(mode, self.harmful_dataset)
            pseudo_harmful_cr = self._measure_cr(mode, self.pseudo_harmful_dataset)

            results['false_refusal_impact'][f'mode_{i}'] = {
                'true_refusal_reduction': harmful_cr,
                'false_refusal_reduction': pseudo_harmful_cr,
                'safety_margin': harmful_cr - pseudo_harmful_cr,
                'surgical_score': min(harmful_cr, 1 - pseudo_harmful_cr)
            }

        return results
```

### 5.2 Dashboard Metrics

```
Discovery Results:
├── Standard Metrics
│   ├── Modes found: N
│   ├── Max refusal ablation: R_max
│   └── Measurements used: M
│
├── False Refusal Metrics (NEW)
│   ├── True refusal protected: ✓
│   ├── False refusal reduction: CR_false (should be high)
│   ├── False refusal penalty used: ✓
│   └── Correlation (true/false): ρ
│
└── Safety Validation
    ├── Modes maintain <5% harmful CR: ✓
    ├── Modes increase >50% pseudo-harmful CR: ✓
    └── General capability <1% drop: ✓
```

---

## 6. Practical Integration Examples

### 6.1 End-to-End Workflow with False Refusal Protection

```bash
# 1. Prepare data (if not available)
python scripts/prepare_pseudo_harmful_data.py \
  --model meta-llama/Llama-2-7b-chat-hf \
  --output data/pseudo_harmful_llama2.json

# 2. Discover geometry with false refusal protection
uv run python -m src.discovery.gradient_discovery \
  --model_name llama2-7b-chat \
  --protect_true_refusal true \
  --use_false_refusal_penalty true \
  --config experiments/configs/e1_discovery_false_refusal_protected.yaml

# 3. Train vectors with false refusal awareness
uv run python experiments/e1_discovery/run_quality.py \
  --discovered_modes_path results/e1/discovered_modes.pt \
  --false_refusal_penalty_weight 0.1

# 4. Evaluate results including false refusal metrics
uv run python scripts/evaluate_false_refusal_impact.py \
  --model_path meta-llama/Llama-2-7b-chat-hf \
  --trained_vectors results/e1/trained_vectors.pt \
  --pseudo_harmful_dataset data/pseudo_harmful_llama2.json
```

### 6.2 Partial Orthogonalization Tuning

```python
# experiments/e5_ablations/run_lambda_sweep.py

from src.measurement.vllm_hybrid_measurement import VLLMHybridMeasurement

def run_lambda_sweep():
    """Explore λ parameter for safety-helpfulness tradeoff."""

    # Extract false refusal once
    false_refusal = extract_false_refusal_vector(model, datasets)

    results = {}
    for lambda_val in [0.0, 0.25, 0.5, 0.75, 1.0]:

        # Create measurement with given λ
        measurement = VLLMHybridMeasurement(
            model=model,
            false_refusal_vector=false_refusal,
            lambda_ortho=lambda_val
        )

        # Evaluate on all benchmarks
        results[lambda_val] = {
            'harmful_cr': measure_compliance(harmful_data),
            'pseudo_harmful_cr': measure_compliance(pseudo_harmful_data),
            'mmlu_score': measure_mmlu(),
            'margin': harmful_cr - pseudo_harmful_cr
        }

    # Plot tradeoff curve
    plot_lambda_tradeoff(results)
    return results
```

---

## 7. Safety Implications

### 7.1 Why This Matters

**Risk:** Refusal cones discovers jailbreak vectors. Accidentally discovering vectors that exploit **false refusal patterns** would be:
- Weaker than true refusal vectors (harder to find)
- More likely to harm legitimate use (over-refusal is bad)
- Less interpretable (harder to defend against)

**Protection:** Wang et al.'s orthogonalization ensures:
1. Discovered vectors target **true refusal**, not false refusal shortcuts
2. Jailbreak vectors are "surgical" (don't exploit over-caution)
3. More transferable and interpretable

**Benefit:** Integration makes refusal cones research:
- More scientifically rigorous
- More defensible ethically
- More useful for understanding refusal mechanisms

### 7.2 Recommended Safeguards

```python
# src/discovery/safety_monitor.py

class FalseRefusalSafetyMonitor:

    def validate_discovery(self, discovered_modes, true_refusal, false_refusal):
        """Ensure discovered modes don't exploit false refusal."""

        for mode in discovered_modes:
            # Compute alignment with true/false refusal
            align_true = abs(mode @ true_refusal)
            align_false = abs(mode @ false_refusal)

            # Check: true refusal much stronger
            if align_false > 0.5 * align_true:
                warn(f"Mode {mode} suspiciously aligned with false refusal!")

            # Safety test: should reduce refusal on harmful only
            harmful_cr = measure_cr(harmful_data)
            pseudo_cr = measure_cr(pseudo_harmful_data)

            if pseudo_cr > 0.7 * harmful_cr:
                warn(f"Mode {mode} may be exploiting false refusal patterns")

            # Document in results
            log({
                'mode_id': id(mode),
                'true_alignment': align_true,
                'false_alignment': align_false,
                'safety_check_passed': align_false < 0.5 * align_true
            })
```

---

## 8. Research Questions Enabled

Integration of Wang et al. enables new research:

1. **Geometry of False Refusal:**
   - Is false refusal a simple orthogonal component?
   - How does it relate to true refusal across models?
   - Is the relationship model-dependent?

2. **Discovery Safety:**
   - Can we discover refusal geometry while guaranteeing true refusal protection?
   - What's the impact of λ on discovered geometry?
   - Do different λ values discover different modes?

3. **Transferability:**
   - Does false refusal vector transfer across models?
   - Can we use Llama2's false refusal to protect Llama3 discovery?
   - How does pseudo-harmful dataset diversity affect transfer?

4. **Theoretical Understanding:**
   - Why does orthogonalization cleanly separate true/false refusal?
   - Can we predict which directions will exploit false refusal?
   - Is there a principled way to measure "surgical-ness"?

---

## 9. Implementation Timeline

### Phase 1 (Week 1): Integration Preparation
- [ ] Add false refusal extraction to measurement module
- [ ] Update discovery config to accept false_refusal_vector
- [ ] Create evaluation metrics for false refusal impact
- [ ] Write tests for orthogonalization operation

### Phase 2 (Week 2): Discovery Enhancement
- [ ] Modify gradient ascent to apply orthogonalization constraints
- [ ] Add false refusal penalty to discovery objectives
- [ ] Run experiments comparing with/without protection
- [ ] Document findings in ablation study

### Phase 3 (Week 3): Training & Evaluation
- [ ] Integrate false refusal penalty into RDO trainer
- [ ] Implement λ sweep experiments
- [ ] Comprehensive evaluation across all benchmarks
- [ ] Create dashboard showing false refusal metrics

### Phase 4 (Week 4): Documentation & Release
- [ ] Write tutorials on false refusal protection
- [ ] Create reproducible examples
- [ ] Update main README with new capabilities
- [ ] Release updated code with examples

---

## 10. Conclusion

Wang et al. (2410.03415) provides a **complementary technique** that enhances refusal cones:

| Dimension | Refusal Cones | + Wang et al. |
|-----------|---------------|-------------|
| Discovery Focus | "Where is refusal?" | "Where is TRUE refusal?" |
| Training Objective | Maximize ablation | Maximize ablation + minimize false refusal exploitation |
| Safety Assurance | Empirical testing | Explicit orthogonalization |
| Interpretability | Geometric modes | Geometric modes + false refusal alignment |
| Post-Deployment | Fixed vectors | Tunable vectors (λ) |

**Integration recommendation:** Use Wang et al.'s false refusal vector extraction as:
1. Prior constraint during discovery (protect via orthogonalization)
2. Penalty term during training (discourage false refusal exploitation)
3. Validation metric post-training (verify surgical nature)

This makes refusal cones research more rigorous, safer, and better-aligned with interpretability goals.

---

*Integration analysis completed: 2026-01-19*
*Prepared for implementation starting Week of 2026-01-20*
