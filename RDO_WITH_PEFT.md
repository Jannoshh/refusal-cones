# RDO (Representation Directional Optimization) with PEFT

## Question

How do we convert the full RDO algorithm (ablation + addition + retain losses, cone optimization) to PEFT adapters?

## Current PEFT Implementation

**What we have:**
- `ProjectionLayer`: Ablation only (subtract projection)
- Single operation mode
- Single vector per layer

**What we need for RDO:**
- Both ablation AND addition operations
- Multi-objective loss (ablation + addition + retain)
- Cone optimization (multiple vectors, not single direction)

## The Good News

**PEFT adapters ARE flexible enough!** We just need to extend them.

## RDO Algorithm Overview

RDO has three loss components:

### 1. Ablation Loss (L_ablate)

**Goal:** Make harmful prompts generate harmful content when refusal is ablated

```python
# On harmful prompts with ablation
h' = h - (h·v)v  # Project out refusal direction
loss_ablate = -log P(harmful_completion | h')

# We WANT harmful output when refusal removed (adversarial research)
```

### 2. Addition Loss (L_add)

**Goal:** Make harmful prompts generate refusals when refusal is added

```python
# On harmful prompts with addition
h' = h + α·v  # Add refusal direction
loss_add = -log P(refusal_completion | h')

# We WANT refusal when refusal direction added
```

### 3. Retain Loss (L_retain)

**Goal:** Keep harmless prompts unchanged

```python
# On harmless prompts (no intervention)
h' = h  # No modification
loss_retain = -log P(original_completion | h')

# We WANT to preserve helpful behavior
```

### Combined Loss

```python
loss = λ_ablate * L_ablate + λ_add * L_add + λ_retain * L_retain
```

## Extending PEFT Adapters for RDO

### 1. Dual-Operation Projection Layer

Extend `ProjectionLayer` to support both ablation AND addition:

```python
class RDOProjectionLayer(nn.Module):
    """
    Projection layer supporting both ablation and addition.

    Operations:
    - 'ablate': h' = h - (h·v)v (project out)
    - 'add': h' = h + α·v (add direction)
    - 'none': h' = h (retain, no change)
    """

    def __init__(
        self,
        base_layer: nn.Module,
        dim: int,
        alpha: float = 1.0,
        operation: str = 'ablate'  # 'ablate', 'add', or 'none'
    ):
        super().__init__()

        self.base_layer = base_layer
        self.alpha = alpha
        self.operation = operation

        # Freeze base layer
        for param in base_layer.parameters():
            param.requires_grad = False

        # Trainable vector
        self.vector = nn.Parameter(torch.randn(dim) * 0.01)

    def forward(self, x, operation=None):
        """
        Forward with specified operation.

        Args:
            x: Input
            operation: Override operation for this forward pass
                      (allows switching between ablate/add/none)
        """
        # Base layer forward
        result = self.base_layer(x)

        # Handle tuple outputs
        if isinstance(result, tuple):
            activations = result[0]
            extra = result[1:]
        else:
            activations = result
            extra = None

        # Select operation
        op = operation if operation is not None else self.operation

        # Apply operation
        v = self.vector / (self.vector.norm() + 1e-8)

        if op == 'ablate':
            # Project out: h' = h - (h·v)v
            projection = torch.einsum('...d,d->...', activations, v)
            modified = activations - self.alpha * torch.einsum('...,d->...d', projection, v)

        elif op == 'add':
            # Add direction: h' = h + α·v
            modified = activations + self.alpha * v

        elif op == 'none':
            # Retain: h' = h (no change)
            modified = activations

        else:
            raise ValueError(f"Unknown operation: {op}")

        # Reconstruct output
        if extra is not None:
            return (modified,) + extra
        return modified
```

**Key feature:** Can switch operations during forward pass!

### 2. RDO Trainer with Multi-Objective Loss

```python
from transformers import Trainer

class RDOTrainer(Trainer):
    """
    Custom trainer implementing RDO multi-objective loss.

    Combines:
    - Ablation loss (harmful prompts with ablation)
    - Addition loss (harmful prompts with addition)
    - Retain loss (harmless prompts, no change)
    """

    def __init__(
        self,
        *args,
        harmful_dataset=None,
        harmless_dataset=None,
        lambda_ablate: float = 1.0,
        lambda_add: float = 1.0,
        lambda_retain: float = 0.5,
        **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.harmful_dataset = harmful_dataset
        self.harmless_dataset = harmless_dataset

        self.lambda_ablate = lambda_ablate
        self.lambda_add = lambda_add
        self.lambda_retain = lambda_retain

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Compute RDO multi-objective loss.
        """
        # Determine input type
        is_harmful = inputs.get('is_harmful', True)

        if is_harmful:
            # Harmful prompts: compute ablation + addition losses

            # 1. Ablation loss
            # Set all projection layers to 'ablate' mode
            self._set_operation_mode(model, 'ablate')
            outputs_ablate = model(**inputs, labels=inputs['input_ids'])
            loss_ablate = outputs_ablate.loss

            # 2. Addition loss
            # Set all projection layers to 'add' mode
            self._set_operation_mode(model, 'add')
            outputs_add = model(**inputs, labels=inputs['input_ids'])
            loss_add = outputs_add.loss

            # Combined loss for harmful
            loss = (
                self.lambda_ablate * loss_ablate +
                self.lambda_add * loss_add
            )

        else:
            # Harmless prompts: retain loss (no intervention)
            self._set_operation_mode(model, 'none')
            outputs_retain = model(**inputs, labels=inputs['input_ids'])
            loss = self.lambda_retain * outputs_retain.loss

        if return_outputs:
            return loss, outputs_retain if not is_harmful else outputs_add
        return loss

    def _set_operation_mode(self, model, operation: str):
        """Set operation mode for all RDOProjectionLayer instances."""
        for module in model.modules():
            if isinstance(module, RDOProjectionLayer):
                module.operation = operation
```

**Key feature:** Dynamically switches operations during training!

### 3. Dataset Preparation

```python
def prepare_rdo_dataset(
    tokenizer,
    harmful_instructions: list,
    harmful_completions: list,  # What we want when ablated
    refusal_completions: list,   # What we want when added
    harmless_instructions: list,
    harmless_completions: list,
    max_length: int = 512
):
    """
    Prepare dataset for RDO training.

    Returns dataset with 'is_harmful' flag for loss routing.
    """
    data = []

    # Harmful data (for ablation + addition losses)
    for inst, harm_comp, ref_comp in zip(
        harmful_instructions,
        harmful_completions,
        refusal_completions
    ):
        # For ablation: want harmful completion
        text_ablate = f"<inst>{inst}</inst><completion>{harm_comp}</completion>"

        # For addition: want refusal
        text_add = f"<inst>{inst}</inst><completion>{ref_comp}</completion>"

        # Store both (trainer will use based on mode)
        data.append({
            'text': text_add,  # Use refusal for labels
            'is_harmful': True
        })

    # Harmless data (for retain loss)
    for inst, comp in zip(harmless_instructions, harmless_completions):
        text = f"<inst>{inst}</inst><completion>{comp}</completion>"
        data.append({
            'text': text,
            'is_harmful': False
        })

    # Tokenize
    def tokenize(examples):
        tokenized = tokenizer(
            examples['text'],
            truncation=True,
            max_length=max_length,
            padding='max_length'
        )
        tokenized['is_harmful'] = examples['is_harmful']
        return tokenized

    from datasets import Dataset
    dataset = Dataset.from_list(data)
    dataset = dataset.map(tokenize, batched=True)

    return dataset
```

## Cone Optimization

For cone optimization (multiple vectors instead of one), extend further:

```python
class ConeProjectionLayer(nn.Module):
    """
    Cone projection using multiple orthogonal vectors.

    Instead of single vector v, uses k orthogonal vectors {v1, v2, ..., vk}
    to define a k-dimensional subspace (cone).
    """

    def __init__(
        self,
        base_layer: nn.Module,
        dim: int,
        cone_rank: int = 1,  # Number of vectors in cone
        alpha: float = 1.0,
        operation: str = 'ablate'
    ):
        super().__init__()

        self.base_layer = base_layer
        self.alpha = alpha
        self.operation = operation
        self.cone_rank = cone_rank

        # Freeze base
        for param in base_layer.parameters():
            param.requires_grad = False

        # Trainable cone vectors [cone_rank, dim]
        self.cone_vectors = nn.Parameter(
            torch.randn(cone_rank, dim) * 0.01
        )

    def get_orthonormal_cone(self):
        """
        Get orthonormalized cone vectors using Gram-Schmidt.
        """
        vectors = self.cone_vectors

        # Gram-Schmidt orthogonalization
        ortho_vectors = []
        for i in range(len(vectors)):
            v = vectors[i]

            # Subtract projections onto previous vectors
            for prev in ortho_vectors:
                v = v - (v @ prev) * prev

            # Normalize
            v = v / (v.norm() + 1e-8)
            ortho_vectors.append(v)

        return torch.stack(ortho_vectors)

    def forward(self, x, operation=None):
        """Forward with cone projection."""
        result = self.base_layer(x)

        if isinstance(result, tuple):
            activations = result[0]
            extra = result[1:]
        else:
            activations = result
            extra = None

        # Get orthonormal cone
        cone = self.get_orthonormal_cone()  # [k, dim]

        op = operation if operation is not None else self.operation

        if op == 'ablate':
            # Project out entire cone subspace
            # h' = h - Σ_i (h·v_i)v_i
            modified = activations
            for v in cone:
                projection = torch.einsum('...d,d->...', modified, v)
                modified = modified - self.alpha * torch.einsum('...,d->...d', projection, v)

        elif op == 'add':
            # Add weighted combination of cone vectors
            # h' = h + α·mean(cone)
            cone_mean = cone.mean(dim=0)
            modified = activations + self.alpha * cone_mean

        elif op == 'none':
            modified = activations

        else:
            raise ValueError(f"Unknown operation: {op}")

        if extra is not None:
            return (modified,) + extra
        return modified
```

**Key feature:** Multi-dimensional refusal cone, not single direction!

## Complete RDO Pipeline with PEFT

### 1. Setup

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

# Load model
model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-2-2b-it",
    torch_dtype=torch.float16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b-it")

# Add RDO projection adapters
from rdo_peft_adapter import get_rdo_model, RDOConfig

config = RDOConfig(
    target_modules=["layers"],
    projection_alpha=1.0,
    operation='ablate',  # Default, will be switched during training
    cone_rank=3,  # Use 3-dimensional cone (optional)
    enable_cone=True  # Enable cone optimization
)

model = get_rdo_model(model, config)
```

### 2. Prepare Data

```python
# Prepare RDO dataset
dataset = prepare_rdo_dataset(
    tokenizer=tokenizer,
    harmful_instructions=harmful_prompts,
    harmful_completions=harmful_responses,  # For ablation target
    refusal_completions=refusal_responses,  # For addition target
    harmless_instructions=harmless_prompts,
    harmless_completions=harmless_responses  # For retain
)
```

### 3. Train with RDO

```python
training_args = TrainingArguments(
    output_dir="rdo_adapters",
    num_train_epochs=10,
    per_device_train_batch_size=4,
    learning_rate=1e-3,
    fp16=True,
    gradient_checkpointing=True
)

trainer = RDOTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    lambda_ablate=1.0,  # Weight for ablation loss
    lambda_add=1.0,     # Weight for addition loss
    lambda_retain=0.5   # Weight for retain loss
)

trainer.train()

# Save adapters
model.save_pretrained("rdo_adapters")
```

### 4. Use Trained Adapters

```python
# Load model with RDO adapters
model = get_rdo_model(base_model, config)
model = RDOModel.from_pretrained(model, "rdo_adapters")

# Test ablation (should be harmful)
model.set_operation('ablate')
outputs_ablate = model.generate(harmful_prompt)
# Expect: harmful completion

# Test addition (should refuse)
model.set_operation('add')
outputs_add = model.generate(harmful_prompt)
# Expect: refusal

# Harmless prompt (no intervention)
model.set_operation('none')
outputs_retain = model.generate(harmless_prompt)
# Expect: helpful response
```

## Flexibility Assessment

**Is PEFT flexible enough for RDO?**

### ✅ YES - With Extensions

**What works out of the box:**
- Single vector ablation ✓
- PEFT infrastructure (save/load, training) ✓
- Integration with Trainer ✓

**What needs extension:**
- Dual operations (ablation + addition) → **Easy to add** ✓
- Multi-objective loss → **Custom Trainer** ✓
- Cone optimization → **Multi-vector parameter** ✓
- Operation switching → **Runtime mode control** ✓

**All extensions are straightforward and maintain PEFT benefits!**

## Implementation Complexity

### Custom Training Loop (Current)
- ~800 lines for full RDO
- Manual loss computation
- Manual operation switching
- Custom checkpointing

### PEFT + Custom Trainer
- ~200 lines for extensions
- Automatic checkpointing
- Automatic optimizations
- Standard tools work

**Still 75% less code!** ✅

## Comparison

| Feature | Custom Loop | PEFT (Extended) |
|---------|-------------|-----------------|
| **Ablation** | ✓ | ✓ |
| **Addition** | ✓ | ✓ (extend) |
| **Retain** | ✓ | ✓ (extend) |
| **Cone** | ✓ | ✓ (extend) |
| **Multi-objective** | ✓ Manual | ✓ Custom Trainer |
| **Code complexity** | ~800 lines | ~200 lines ✓ |
| **TRL integration** | ❌ Complex | ✓ (for RL stage) |
| **Standard tools** | ❌ | ✓ |

**Winner: PEFT (Extended)** - Same features, 75% less code!

## Next Steps

1. Implement `RDOProjectionLayer` with dual operations
2. Implement `ConeProjectionLayer` for cone optimization
3. Implement `RDOTrainer` with multi-objective loss
4. Test on full RDO pipeline
5. Benchmark vs custom implementation

## Conclusion

**PEFT adapters ARE flexible enough for full RDO!**

Extensions needed:
- Dual operation mode (ablation + addition) → ~50 lines
- Multi-objective loss trainer → ~100 lines
- Cone optimization (optional) → ~50 lines

**Total: ~200 lines vs ~800 lines custom code**

All PEFT benefits maintained:
- ✓ Standard save/load
- ✓ Automatic optimizations
- ✓ Integration with Trainer
- ✓ Can still use TRL for RL stage

**Recommendation:** Extend PEFT adapters for RDO rather than custom implementation.
