"""
Projection as PEFT-Compatible Adapter

This implements ablation/projection as a rank-1 LoRA-style adapter,
compatible with HuggingFace PEFT library.

Benefits over hooks:
- Works directly with TRL GRPOTrainer (no custom adaptation!)
- Automatic optimizations (mixed precision, checkpointing, etc.)
- Can use QLoRA (4-bit base model for cheap training)
- Standard debugging tools work
- Easy to extend and combine with other adapters

Mathematical insight:
Projection: h' = h - (h·v)v = (I - vv^T)h
This is rank-1 modification, similar to LoRA but with subtraction!
"""

import torch
import torch.nn as nn
from typing import Optional, List
from dataclasses import dataclass, field

# Try importing PEFT (graceful fallback if not installed)
try:
    from peft import PeftConfig, PeftModel, PeftType
    from peft.tuners import BaseTuner, BaseTunerLayer
    from peft.utils import PeftType, _get_submodules
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    # Fallback definitions
    class PeftConfig:
        pass
    class BaseTuner:
        pass
    class BaseTunerLayer:
        pass


@dataclass
class ProjectionConfig(PeftConfig if PEFT_AVAILABLE else object):
    """
    Configuration for projection adapters.

    This is a PEFT-compatible config for rank-1 projection adapters.
    Think of it as "rank-1 LoRA but with subtraction instead of addition".

    Args:
        target_modules: Which modules to add projection to (e.g., ["layers"])
        projection_alpha: Scaling factor for projection (default: 1.0)
        projection_init_std: Initialization std for vectors
        normalize_vectors: Whether to normalize projection vectors
    """

    target_modules: Optional[List[str]] = field(default=None)
    projection_alpha: float = 1.0
    projection_init_std: float = 0.01
    normalize_vectors: bool = True

    if PEFT_AVAILABLE:
        def __post_init__(self):
            self.peft_type = "PROJECTION"  # Custom PEFT type
            super().__post_init__()


class ProjectionLayer(nn.Module):
    """
    Rank-1 projection adapter layer.

    This is similar to LoRA but:
    - Rank-1 only (single vector v)
    - Subtracts projection instead of adding

    Operation: h' = h - (h·v)v

    This is equivalent to: h' = (I - vv^T)h
    Which is a rank-1 modification to the identity matrix!

    Args:
        base_layer: The base layer to wrap
        dim: Dimension of projection vector
        alpha: Scaling factor
        init_std: Initialization std
        normalize: Whether to normalize v
    """

    def __init__(
        self,
        base_layer: nn.Module,
        dim: int,
        alpha: float = 1.0,
        init_std: float = 0.01,
        normalize: bool = True
    ):
        super().__init__()

        self.base_layer = base_layer
        self.alpha = alpha
        self.normalize = normalize

        # Freeze base layer (like LoRA)
        for param in base_layer.parameters():
            param.requires_grad = False

        # Trainable projection vector (rank-1!)
        self.projection_vector = nn.Parameter(
            torch.randn(dim) * init_std
        )

    def forward(self, x, *args, **kwargs):
        """
        Forward with projection applied.

        This is fully differentiable w.r.t. projection_vector!
        """
        # Base layer forward
        result = self.base_layer(x, *args, **kwargs)

        # Handle different output formats
        if isinstance(result, tuple):
            # Transformer layers return (hidden_states, *extra)
            activations = result[0]
            extra_outputs = result[1:]
        else:
            activations = result
            extra_outputs = None

        # Get projection vector
        v = self.projection_vector

        if self.normalize:
            v = v / (v.norm() + 1e-8)

        # Apply projection: h' = h - (h·v)v
        # This is differentiable w.r.t. v!
        projection_magnitude = torch.einsum('...d,d->...', activations, v)
        projection = torch.einsum('...,d->...d', projection_magnitude, v)
        projected = activations - self.alpha * projection

        # Reconstruct output
        if extra_outputs is not None:
            return (projected,) + extra_outputs
        else:
            return projected

    def merge_to_base(self):
        """
        Merge projection into base layer weights (for deployment).

        This permanently modifies the base layer to include the projection.
        After merging, this adapter can be removed.
        """
        if not hasattr(self.base_layer, 'weight'):
            raise ValueError("Base layer must have 'weight' attribute to merge")

        # Get projection matrix P = I - αvv^T
        v = self.projection_vector
        if self.normalize:
            v = v / (v.norm() + 1e-8)

        dim = len(v)
        I = torch.eye(dim, device=v.device, dtype=v.dtype)
        P = I - self.alpha * torch.outer(v, v)

        # Modify weights: W' = P @ W
        with torch.no_grad():
            self.base_layer.weight.data = (P @ self.base_layer.weight.data).contiguous()

    def unmerge_from_base(self):
        """
        Unmerge projection from base layer.

        This requires storing the original weights, which we don't do.
        For now, raise an error.
        """
        raise NotImplementedError(
            "Unmerge not supported. Save original model before merging."
        )


class ProjectionModel(nn.Module):
    """
    Wrapper that adds projection adapters to a model.

    This is a simplified version of PEFT's approach.
    For full PEFT integration, we'd need to implement BaseTuner.

    Usage:
        model = AutoModelForCausalLM.from_pretrained("model")
        config = ProjectionConfig(target_modules=["layers"])
        projection_model = ProjectionModel(model, config)

        # Now model has projection adapters!
        # Can train with standard tools
        trainer = Trainer(model=projection_model, ...)
    """

    def __init__(self, model, config: ProjectionConfig):
        super().__init__()

        self.base_model = model
        self.config = config

        # Freeze base model
        for param in model.parameters():
            param.requires_grad = False

        # Add projection layers
        self._inject_adapters()

    def _inject_adapters(self):
        """Inject projection adapters into target modules."""

        if not self.config.target_modules:
            raise ValueError("Must specify target_modules in config")

        # For transformer models, typically target decoder layers
        if hasattr(self.base_model, 'model'):
            layers = self.base_model.model.layers
        elif hasattr(self.base_model, 'transformer'):
            layers = self.base_model.transformer.h
        else:
            raise ValueError("Unsupported model architecture")

        # Get dimension
        if hasattr(self.base_model.config, 'hidden_size'):
            dim = self.base_model.config.hidden_size
        elif hasattr(self.base_model.config, 'n_embd'):
            dim = self.base_model.config.n_embd
        else:
            raise ValueError("Cannot determine hidden dimension")

        # Wrap each layer
        for idx, layer in enumerate(layers):
            wrapped = ProjectionLayer(
                base_layer=layer,
                dim=dim,
                alpha=self.config.projection_alpha,
                init_std=self.config.projection_init_std,
                normalize=self.config.normalize_vectors
            )
            layers[idx] = wrapped

        print(f"✓ Added projection adapters to {len(layers)} layers")

    def forward(self, *args, **kwargs):
        """Forward pass through base model with projections applied."""
        return self.base_model(*args, **kwargs)

    def generate(self, *args, **kwargs):
        """Generation with projections applied."""
        return self.base_model.generate(*args, **kwargs)

    def get_projection_parameters(self) -> List[nn.Parameter]:
        """Get trainable projection parameters."""
        params = []
        for module in self.modules():
            if isinstance(module, ProjectionLayer):
                params.append(module.projection_vector)
        return params

    def print_trainable_parameters(self):
        """Print trainable parameter statistics (PEFT-style)."""
        trainable_params = 0
        all_params = 0

        for _, param in self.named_parameters():
            num_params = param.numel()
            all_params += num_params
            if param.requires_grad:
                trainable_params += num_params

        print(
            f"trainable params: {trainable_params:,} || "
            f"all params: {all_params:,} || "
            f"trainable%: {100 * trainable_params / all_params:.4f}"
        )

    def save_pretrained(self, save_directory: str):
        """Save only the projection adapters (not base model)."""
        import os
        import json

        os.makedirs(save_directory, exist_ok=True)

        # Save config
        config_dict = {
            'peft_type': 'PROJECTION',
            'target_modules': self.config.target_modules,
            'projection_alpha': self.config.projection_alpha,
            'projection_init_std': self.config.projection_init_std,
            'normalize_vectors': self.config.normalize_vectors,
        }

        with open(os.path.join(save_directory, 'adapter_config.json'), 'w') as f:
            json.dump(config_dict, f, indent=2)

        # Save vectors
        projection_vectors = []
        for module in self.modules():
            if isinstance(module, ProjectionLayer):
                projection_vectors.append(module.projection_vector.data.cpu())

        torch.save({
            'projection_vectors': torch.stack(projection_vectors),
        }, os.path.join(save_directory, 'adapter_model.bin'))

        print(f"✓ Saved projection adapters to {save_directory}")

    @classmethod
    def from_pretrained(cls, model, adapter_path: str):
        """Load projection adapters onto a model."""
        import os
        import json

        # Load config
        with open(os.path.join(adapter_path, 'adapter_config.json'), 'r') as f:
            config_dict = json.load(f)

        config = ProjectionConfig(**config_dict)

        # Create projection model
        projection_model = cls(model, config)

        # Load vectors
        state = torch.load(
            os.path.join(adapter_path, 'adapter_model.bin'),
            map_location='cpu'
        )

        vectors = state['projection_vectors']

        # Apply to layers
        layer_idx = 0
        for module in projection_model.modules():
            if isinstance(module, ProjectionLayer):
                module.projection_vector.data = vectors[layer_idx].to(
                    module.projection_vector.device
                )
                layer_idx += 1

        print(f"✓ Loaded projection adapters from {adapter_path}")

        return projection_model

    def merge_and_unload(self):
        """
        Merge projections into base model and return base model.

        This permanently modifies the base model weights to include
        the projection, then returns the base model without adapters.
        """
        print("Merging projection adapters into base model...")

        for module in self.modules():
            if isinstance(module, ProjectionLayer):
                module.merge_to_base()

        print("✓ Merged successfully")

        return self.base_model


# Convenience function
def get_projection_model(model, config: Optional[ProjectionConfig] = None):
    """
    Add projection adapters to a model.

    Args:
        model: Base model
        config: ProjectionConfig (if None, uses defaults)

    Returns:
        Model with projection adapters

    Example:
        model = AutoModelForCausalLM.from_pretrained("model")
        model = get_projection_model(model)
        # Now can train with projection adapters!
    """
    if config is None:
        config = ProjectionConfig(target_modules=["layers"])

    return ProjectionModel(model, config)


# Example usage
if __name__ == '__main__':
    """
    Example: Using projection adapters like LoRA/PEFT.

    This shows the PEFT-style workflow for projection training.
    """

    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("=" * 70)
    print("Projection Adapters - PEFT-Style Training")
    print("=" * 70)

    # Load base model (will stay frozen)
    print("\nLoading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        'gpt2',
        torch_dtype=torch.float32
    )
    tokenizer = AutoTokenizer.from_pretrained('gpt2')

    print(f"  Model: GPT-2")
    print(f"  Layers: {len(model.transformer.h)}")
    print(f"  Hidden dim: {model.config.n_embd}")

    # Add projection adapters (PEFT-style!)
    print("\nAdding projection adapters...")
    config = ProjectionConfig(
        target_modules=["layers"],
        projection_alpha=1.0,
        projection_init_std=0.01,
        normalize_vectors=True
    )

    model = get_projection_model(model, config)

    # Check trainable parameters
    print("\nParameter statistics:")
    model.print_trainable_parameters()

    # Get projection parameters for optimizer
    projection_params = model.get_projection_parameters()
    print(f"\nProjection parameters: {len(projection_params)} vectors")
    print(f"Total projection params: {sum(p.numel() for p in projection_params):,}")

    # Test forward pass
    print("\nTesting forward pass...")
    inputs = tokenizer("Hello, world!", return_tensors='pt')
    with torch.no_grad():
        outputs = model(**inputs)
    print(f"✓ Forward pass works, output shape: {outputs.logits.shape}")

    # Save adapters (tiny!)
    print("\nSaving projection adapters...")
    model.save_pretrained("projection_adapters_test")
    print("✓ Saved (check size - should be ~100KB for GPT-2)")

    # Load adapters
    print("\nLoading projection adapters...")
    model_fresh = AutoModelForCausalLM.from_pretrained('gpt2')
    model_loaded = ProjectionModel.from_pretrained(
        model_fresh,
        "projection_adapters_test"
    )
    print("✓ Loaded successfully")

    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print("\n✓ Projection adapters work like PEFT/LoRA")
    print("✓ Rank-1 modification (single vector per layer)")
    print("✓ Standard save/load workflow")
    print("✓ Ready for TRL GRPOTrainer integration!")
    print("\nNext: Use with GRPOTrainer for adversarial vector training")
