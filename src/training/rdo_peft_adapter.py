"""
RDO (Representation Directional Optimization) with PEFT Adapters

Extends projection_adapter.py to support full RDO:
- Ablation + Addition operations (dual mode)
- Multi-objective loss (ablate + add + retain)
- Cone optimization (multi-dimensional subspace)

This maintains all PEFT benefits while adding RDO features.
"""

import torch
import torch.nn as nn
from typing import Optional, List, Literal
from dataclasses import dataclass, field
import copy

from .projection_adapter import ProjectionConfig, ProjectionLayer


@dataclass
class RDOConfig(ProjectionConfig):
    """
    Configuration for RDO projection adapters.

    Extends ProjectionConfig with RDO-specific options.
    """

    # Operation mode
    operation: Literal['ablate', 'add', 'both'] = 'both'  # Default: support both
    addition_alpha: float = 1.0  # Scaling for addition

    # Cone optimization
    enable_cone: bool = False  # Use cone (multi-vector) instead of single direction
    cone_rank: int = 1  # Number of vectors in cone

    # Loss weights (for RDOTrainer)
    lambda_ablate: float = 1.0
    lambda_add: float = 1.0
    lambda_retain: float = 0.5


class RDOProjectionLayer(nn.Module):
    """
    Projection layer supporting both ablation and addition.

    This is the core RDO layer that can:
    - Ablate: h' = h - (h·v)v (project out refusal)
    - Add: h' = h + α·v (add refusal)
    - None: h' = h (retain original)

    Operations can be switched dynamically during training.
    """

    def __init__(
        self,
        base_layer: nn.Module,
        dim: int,
        ablation_alpha: float = 1.0,
        addition_alpha: float = 1.0,
        default_operation: str = 'ablate',
        normalize: bool = True
    ):
        super().__init__()

        self.base_layer = base_layer
        self.ablation_alpha = ablation_alpha
        self.addition_alpha = addition_alpha
        self.default_operation = default_operation
        self.normalize = normalize

        # Freeze base layer
        for param in base_layer.parameters():
            param.requires_grad = False

        # Trainable projection vector
        self.vector = nn.Parameter(torch.randn(dim) * 0.01)

    def forward(self, x, operation: Optional[str] = None, *args, **kwargs):
        """
        Forward with specified operation.

        Args:
            x: Input
            operation: 'ablate', 'add', or 'none' (overrides default)
        """
        # Base layer forward
        result = self.base_layer(x, *args, **kwargs)

        # Handle tuple outputs (transformer layers)
        if isinstance(result, tuple):
            activations = result[0]
            extra_outputs = result[1:]
        else:
            activations = result
            extra_outputs = None

        # Select operation
        op = operation if operation is not None else self.default_operation

        # Get vector
        v = self.vector
        if self.normalize:
            v = v / (v.norm() + 1e-8)

        # Apply operation
        if op == 'ablate':
            # Project out: h' = h - (h·v)v
            projection_magnitude = torch.einsum('...d,d->...', activations, v)
            projection = torch.einsum('...,d->...d', projection_magnitude, v)
            modified = activations - self.ablation_alpha * projection

        elif op == 'add':
            # Add direction: h' = h + α·v
            modified = activations + self.addition_alpha * v

        elif op == 'none':
            # Retain: h' = h (no change)
            modified = activations

        else:
            raise ValueError(f"Unknown operation: {op}. Use 'ablate', 'add', or 'none'")

        # Reconstruct output
        if extra_outputs is not None:
            return (modified,) + extra_outputs
        return modified


class ConeProjectionLayer(nn.Module):
    """
    Cone projection using multiple orthogonal vectors.

    Instead of single direction, uses k orthogonal vectors to define
    a k-dimensional subspace (cone).

    For ablation: Projects out entire subspace
    For addition: Adds weighted combination of vectors
    """

    def __init__(
        self,
        base_layer: nn.Module,
        dim: int,
        cone_rank: int = 3,
        ablation_alpha: float = 1.0,
        addition_alpha: float = 1.0,
        default_operation: str = 'ablate',
        normalize: bool = True
    ):
        super().__init__()

        self.base_layer = base_layer
        self.cone_rank = cone_rank
        self.ablation_alpha = ablation_alpha
        self.addition_alpha = addition_alpha
        self.default_operation = default_operation
        self.normalize = normalize

        # Freeze base layer
        for param in base_layer.parameters():
            param.requires_grad = False

        # Trainable cone vectors [cone_rank, dim]
        self.cone_vectors = nn.Parameter(
            torch.randn(cone_rank, dim) * 0.01
        )

    def get_orthonormal_cone(self):
        """
        Get orthonormalized cone vectors using Gram-Schmidt.

        Returns:
            Orthonormal basis for the cone [cone_rank, dim]
        """
        vectors = self.cone_vectors

        # Gram-Schmidt orthogonalization
        ortho_vectors = []
        for i in range(len(vectors)):
            v = vectors[i].clone()

            # Subtract projections onto previous vectors
            for prev in ortho_vectors:
                v = v - (v @ prev) * prev

            # Normalize
            if self.normalize:
                v = v / (v.norm() + 1e-8)

            ortho_vectors.append(v)

        return torch.stack(ortho_vectors)

    def forward(self, x, operation: Optional[str] = None, *args, **kwargs):
        """
        Forward with cone projection.

        Args:
            x: Input
            operation: 'ablate', 'add', or 'none'
        """
        # Base layer forward
        result = self.base_layer(x, *args, **kwargs)

        # Handle tuple outputs
        if isinstance(result, tuple):
            activations = result[0]
            extra_outputs = result[1:]
        else:
            activations = result
            extra_outputs = None

        # Select operation
        op = operation if operation is not None else self.default_operation

        # Get orthonormal cone basis
        cone = self.get_orthonormal_cone()  # [k, dim]

        # Apply operation
        if op == 'ablate':
            # Project out entire cone subspace
            # h' = h - Σ_i (h·v_i)v_i
            modified = activations
            for v in cone:
                projection_magnitude = torch.einsum('...d,d->...', modified, v)
                projection = torch.einsum('...,d->...d', projection_magnitude, v)
                modified = modified - self.ablation_alpha * projection

        elif op == 'add':
            # Add weighted combination (mean) of cone vectors
            cone_mean = cone.mean(dim=0)
            modified = activations + self.addition_alpha * cone_mean

        elif op == 'none':
            # Retain: no change
            modified = activations

        else:
            raise ValueError(f"Unknown operation: {op}")

        # Reconstruct output
        if extra_outputs is not None:
            return (modified,) + extra_outputs
        return modified


class RDOModel(nn.Module):
    """
    Wrapper that adds RDO projection adapters to a model.

    This extends ProjectionModel with:
    - Dual operations (ablation + addition)
    - Runtime operation switching
    - Cone optimization support
    """

    def __init__(self, model, config: RDOConfig):
        super().__init__()

        self.base_model = model
        self.config = config

        # Freeze base model
        for param in model.parameters():
            param.requires_grad = False

        # Add RDO projection layers
        self._inject_adapters()

    def _inject_adapters(self):
        """Inject RDO projection adapters into target modules."""

        if not self.config.target_modules:
            raise ValueError("Must specify target_modules in config")

        # Get layers
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
        layer_class = ConeProjectionLayer if self.config.enable_cone else RDOProjectionLayer

        for idx, layer in enumerate(layers):
            if self.config.enable_cone:
                wrapped = ConeProjectionLayer(
                    base_layer=layer,
                    dim=dim,
                    cone_rank=self.config.cone_rank,
                    ablation_alpha=self.config.projection_alpha,
                    addition_alpha=self.config.addition_alpha,
                    default_operation=self.config.operation if self.config.operation != 'both' else 'ablate',
                    normalize=self.config.normalize_vectors
                )
            else:
                wrapped = RDOProjectionLayer(
                    base_layer=layer,
                    dim=dim,
                    ablation_alpha=self.config.projection_alpha,
                    addition_alpha=self.config.addition_alpha,
                    default_operation=self.config.operation if self.config.operation != 'both' else 'ablate',
                    normalize=self.config.normalize_vectors
                )
            layers[idx] = wrapped

        layer_type = "cone projection" if self.config.enable_cone else "RDO projection"
        print(f"✓ Added {layer_type} adapters to {len(layers)} layers")
        if self.config.enable_cone:
            print(f"  Cone rank: {self.config.cone_rank} vectors per layer")

    def forward(self, *args, operation: Optional[str] = None, **kwargs):
        """
        Forward pass with optional operation override.

        Args:
            operation: Override operation for this forward pass
        """
        if operation is not None:
            self.set_operation(operation)

        return self.base_model(*args, **kwargs)

    def generate(self, *args, operation: Optional[str] = None, **kwargs):
        """
        Generation with optional operation override.

        Args:
            operation: Override operation for generation
        """
        if operation is not None:
            self.set_operation(operation)

        return self.base_model.generate(*args, **kwargs)

    def set_operation(self, operation: str):
        """
        Set operation mode for all RDO layers.

        Args:
            operation: 'ablate', 'add', or 'none'
        """
        layer_class = ConeProjectionLayer if self.config.enable_cone else RDOProjectionLayer

        for module in self.modules():
            if isinstance(module, layer_class):
                module.default_operation = operation

        print(f"✓ Set operation mode to: {operation}")

    def get_projection_parameters(self) -> List[nn.Parameter]:
        """Get trainable projection/cone parameters."""
        params = []
        for module in self.modules():
            if isinstance(module, RDOProjectionLayer):
                params.append(module.vector)
            elif isinstance(module, ConeProjectionLayer):
                params.append(module.cone_vectors)
        return params

    def print_trainable_parameters(self):
        """Print trainable parameter statistics."""
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
        """Save RDO adapters."""
        import os
        import json

        os.makedirs(save_directory, exist_ok=True)

        # Save config
        config_dict = {
            'peft_type': 'RDO_PROJECTION',
            'target_modules': self.config.target_modules,
            'projection_alpha': self.config.projection_alpha,
            'addition_alpha': self.config.addition_alpha,
            'operation': self.config.operation,
            'enable_cone': self.config.enable_cone,
            'cone_rank': self.config.cone_rank,
            'normalize_vectors': self.config.normalize_vectors,
        }

        with open(os.path.join(save_directory, 'adapter_config.json'), 'w') as f:
            json.dump(config_dict, f, indent=2)

        # Save vectors/cones
        projection_data = []
        for module in self.modules():
            if isinstance(module, RDOProjectionLayer):
                projection_data.append(module.vector.data.cpu())
            elif isinstance(module, ConeProjectionLayer):
                projection_data.append(module.cone_vectors.data.cpu())

        torch.save({
            'projection_data': projection_data,
            'enable_cone': self.config.enable_cone
        }, os.path.join(save_directory, 'adapter_model.bin'))

        print(f"✓ Saved RDO adapters to {save_directory}")

    @classmethod
    def from_pretrained(cls, model, adapter_path: str):
        """Load RDO adapters onto a model."""
        import os
        import json

        # Load config
        with open(os.path.join(adapter_path, 'adapter_config.json'), 'r') as f:
            config_dict = json.load(f)

        config = RDOConfig(**config_dict)

        # Create RDO model
        rdo_model = cls(model, config)

        # Load data
        state = torch.load(
            os.path.join(adapter_path, 'adapter_model.bin'),
            map_location='cpu'
        )

        projection_data = state['projection_data']
        enable_cone = state['enable_cone']

        # Apply to layers
        layer_idx = 0
        for module in rdo_model.modules():
            if isinstance(module, RDOProjectionLayer):
                module.vector.data = projection_data[layer_idx].to(module.vector.device)
                layer_idx += 1
            elif isinstance(module, ConeProjectionLayer):
                module.cone_vectors.data = projection_data[layer_idx].to(module.cone_vectors.device)
                layer_idx += 1

        print(f"✓ Loaded RDO adapters from {adapter_path}")

        return rdo_model


def get_rdo_model(model, config: Optional[RDOConfig] = None):
    """
    Add RDO projection adapters to a model.

    Args:
        model: Base model
        config: RDOConfig (if None, uses defaults)

    Returns:
        Model with RDO adapters

    Example:
        model = AutoModelForCausalLM.from_pretrained("model")
        config = RDOConfig(
            target_modules=["layers"],
            enable_cone=True,
            cone_rank=3
        )
        model = get_rdo_model(model, config)
    """
    if config is None:
        config = RDOConfig(target_modules=["layers"])

    return RDOModel(model, config)


# Example usage
if __name__ == '__main__':
    """
    Example: Using RDO adapters for full RDO pipeline.
    """

    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("=" * 70)
    print("RDO Projection Adapters - Full RDO Support")
    print("=" * 70)

    # Load model
    print("\nLoading model...")
    model = AutoModelForCausalLM.from_pretrained(
        'gpt2',
        torch_dtype=torch.float32
    )
    tokenizer = AutoTokenizer.from_pretrained('gpt2')

    print(f"  Model: GPT-2")
    print(f"  Layers: {len(model.transformer.h)}")

    # Test 1: Single vector RDO
    print("\n" + "-" * 70)
    print("Test 1: Single Vector RDO (ablation + addition)")
    print("-" * 70)

    config = RDOConfig(
        target_modules=["layers"],
        operation='both',
        projection_alpha=1.0,
        addition_alpha=1.0,
        enable_cone=False
    )

    model_rdo = get_rdo_model(model, config)
    model_rdo.print_trainable_parameters()

    # Test operations
    print("\nTesting operation switching:")

    inputs = tokenizer("Test prompt", return_tensors='pt')

    model_rdo.set_operation('ablate')
    with torch.no_grad():
        out_ablate = model_rdo(**inputs)
    print(f"  ✓ Ablation works, shape: {out_ablate.logits.shape}")

    model_rdo.set_operation('add')
    with torch.no_grad():
        out_add = model_rdo(**inputs)
    print(f"  ✓ Addition works, shape: {out_add.logits.shape}")

    model_rdo.set_operation('none')
    with torch.no_grad():
        out_none = model_rdo(**inputs)
    print(f"  ✓ Retain works, shape: {out_none.logits.shape}")

    # Test 2: Cone optimization
    print("\n" + "-" * 70)
    print("Test 2: Cone Optimization (multi-dimensional subspace)")
    print("-" * 70)

    config_cone = RDOConfig(
        target_modules=["layers"],
        enable_cone=True,
        cone_rank=3,
        projection_alpha=1.0
    )

    model_cone = get_rdo_model(model, config_cone)
    model_cone.print_trainable_parameters()

    with torch.no_grad():
        out_cone = model_cone(**inputs, operation='ablate')
    print(f"  ✓ Cone projection works, shape: {out_cone.logits.shape}")

    # Test 3: Save/load
    print("\n" + "-" * 70)
    print("Test 3: Save/Load")
    print("-" * 70)

    model_rdo.save_pretrained("rdo_adapters_test")
    print("  ✓ Saved")

    model_fresh = AutoModelForCausalLM.from_pretrained('gpt2')
    model_loaded = RDOModel.from_pretrained(model_fresh, "rdo_adapters_test")
    print("  ✓ Loaded")

    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print("\n✓ RDO adapters support:")
    print("  - Ablation operation (project out)")
    print("  - Addition operation (add direction)")
    print("  - Retain operation (no change)")
    print("  - Runtime operation switching")
    print("  - Cone optimization (multi-dimensional)")
    print("  - Standard save/load")
    print("\n✓ Ready for multi-objective RDO training!")
    print("  See rdo_peft_trainer.py for training implementation")
