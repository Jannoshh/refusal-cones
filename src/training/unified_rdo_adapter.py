#!/usr/bin/env python3
"""
Unified RDO Adapter - Affine Concept Editing

Implements the unified affine transformation from "Refusal in LLMs is an Affine Function"
(https://arxiv.org/abs/2411.09003v3)

Key insight: Combine projection and addition into single affine transformation:
    h' = P @ h + α·v

Where:
    P = (I - vv^T) or more generally (I - L@R) for rank-k
    v = steering vector
    α = scaling factor

This is more expressive than separate ablate/add modes.
"""

import torch
import torch.nn as nn
from typing import Optional, List, Literal
from dataclasses import dataclass, field

from projection_adapter import ProjectionConfig


@dataclass
class UnifiedRDOConfig(ProjectionConfig):
    """
    Configuration for unified affine RDO.

    Key difference from standard RDO:
    - Single affine transformation instead of separate ablate/add modes
    - Projection and addition happen simultaneously
    """

    # Affine transformation parameters
    projection_alpha: float = 1.0  # Scaling for projection (usually 1.0)
    addition_alpha: float = 1.0    # Scaling for addition component

    # Operation mode (simplified)
    operation: Literal['affine', 'project_only', 'add_only', 'none'] = 'affine'

    # Advanced: rank-k projection (for discovered geometry)
    enable_rank_k: bool = False
    rank_k: int = 1  # Number of vectors in subspace

    # Loss weights
    lambda_harmful: float = 1.0    # Weight for harmful examples
    lambda_harmless: float = 0.5   # Weight for harmless examples


class UnifiedRDOLayer(nn.Module):
    """
    Unified affine transformation layer.

    Implements: h' = (I - β·vv^T)h + α·v

    Where:
        β = projection_alpha (controls strength of ablation)
        α = addition_alpha (controls strength of addition)

    Special cases:
        - β=1, α=0: Pure projection (ablation)
        - β=0, α=1: Pure addition
        - β=1, α=1: Full affine (recommended)
    """

    def __init__(
        self,
        base_layer: nn.Module,
        dim: int,
        projection_alpha: float = 1.0,
        addition_alpha: float = 1.0,
        normalize: bool = True
    ):
        super().__init__()

        self.base_layer = base_layer
        self.projection_alpha = projection_alpha
        self.addition_alpha = addition_alpha
        self.normalize = normalize

        # Freeze base layer
        for param in base_layer.parameters():
            param.requires_grad = False

        # Trainable steering vector
        self.vector = nn.Parameter(torch.randn(dim) * 0.01)

    def forward(self, x, *args, **kwargs):
        """
        Forward with unified affine transformation.

        Computes: h' = (I - β·vv^T)h + α·v
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

        # Get normalized vector
        v = self.vector
        if self.normalize:
            v = v / (v.norm() + 1e-8)

        # Apply unified affine transformation
        # h' = h - β·(h·v)v + α·v
        #    = h + [α·v - β·(h·v)v]

        # Projection component: β·(h·v)v
        projection_magnitude = torch.einsum('...d,d->...', activations, v)
        projection = torch.einsum('...,d->...d', projection_magnitude, v)

        # Affine transformation
        modified = activations - self.projection_alpha * projection + self.addition_alpha * v

        # Reconstruct output
        if extra_outputs is not None:
            return (modified,) + extra_outputs
        return modified

    def get_projection_matrix(self) -> torch.Tensor:
        """
        Get the projection matrix P = I - β·vv^T.

        Returns:
            Projection matrix [dim, dim]
        """
        v = self.vector
        if self.normalize:
            v = v / (v.norm() + 1e-8)

        dim = len(v)
        I = torch.eye(dim, device=v.device, dtype=v.dtype)
        P = I - self.projection_alpha * torch.outer(v, v)
        return P

    def get_affine_components(self):
        """
        Get affine transformation components.

        Returns:
            (P, b) where h' = P @ h + b
        """
        v = self.vector
        if self.normalize:
            v = v / (v.norm() + 1e-8)

        P = self.get_projection_matrix()
        b = self.addition_alpha * v

        return P, b


class RankKUnifiedLayer(nn.Module):
    """
    Rank-k unified affine transformation.

    Instead of single vector, uses k orthogonal vectors to define
    a k-dimensional subspace.

    Implements: h' = (I - β·LL^T)h + α·v_mean

    Where:
        L = [v_1, v_2, ..., v_k] (orthonormal basis)
        v_mean = mean(v_1, ..., v_k)

    More powerful for complex refusal geometries discovered via
    gradient-based discovery.
    """

    def __init__(
        self,
        base_layer: nn.Module,
        dim: int,
        rank_k: int = 3,
        projection_alpha: float = 1.0,
        addition_alpha: float = 1.0,
        normalize: bool = True
    ):
        super().__init__()

        self.base_layer = base_layer
        self.rank_k = rank_k
        self.projection_alpha = projection_alpha
        self.addition_alpha = addition_alpha
        self.normalize = normalize

        # Freeze base layer
        for param in base_layer.parameters():
            param.requires_grad = False

        # Trainable subspace vectors [rank_k, dim]
        self.subspace_vectors = nn.Parameter(
            torch.randn(rank_k, dim) * 0.01
        )

    def get_orthonormal_basis(self):
        """
        Get orthonormalized subspace basis using Gram-Schmidt.

        Returns:
            Orthonormal basis [rank_k, dim]
        """
        vectors = self.subspace_vectors

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

    def forward(self, x, *args, **kwargs):
        """
        Forward with rank-k affine transformation.

        Computes: h' = (I - β·LL^T)h + α·v_mean
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

        # Get orthonormal basis
        basis = self.get_orthonormal_basis()  # [k, dim]

        # Project out entire subspace: h - Σ_i β·(h·v_i)v_i
        modified = activations
        for v in basis:
            projection_magnitude = torch.einsum('...d,d->...', modified, v)
            projection = torch.einsum('...,d->...d', projection_magnitude, v)
            modified = modified - self.projection_alpha * projection

        # Add mean steering direction: α·mean(basis)
        v_mean = basis.mean(dim=0)
        modified = modified + self.addition_alpha * v_mean

        # Reconstruct output
        if extra_outputs is not None:
            return (modified,) + extra_outputs
        return modified

    def get_projection_matrix(self) -> torch.Tensor:
        """
        Get the rank-k projection matrix P = I - β·LL^T.

        Returns:
            Projection matrix [dim, dim]
        """
        basis = self.get_orthonormal_basis()  # [k, dim]

        dim = basis.shape[1]
        I = torch.eye(dim, device=basis.device, dtype=basis.dtype)

        # LL^T = sum of outer products
        projection_sum = sum(torch.outer(v, v) for v in basis)
        P = I - self.projection_alpha * projection_sum

        return P


class UnifiedRDOModel(nn.Module):
    """
    Model wrapper with unified affine RDO adapters.

    Key difference from standard RDO:
    - Single affine transformation per layer (not separate modes)
    - Projection and addition happen simultaneously
    - Can use rank-k for complex geometries
    """

    def __init__(self, model, config: UnifiedRDOConfig):
        super().__init__()

        self.base_model = model
        self.config = config

        # Freeze base model
        for param in model.parameters():
            param.requires_grad = False

        # Add unified RDO layers
        self._inject_adapters()

    def _inject_adapters(self):
        """Inject unified affine adapters into target modules."""

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

        # Choose layer class
        layer_class = RankKUnifiedLayer if self.config.enable_rank_k else UnifiedRDOLayer

        # Wrap each layer
        for idx, layer in enumerate(layers):
            if self.config.enable_rank_k:
                wrapped = RankKUnifiedLayer(
                    base_layer=layer,
                    dim=dim,
                    rank_k=self.config.rank_k,
                    projection_alpha=self.config.projection_alpha,
                    addition_alpha=self.config.addition_alpha,
                    normalize=self.config.normalize_vectors
                )
            else:
                wrapped = UnifiedRDOLayer(
                    base_layer=layer,
                    dim=dim,
                    projection_alpha=self.config.projection_alpha,
                    addition_alpha=self.config.addition_alpha,
                    normalize=self.config.normalize_vectors
                )
            layers[idx] = wrapped

        layer_type = "rank-k unified" if self.config.enable_rank_k else "unified affine"
        print(f"✓ Added {layer_type} RDO adapters to {len(layers)} layers")
        if self.config.enable_rank_k:
            print(f"  Subspace rank: {self.config.rank_k} vectors per layer")

    def forward(self, *args, **kwargs):
        """Forward pass with unified affine transformation."""
        return self.base_model(*args, **kwargs)

    def generate(self, *args, **kwargs):
        """Generation with unified affine transformation."""
        return self.base_model.generate(*args, **kwargs)

    def get_affine_parameters(self) -> List[nn.Parameter]:
        """Get trainable affine transformation parameters."""
        params = []
        for module in self.modules():
            if isinstance(module, UnifiedRDOLayer):
                params.append(module.vector)
            elif isinstance(module, RankKUnifiedLayer):
                params.append(module.subspace_vectors)
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
        """Save unified RDO adapters."""
        import os
        import json

        os.makedirs(save_directory, exist_ok=True)

        # Save config
        config_dict = {
            'peft_type': 'UNIFIED_RDO',
            'target_modules': self.config.target_modules,
            'projection_alpha': self.config.projection_alpha,
            'addition_alpha': self.config.addition_alpha,
            'operation': self.config.operation,
            'enable_rank_k': self.config.enable_rank_k,
            'rank_k': self.config.rank_k,
            'normalize_vectors': self.config.normalize_vectors,
        }

        with open(os.path.join(save_directory, 'adapter_config.json'), 'w') as f:
            json.dump(config_dict, f, indent=2)

        # Save vectors/subspaces
        affine_data = []
        for module in self.modules():
            if isinstance(module, UnifiedRDOLayer):
                affine_data.append(module.vector.data.cpu())
            elif isinstance(module, RankKUnifiedLayer):
                affine_data.append(module.subspace_vectors.data.cpu())

        torch.save({
            'affine_data': affine_data,
            'enable_rank_k': self.config.enable_rank_k
        }, os.path.join(save_directory, 'adapter_model.bin'))

        print(f"✓ Saved unified RDO adapters to {save_directory}")

    @classmethod
    def from_pretrained(cls, model, adapter_path: str):
        """Load unified RDO adapters onto a model."""
        import os
        import json

        # Load config
        with open(os.path.join(adapter_path, 'adapter_config.json'), 'r') as f:
            config_dict = json.load(f)

        config = UnifiedRDOConfig(**config_dict)

        # Create unified RDO model
        unified_model = cls(model, config)

        # Load data
        state = torch.load(
            os.path.join(adapter_path, 'adapter_model.bin'),
            map_location='cpu'
        )

        affine_data = state['affine_data']

        # Apply to layers
        layer_idx = 0
        for module in unified_model.modules():
            if isinstance(module, UnifiedRDOLayer):
                module.vector.data = affine_data[layer_idx].to(module.vector.device)
                layer_idx += 1
            elif isinstance(module, RankKUnifiedLayer):
                module.subspace_vectors.data = affine_data[layer_idx].to(module.subspace_vectors.device)
                layer_idx += 1

        print(f"✓ Loaded unified RDO adapters from {adapter_path}")

        return unified_model


def get_unified_rdo_model(model, config: Optional[UnifiedRDOConfig] = None):
    """
    Add unified affine RDO adapters to a model.

    Args:
        model: Base model
        config: UnifiedRDOConfig (if None, uses defaults)

    Returns:
        Model with unified RDO adapters

    Example:
        >>> model = AutoModelForCausalLM.from_pretrained("model")
        >>> config = UnifiedRDOConfig(
        ...     target_modules=["layers"],
        ...     projection_alpha=1.0,
        ...     addition_alpha=1.0,
        ...     operation='affine'
        ... )
        >>> model = get_unified_rdo_model(model, config)
    """
    if config is None:
        config = UnifiedRDOConfig(target_modules=["layers"])

    return UnifiedRDOModel(model, config)


# Example usage
if __name__ == '__main__':
    """
    Example: Unified affine RDO vs separate operations.
    """

    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("=" * 70)
    print("Unified Affine RDO - Single Transformation")
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

    # Test 1: Unified affine transformation
    print("\n" + "-" * 70)
    print("Test 1: Unified Affine Transformation")
    print("-" * 70)

    config = UnifiedRDOConfig(
        target_modules=["layers"],
        projection_alpha=1.0,  # Full ablation
        addition_alpha=1.0,    # Full addition
        operation='affine'
    )

    model_unified = get_unified_rdo_model(model, config)
    model_unified.print_trainable_parameters()

    # Test forward
    inputs = tokenizer("Test prompt", return_tensors='pt')

    with torch.no_grad():
        output = model_unified(**inputs)
    print(f"  ✓ Unified affine works, shape: {output.logits.shape}")

    # Test 2: Rank-k unified
    print("\n" + "-" * 70)
    print("Test 2: Rank-k Unified (for complex geometry)")
    print("-" * 70)

    config_rank_k = UnifiedRDOConfig(
        target_modules=["layers"],
        projection_alpha=1.0,
        addition_alpha=1.0,
        enable_rank_k=True,
        rank_k=3  # 3-dimensional subspace
    )

    model_rank_k = get_unified_rdo_model(model, config_rank_k)
    model_rank_k.print_trainable_parameters()

    with torch.no_grad():
        output = model_rank_k(**inputs)
    print(f"  ✓ Rank-k unified works, shape: {output.logits.shape}")

    # Comparison
    print("\n" + "=" * 70)
    print("Comparison: Separate vs Unified")
    print("=" * 70)

    print("\nSeparate Operations (old):")
    print("  Ablation:  h' = (I - vv^T)h")
    print("  Addition:  h' = h + α·v")
    print("  → Need to switch modes during training")
    print("  → Two forward passes for multi-objective loss")

    print("\nUnified Affine (new):")
    print("  Combined:  h' = (I - vv^T)h + α·v")
    print("  → Single transformation")
    print("  → One forward pass")
    print("  → More expressive (can ablate AND steer)")

    print("\n" + "=" * 70)
    print("✓ Unified affine RDO ready!")
    print("=" * 70)
