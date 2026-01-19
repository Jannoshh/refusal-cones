"""
Utilities for converting steering vectors to weight/bias modifications.

This module provides two approaches:
1. Runtime hooks (current approach)
2. Weight modifications (LoRA-style, differentiable)

Both are mathematically equivalent and can be used in training or deployment.
"""

import torch
import torch.nn as nn
from typing import Optional, List
import copy


def get_projection_matrix(v: torch.Tensor) -> torch.Tensor:
    """
    Compute projection matrix that removes direction v.

    P = I - vv^T (for normalized v)

    This is differentiable w.r.t. v!

    Args:
        v: Direction to project out (normalized)

    Returns:
        Projection matrix [dim, dim]
    """
    assert v.ndim == 1, "v must be 1D vector"
    # Normalize if not already
    v = v / (v.norm() + 1e-8)

    # I - vv^T
    I = torch.eye(v.shape[0], dtype=v.dtype, device=v.device)
    P = I - torch.outer(v, v)

    return P


def apply_ablation_to_weights(
    weight: torch.Tensor,
    v: torch.Tensor,
    in_place: bool = False
) -> torch.Tensor:
    """
    Apply ablation by modifying weight matrix.

    For a linear layer: h_out = W @ h_in
    To remove v from outputs: h_out = (W @ P) @ h_in
    where P = I - vv^T

    This is differentiable w.r.t. both W and v!

    Args:
        weight: Weight matrix [out_dim, in_dim] or [out_dim, in_dim, ...]
        v: Direction to ablate from outputs [out_dim]
        in_place: Modify weight in place (only for deployment)

    Returns:
        Modified weight matrix
    """
    P = get_projection_matrix(v)

    # For linear layer: W is [out_features, in_features]
    # We want to project outputs, so: W' = P @ W
    # (Applying P to left projects the output space)

    if weight.ndim == 2:
        # Standard linear layer
        W_modified = P @ weight
    else:
        # Handle other weight shapes (e.g., conv layers)
        # Assume first dimension is output features
        W_modified = torch.einsum('ij,...j->...i', P, weight)

    if in_place:
        weight.data.copy_(W_modified)
        return weight
    else:
        return W_modified


def apply_addition_to_bias(
    bias: Optional[torch.Tensor],
    v: torch.Tensor,
    alpha: float = 1.0,
    in_place: bool = False
) -> torch.Tensor:
    """
    Apply activation addition by modifying bias.

    For a layer: h_out = W @ h_in + b
    To add v: h_out = W @ h_in + (b + α*v)

    This is differentiable w.r.t. both b and v!

    Args:
        bias: Bias vector [out_dim] or None
        v: Vector to add [out_dim]
        alpha: Scaling factor
        in_place: Modify bias in place (only for deployment)

    Returns:
        Modified bias vector
    """
    if bias is None:
        # No existing bias, create new one
        return alpha * v

    b_modified = bias + alpha * v

    if in_place:
        bias.data.copy_(b_modified)
        return bias
    else:
        return b_modified


class VectorModifiedLayer(nn.Module):
    """
    Wrapper that applies vector modifications to a layer.

    This is differentiable and can be used in training!
    Similar to how LoRA wraps layers with low-rank updates.
    """

    def __init__(
        self,
        base_layer: nn.Module,
        ablation_vector: Optional[torch.Tensor] = None,
        addition_vector: Optional[torch.Tensor] = None,
        addition_alpha: float = 1.0
    ):
        """
        Initialize modified layer.

        Args:
            base_layer: Original layer (will be frozen)
            ablation_vector: Vector to ablate (optional)
            addition_vector: Vector to add (optional)
            addition_alpha: Scaling for addition
        """
        super().__init__()

        self.base_layer = base_layer

        # Freeze base layer
        for param in self.base_layer.parameters():
            param.requires_grad = False

        # Register vectors as parameters (trainable!)
        if ablation_vector is not None:
            self.ablation_vector = nn.Parameter(ablation_vector)
        else:
            self.register_parameter('ablation_vector', None)

        if addition_vector is not None:
            self.addition_vector = nn.Parameter(addition_vector)
        else:
            self.register_parameter('addition_vector', None)

        self.addition_alpha = addition_alpha

    def forward(self, x):
        """
        Forward pass with vector modifications applied.

        This is fully differentiable w.r.t. vectors!
        """
        # Get base layer output
        out = self.base_layer(x)

        # Apply ablation if specified
        if self.ablation_vector is not None:
            # Project out ablation direction
            v = self.ablation_vector / (self.ablation_vector.norm() + 1e-8)

            if isinstance(out, tuple):
                # Handle layers that return tuples (attention, etc.)
                activation = out[0]

                # Project out: h' = h - (h · v)v
                projection = torch.einsum('...d,d->...', activation, v)
                ablated = activation - torch.einsum('...,d->...d', projection, v)

                out = (ablated,) + out[1:]
            else:
                # Simple tensor output
                projection = torch.einsum('...d,d->...', out, v)
                out = out - torch.einsum('...,d->...d', projection, v)

        # Apply addition if specified
        if self.addition_vector is not None:
            if isinstance(out, tuple):
                activation = out[0]
                added = activation + self.addition_alpha * self.addition_vector
                out = (added,) + out[1:]
            else:
                out = out + self.addition_alpha * self.addition_vector

        return out


def convert_model_to_vector_modified(
    model,
    vectors: torch.Tensor,
    operation: str = 'ablation',
    alpha: float = 1.0,
    freeze_base: bool = True
) -> nn.Module:
    """
    Convert model to use VectorModifiedLayer wrappers.

    This creates a model where vector modifications are part of the
    forward pass and fully differentiable. Can be used for training!

    Similar to how PEFT wraps layers for LoRA.

    Args:
        model: Base model
        vectors: Per-layer vectors [n_layers, hidden_dim]
        operation: 'ablation' or 'addition'
        alpha: Scaling factor for addition
        freeze_base: Whether to freeze base model

    Returns:
        Modified model with vector operations in forward pass
    """
    model = copy.deepcopy(model)

    if freeze_base:
        for param in model.parameters():
            param.requires_grad = False

    # Wrap each transformer layer
    for idx, layer in enumerate(model.model.layers):
        v = vectors[idx]

        if operation == 'ablation':
            wrapped = VectorModifiedLayer(
                base_layer=layer,
                ablation_vector=v.clone()
            )
        elif operation == 'addition':
            wrapped = VectorModifiedLayer(
                base_layer=layer,
                addition_vector=v.clone(),
                addition_alpha=alpha
            )
        else:
            raise ValueError(f"Unknown operation: {operation}")

        # Replace layer
        model.model.layers[idx] = wrapped

    return model


def apply_vectors_to_weights_permanent(
    model,
    vectors: torch.Tensor,
    operation: str = 'ablation',
    alpha: float = 1.0
):
    """
    Permanently apply vectors to model weights (for deployment).

    DESTRUCTIVE: Modifies model weights in place.
    Use this AFTER training for deployment.

    Args:
        model: Model to modify
        vectors: Per-layer vectors [n_layers, hidden_dim]
        operation: 'ablation' or 'addition'
        alpha: Scaling factor for addition
    """
    for idx, layer in enumerate(model.model.layers):
        v = vectors[idx].to(layer.self_attn.o_proj.weight.device)

        if operation == 'ablation':
            # Modify output projection to ablate v
            # Note: For transformer layers, we typically want to modify
            # the output of the entire layer, so we modify o_proj

            apply_ablation_to_weights(
                layer.self_attn.o_proj.weight,
                v,
                in_place=True
            )

        elif operation == 'addition':
            # Add to layer norm bias (or create if doesn't exist)
            if hasattr(layer, 'post_attention_layernorm'):
                norm_layer = layer.post_attention_layernorm
                if norm_layer.bias is not None:
                    apply_addition_to_bias(
                        norm_layer.bias,
                        v,
                        alpha=alpha,
                        in_place=True
                    )
                else:
                    # Create bias if doesn't exist
                    norm_layer.bias = nn.Parameter(alpha * v)


def get_trainable_vector_parameters(model) -> List[nn.Parameter]:
    """
    Extract trainable vector parameters from a vector-modified model.

    Use this to get parameters for optimizer.

    Args:
        model: Model with VectorModifiedLayer wrappers

    Returns:
        List of vector parameters
    """
    params = []
    for module in model.modules():
        if isinstance(module, VectorModifiedLayer):
            if module.ablation_vector is not None:
                params.append(module.ablation_vector)
            if module.addition_vector is not None:
                params.append(module.addition_vector)
    return params
