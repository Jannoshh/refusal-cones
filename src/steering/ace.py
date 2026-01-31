"""
Affine Concept Editing (ACE) - Clean implementation following Marshall et al. (2024)

Paper: "Refusal in LLMs is an Affine Function" (arXiv:2411.09003v3)

The ACE formula (Equation 5):

    h' = h - proj_r(h) + proj_r(r⁻) + α·r

Where:
    h   = input activation
    r   = r⁺ - r⁻ (mean harmful - mean harmless, unnormalized)
    r⁻  = mean activation on harmless prompts (baseline/reference point)
    r⁺  = mean activation on harmful prompts
    α   = steering parameter:
          - α = 0: behave like harmless (no refusal)
          - α = 1: behave like harmful (full refusal)
    proj_r(x) = (x · r̂) · r̂  where r̂ = r / ||r||

Key insight: The baseline term proj_r(r⁻) ensures activations stay in a
sensible region of activation space after ablation, preventing nonsense outputs.

Usage:
    # Create steerer
    steerer = ACESteerer.from_activations(
        model=model,
        harmless_acts=harmless_activations,  # [n_samples, hidden_dim]
        harmful_acts=harmful_activations,     # [n_samples, hidden_dim]
        layer=15  # Single layer (paper recommends middle layers)
    )

    # Apply steering
    with steerer.apply(alpha=0.0):  # α=0 removes refusal
        output = model.generate(...)
"""

import torch
import torch.nn as nn
from typing import Optional, Union, List
from dataclasses import dataclass
from contextlib import contextmanager


@dataclass
class ACEConfig:
    """Configuration for ACE steering."""

    # Which layer to apply ACE (paper uses single layer near middle)
    layer: int = 15

    # Steering parameter: 0 = no refusal, 1 = full refusal
    alpha: float = 0.0

    # Position control (like EleutherAI steering-llama3)
    # None = all positions, or specify start:end slice
    position_start: Optional[int] = None
    position_end: Optional[int] = None


@dataclass
class ACEVectors:
    """
    Precomputed ACE vectors for a single layer.

    Following paper notation:
        r   = r⁺ - r⁻  (unnormalized steering direction)
        r_minus = r⁻   (baseline: mean harmless)
        r_plus = r⁺    (mean harmful)
    """
    r: torch.Tensor           # Steering direction [hidden_dim] (unnormalized!)
    r_minus: torch.Tensor     # Baseline [hidden_dim]
    r_plus: torch.Tensor      # Mean harmful [hidden_dim]
    layer: int                # Which layer these vectors are for

    @property
    def r_hat(self) -> torch.Tensor:
        """Normalized direction."""
        return self.r / (self.r.norm() + 1e-8)

    @property
    def r_norm(self) -> float:
        """Magnitude of steering direction."""
        return self.r.norm().item()

    @classmethod
    def from_activations(
        cls,
        harmless_acts: torch.Tensor,
        harmful_acts: torch.Tensor,
        layer: int
    ) -> "ACEVectors":
        """
        Compute ACE vectors from activation data.

        Args:
            harmless_acts: Activations on harmless prompts [n_samples, hidden_dim]
            harmful_acts: Activations on harmful prompts [n_samples, hidden_dim]
            layer: Which layer these activations are from

        Returns:
            ACEVectors with r, r_minus, r_plus
        """
        r_minus = harmless_acts.mean(dim=0)  # r⁻
        r_plus = harmful_acts.mean(dim=0)    # r⁺
        r = r_plus - r_minus                  # r (unnormalized!)

        return cls(r=r, r_minus=r_minus, r_plus=r_plus, layer=layer)

    def save(self, path: str):
        """Save vectors to file."""
        torch.save({
            'r': self.r,
            'r_minus': self.r_minus,
            'r_plus': self.r_plus,
            'layer': self.layer
        }, path)

    @classmethod
    def load(cls, path: str) -> "ACEVectors":
        """Load vectors from file."""
        data = torch.load(path)
        return cls(**data)


def ace_transform(
    h: torch.Tensor,
    r: torch.Tensor,
    r_minus: torch.Tensor,
    alpha: float = 0.0
) -> torch.Tensor:
    """
    Apply ACE transformation (Equation 5 from paper).

    h' = h - proj_r(h) + proj_r(r⁻) + α·r

    Args:
        h: Input activations [..., hidden_dim]
        r: Steering direction [hidden_dim] (unnormalized)
        r_minus: Baseline [hidden_dim] (mean harmless)
        alpha: Steering parameter (0 = no refusal, 1 = full refusal)

    Returns:
        Transformed activations [..., hidden_dim]
    """
    # Normalize direction for projection
    r_hat = r / (r.norm() + 1e-8)

    # proj_r(h) = (h · r̂) · r̂
    proj_h = torch.einsum('...d,d->...', h, r_hat)
    proj_h = torch.einsum('...,d->...d', proj_h, r_hat)

    # proj_r(r⁻) = (r⁻ · r̂) · r̂
    proj_r_minus = (r_minus @ r_hat) * r_hat

    # ACE formula: h' = h - proj_r(h) + proj_r(r⁻) + α·r
    h_prime = h - proj_h + proj_r_minus + alpha * r

    return h_prime


class ACESteerer:
    """
    ACE steering hook manager.

    Applies ACE transformation at a single layer (as per paper).

    Example:
        steerer = ACESteerer(model, vectors, config)

        # Method 1: Context manager
        with steerer.apply(alpha=0.0):
            output = model.generate(...)

        # Method 2: Manual control
        steerer.enable(alpha=0.0)
        output = model.generate(...)
        steerer.disable()
    """

    def __init__(
        self,
        model: nn.Module,
        vectors: ACEVectors,
        config: Optional[ACEConfig] = None
    ):
        """
        Initialize ACE steerer.

        Args:
            model: HuggingFace model with model.model.layers
            vectors: Precomputed ACE vectors
            config: Optional configuration
        """
        self.model = model
        self.vectors = vectors
        self.config = config or ACEConfig(layer=vectors.layer)

        self._hook_handle = None
        self._alpha = self.config.alpha

        # Move vectors to model device
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device('cpu')  # Fallback for models with no parameters
        self.vectors.r = self.vectors.r.to(device)
        self.vectors.r_minus = self.vectors.r_minus.to(device)
        self.vectors.r_plus = self.vectors.r_plus.to(device)

    @classmethod
    def from_activations(
        cls,
        model: nn.Module,
        harmless_acts: torch.Tensor,
        harmful_acts: torch.Tensor,
        layer: int,
        **config_kwargs
    ) -> "ACESteerer":
        """
        Create steerer directly from activation data.

        Args:
            model: HuggingFace model
            harmless_acts: Activations on harmless prompts [n_samples, hidden_dim]
            harmful_acts: Activations on harmful prompts [n_samples, hidden_dim]
            layer: Which layer to apply ACE
            **config_kwargs: Additional config options

        Returns:
            Configured ACESteerer
        """
        vectors = ACEVectors.from_activations(harmless_acts, harmful_acts, layer)
        config = ACEConfig(layer=layer, **config_kwargs)
        return cls(model, vectors, config)

    def _create_hook(self, alpha: float):
        """Create forward hook that applies ACE transformation."""
        def hook(module, input, output):
            # Handle tuple outputs (hidden_states, ...)
            if isinstance(output, tuple):
                h = output[0]
                rest = output[1:]
            else:
                h = output
                rest = None

            # Apply position masking if configured
            start = self.config.position_start
            end = self.config.position_end

            if start is not None or end is not None:
                # Only transform specified positions
                h_modified = h.clone()
                h_slice = h[..., start:end, :]
                h_transformed = ace_transform(
                    h_slice,
                    self.vectors.r,
                    self.vectors.r_minus,
                    alpha
                )
                h_modified[..., start:end, :] = h_transformed
            else:
                # Transform all positions
                h_modified = ace_transform(
                    h,
                    self.vectors.r,
                    self.vectors.r_minus,
                    alpha
                )

            if rest is not None:
                return (h_modified,) + rest
            return h_modified

        return hook

    def enable(self, alpha: Optional[float] = None):
        """
        Enable ACE steering.

        Args:
            alpha: Steering parameter (default: use config value)
                   0 = remove refusal, 1 = full refusal
        """
        if self._hook_handle is not None:
            self.disable()

        alpha = alpha if alpha is not None else self._alpha
        layer_module = self.model.model.layers[self.config.layer]
        self._hook_handle = layer_module.register_forward_hook(self._create_hook(alpha))

    def disable(self):
        """Disable ACE steering."""
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None

    @contextmanager
    def apply(self, alpha: Optional[float] = None):
        """
        Context manager for temporary ACE steering.

        Args:
            alpha: Steering parameter (0 = no refusal, 1 = full refusal)

        Example:
            with steerer.apply(alpha=0.0):
                output = model.generate(harmful_prompt)  # Will comply
        """
        self.enable(alpha)
        try:
            yield self
        finally:
            self.disable()

    def set_alpha(self, alpha: float):
        """Update steering parameter (requires re-enabling hook)."""
        self._alpha = alpha
        if self._hook_handle is not None:
            self.enable(alpha)


def collect_activations(
    model: nn.Module,
    tokenizer,
    prompts: List[str],
    layer: int,
    position: str = "last"
) -> torch.Tensor:
    """
    Collect activations at a specific layer for a list of prompts.

    Args:
        model: HuggingFace model
        tokenizer: Tokenizer
        prompts: List of prompts
        layer: Which layer to extract from
        position: "last" for last token, "all" for all tokens

    Returns:
        Activations tensor [n_prompts, hidden_dim] or [n_prompts, seq_len, hidden_dim]
    """
    device = next(model.parameters()).device
    activations = []

    for prompt in prompts:
        # Format with chat template if available
        if hasattr(tokenizer, 'apply_chat_template'):
            messages = [{"role": "user", "content": prompt}]
            formatted = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            formatted = prompt

        inputs = tokenizer(formatted, return_tensors="pt").to(device)

        # Hook to capture activations
        captured = {}
        def hook(module, input, output):
            if isinstance(output, tuple):
                captured['act'] = output[0].detach()
            else:
                captured['act'] = output.detach()

        handle = model.model.layers[layer].register_forward_hook(hook)

        with torch.no_grad():
            model(**inputs)

        handle.remove()

        if position == "last":
            activations.append(captured['act'][0, -1, :])  # Last token
        else:
            activations.append(captured['act'][0])  # All tokens

    return torch.stack(activations)


# Convenience function for quick setup
def setup_ace_steering(
    model: nn.Module,
    tokenizer,
    harmful_prompts: List[str],
    harmless_prompts: List[str],
    layer: int = None
) -> ACESteerer:
    """
    One-line setup for ACE steering.

    Args:
        model: HuggingFace model
        tokenizer: Tokenizer
        harmful_prompts: List of harmful prompts for computing r⁺
        harmless_prompts: List of harmless prompts for computing r⁻
        layer: Which layer to use (default: middle layer)

    Returns:
        Configured ACESteerer ready to use

    Example:
        steerer = setup_ace_steering(model, tokenizer, harmful, harmless)
        with steerer.apply(alpha=0.0):
            output = model.generate(...)  # Refusal removed
    """
    # Default to middle layer
    if layer is None:
        n_layers = len(model.model.layers)
        layer = n_layers // 2
        print(f"Using middle layer: {layer}")

    print(f"Collecting activations at layer {layer}...")
    harmful_acts = collect_activations(model, tokenizer, harmful_prompts, layer)
    harmless_acts = collect_activations(model, tokenizer, harmless_prompts, layer)

    print(f"  Harmful: {harmful_acts.shape}")
    print(f"  Harmless: {harmless_acts.shape}")

    steerer = ACESteerer.from_activations(
        model, harmless_acts, harmful_acts, layer
    )

    print(f"ACE vectors computed:")
    print(f"  ||r|| = {steerer.vectors.r_norm:.4f}")
    print(f"  Layer: {layer}")

    return steerer
