"""
Per-layer refusal vector training with PyTorch hooks.

This module implements training of separate refusal vectors for each layer,
allowing us to study how refusal cones evolve across layers in terms of
performance and dimensionality.

Key features:
- One vector per layer (not shared across layers)
- Each vector tries to minimize loss independently and together
- Smooth max loss to weight worst-performing vectors more heavily
- Uses PyTorch hooks instead of nnsight
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import List, Optional, Dict, Tuple
from jaxtyping import Float
from torch import Tensor
from tqdm import tqdm
import einops


def projection_einops(activation, direction):
    """Project activation onto direction vector."""
    proj = (
        einops.einsum(
            activation, direction.view(-1, 1), "... d_act, d_act single -> ... single"
        )
        * direction
    )
    return proj


def smooth_max_loss(losses: Tensor, temperature: float = 1.0) -> Tensor:
    """
    Smooth maximum loss (log-sum-exp approximation to max).

    This gives higher weight to worse-performing vectors, allowing the
    optimization to focus on improving the weakest layers.

    Args:
        losses: Tensor of per-layer losses [n_layers]
        temperature: Temperature parameter (lower = closer to hard max)
                    temperature → 0: approaches max(losses)
                    temperature → ∞: approaches mean(losses)

    Returns:
        Smooth maximum of the losses

    Reference:
        Common in robust optimization and adversarial training literature.
        Also known as "softmax temperature scaling" or "log-sum-exp trick".
    """
    # Numerically stable log-sum-exp
    max_loss = losses.max()
    exp_losses = torch.exp((losses - max_loss) / temperature)
    smooth_max = max_loss + temperature * torch.log(exp_losses.mean())
    return smooth_max


def weighted_smooth_max_loss(losses: Tensor, temperature: float = 1.0) -> Tensor:
    """
    Weighted smooth max that returns both the loss and the weights.

    The weights show how much each layer contributes to the loss,
    with higher weights for worse-performing layers.

    Args:
        losses: Tensor of per-layer losses [n_layers]
        temperature: Temperature parameter

    Returns:
        Tuple of (weighted_loss, weights)
    """
    # Compute softmax weights based on losses
    weights = torch.softmax(losses / temperature, dim=0)
    # Weighted average (biased toward higher losses)
    weighted_loss = (weights * losses).sum()
    return weighted_loss, weights


class PerLayerRefusalVectors(nn.Module):
    """
    Trainable refusal vectors, one per layer.

    Each layer has its own refusal direction that can be optimized
    independently and together. This allows us to:
    1. Study how refusal evolves across layers
    2. Find layer-specific refusal directions
    3. Analyze dimensionality of refusal subspace per layer
    """

    def __init__(
        self,
        n_layers: int,
        hidden_dim: int,
        init_vectors: Optional[List[Tensor]] = None,
        device: str = 'cuda',
        dtype: torch.dtype = torch.float32
    ):
        """
        Initialize per-layer refusal vectors.

        Args:
            n_layers: Number of transformer layers
            hidden_dim: Hidden dimension of the model
            init_vectors: Optional initial vectors (one per layer)
            device: Device to use
            dtype: Data type for vectors
        """
        super().__init__()
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.device = device
        self.dtype = dtype

        # Initialize vectors
        if init_vectors is not None:
            assert len(init_vectors) == n_layers
            vectors = torch.stack([v.to(dtype).to(device) for v in init_vectors])
        else:
            # Random initialization, normalized
            vectors = torch.randn(n_layers, hidden_dim, dtype=dtype, device=device)
            vectors = vectors / vectors.norm(dim=1, keepdim=True)

        self.vectors = nn.Parameter(vectors)

    def normalize(self):
        """Normalize all vectors to unit length."""
        with torch.no_grad():
            self.vectors.data = self.vectors.data / self.vectors.data.norm(dim=1, keepdim=True)

    def get_vector(self, layer_idx: int) -> Tensor:
        """Get normalized vector for a specific layer."""
        return self.vectors[layer_idx] / self.vectors[layer_idx].norm()

    def get_all_vectors(self) -> Tensor:
        """Get all vectors, normalized."""
        return self.vectors / self.vectors.norm(dim=1, keepdim=True)


def apply_per_layer_ablation(
    model,
    prompts: List[str],
    tokenizer,
    layer_vectors: PerLayerRefusalVectors,
    device: str = 'cuda'
) -> Tensor:
    """
    Apply per-layer directional ablation during forward pass.

    Each layer gets its own ablation direction applied.

    Args:
        model: HuggingFace model
        prompts: List of prompts
        tokenizer: Tokenizer
        layer_vectors: PerLayerRefusalVectors with vectors for each layer
        device: Device to use

    Returns:
        Model outputs with per-layer ablations applied
    """
    # Tokenize
    inputs = tokenizer(
        prompts,
        add_special_tokens=True,
        padding=True,
        truncation=False,
        return_tensors='pt'
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Get normalized vectors for each layer
    vectors = layer_vectors.get_all_vectors()

    # Create hooks for each layer
    handles = []

    def create_hook(layer_idx: int):
        vector = vectors[layer_idx].to(model.dtype)

        def hook(module, input, output):
            # Handle tuple outputs
            if isinstance(output, tuple):
                act = output[0]
            else:
                act = output

            # Apply ablation
            ablated = act - projection_einops(act, vector)

            # Return with same structure
            if isinstance(output, tuple):
                return (ablated,) + output[1:]
            else:
                return ablated

        return hook

    # Register hooks on all layers
    for idx, layer in enumerate(model.model.layers):
        handle = layer.register_forward_hook(create_hook(idx))
        handles.append(handle)

    # Forward pass
    with torch.no_grad():
        outputs = model(**inputs)

    # Clean up hooks
    for handle in handles:
        handle.remove()

    return outputs


def compute_ce_loss(logits: Tensor, labels: Tensor) -> Tensor:
    """
    Compute cross-entropy loss.

    Args:
        logits: Model logits [batch, seq_len, vocab]
        labels: Target labels [batch, seq_len]

    Returns:
        Cross-entropy loss
    """
    logits = logits.view(-1, logits.size(-1))
    labels = labels.view(-1)
    # Pad labels with ignore tokens to match logits shape
    padding = torch.full((logits.size(0),), -100, device=labels.device)
    padding[-labels.size(0):] = labels
    return torch.nn.functional.cross_entropy(logits, padding, ignore_index=-100)


def train_per_layer_vectors(
    model,
    tokenizer,
    train_dataset,
    n_layers: int,
    hidden_dim: int,
    batch_size: int = 8,
    epochs: int = 10,
    lr: float = 1e-3,
    device: str = 'cuda',
    use_smooth_max: bool = True,
    smooth_max_temperature: float = 1.0,
    init_vectors: Optional[List[Tensor]] = None,
    verbose: bool = True
) -> Tuple[PerLayerRefusalVectors, List[Dict]]:
    """
    Train per-layer refusal vectors.

    Args:
        model: HuggingFace model
        tokenizer: Tokenizer
        train_dataset: Training dataset
        n_layers: Number of layers
        hidden_dim: Hidden dimension
        batch_size: Batch size
        epochs: Number of epochs
        lr: Learning rate
        device: Device to use
        use_smooth_max: Whether to use smooth max loss
        smooth_max_temperature: Temperature for smooth max
        init_vectors: Optional initial vectors
        verbose: Whether to print progress

    Returns:
        Tuple of (trained_vectors, training_history)
    """
    # Initialize vectors
    layer_vectors = PerLayerRefusalVectors(
        n_layers=n_layers,
        hidden_dim=hidden_dim,
        init_vectors=init_vectors,
        device=device,
        dtype=torch.float32
    )

    # Optimizer
    optimizer = torch.optim.AdamW([layer_vectors.vectors], lr=lr, betas=(0.9, 0.98))

    # DataLoader
    dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Training history
    history = []

    # Training loop
    for epoch in range(epochs):
        epoch_losses = []
        epoch_per_layer_losses = []

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}") if verbose else dataloader

        for batch in pbar:
            optimizer.zero_grad()

            # Get ablation prompts and targets
            ablation_prompts = batch['ablation_prompt']
            ablation_labels = batch['ablation_labels'].to(device)

            # Tokenize
            inputs = tokenizer(
                ablation_prompts,
                add_special_tokens=True,
                padding=True,
                truncation=False,
                return_tensors='pt'
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # Get normalized vectors
            vectors = layer_vectors.get_all_vectors()

            # Compute per-layer losses
            per_layer_losses = []

            for layer_idx in range(n_layers):
                # Create hook for this layer only
                vector = vectors[layer_idx].to(model.dtype)

                def create_hook(vec):
                    def hook(module, input, output):
                        if isinstance(output, tuple):
                            act = output[0]
                        else:
                            act = output
                        ablated = act - projection_einops(act, vec)
                        if isinstance(output, tuple):
                            return (ablated,) + output[1:]
                        else:
                            return ablated
                    return hook

                # Apply ablation only at this layer
                handle = model.model.layers[layer_idx].register_forward_hook(create_hook(vector))

                # Forward pass
                outputs = model(**inputs)

                # Compute loss
                logits = outputs.logits[:, :-1]
                loss = compute_ce_loss(logits, ablation_labels)
                per_layer_losses.append(loss)

                # Clean up hook
                handle.remove()

            # Stack per-layer losses
            per_layer_losses = torch.stack(per_layer_losses)

            # Compute final loss
            if use_smooth_max:
                total_loss = smooth_max_loss(per_layer_losses, smooth_max_temperature)
            else:
                total_loss = per_layer_losses.mean()

            # Also compute combined loss (all layers together)
            # This encourages vectors to work together
            all_handles = []
            for layer_idx in range(n_layers):
                vector = vectors[layer_idx].to(model.dtype)

                def create_hook(vec):
                    def hook(module, input, output):
                        if isinstance(output, tuple):
                            act = output[0]
                        else:
                            act = output
                        ablated = act - projection_einops(act, vec)
                        if isinstance(output, tuple):
                            return (ablated,) + output[1:]
                        else:
                            return ablated
                    return hook

                handle = model.model.layers[layer_idx].register_forward_hook(create_hook(vector))
                all_handles.append(handle)

            # Forward pass with all ablations
            outputs = model(**inputs)
            logits = outputs.logits[:, :-1]
            combined_loss = compute_ce_loss(logits, ablation_labels)

            # Clean up hooks
            for handle in all_handles:
                handle.remove()

            # Final loss: combination of per-layer and combined
            final_loss = 0.5 * total_loss + 0.5 * combined_loss

            # Backward
            final_loss.backward()

            # Clip gradients
            torch.nn.utils.clip_grad_norm_(layer_vectors.vectors, max_norm=1.0)

            # Update
            optimizer.step()

            # Normalize vectors
            layer_vectors.normalize()

            # Log
            epoch_losses.append(final_loss.item())
            epoch_per_layer_losses.append(per_layer_losses.detach().cpu())

            if verbose and isinstance(pbar, tqdm):
                pbar.set_postfix({'loss': final_loss.item()})

        # Epoch summary
        avg_loss = sum(epoch_losses) / len(epoch_losses)
        avg_per_layer = torch.stack(epoch_per_layer_losses).mean(dim=0)

        history.append({
            'epoch': epoch,
            'avg_loss': avg_loss,
            'per_layer_losses': avg_per_layer.tolist()
        })

        if verbose:
            print(f"Epoch {epoch+1}: Avg Loss = {avg_loss:.4f}")
            print(f"Per-layer losses: {avg_per_layer.tolist()}")

    return layer_vectors, history


def evaluate_per_layer_vectors(
    model,
    tokenizer,
    eval_dataset,
    layer_vectors: PerLayerRefusalVectors,
    refusal_tokens: List[int],
    device: str = 'cuda',
    batch_size: int = 16
) -> Dict:
    """
    Evaluate per-layer refusal vectors.

    Returns metrics for:
    - Each layer independently
    - All layers together
    - Refusal scores

    Args:
        model: HuggingFace model
        tokenizer: Tokenizer
        eval_dataset: Evaluation dataset
        layer_vectors: Trained PerLayerRefusalVectors
        refusal_tokens: Token IDs indicating refusal
        device: Device to use
        batch_size: Batch size

    Returns:
        Dictionary with evaluation metrics
    """
    from scoring import refusal_score_fn

    model.eval()

    # Get evaluation prompts
    eval_prompts = [d['instruction'] if isinstance(d, dict) else d for d in eval_dataset]

    results = {
        'per_layer_scores': [],
        'combined_score': 0.0,
        'per_layer_refusal_rates': []
    }

    # Get vectors
    vectors = layer_vectors.get_all_vectors()

    # Evaluate each layer independently
    for layer_idx in range(layer_vectors.n_layers):
        layer_scores = []

        for i in range(0, len(eval_prompts), batch_size):
            batch_prompts = eval_prompts[i:i+batch_size]

            # Tokenize
            inputs = tokenizer(batch_prompts, add_special_tokens=True, padding=True,
                             truncation=False, return_tensors='pt')
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # Apply ablation at this layer only
            vector = vectors[layer_idx].to(model.dtype)

            def create_hook(vec):
                def hook(module, input, output):
                    if isinstance(output, tuple):
                        act = output[0]
                    else:
                        act = output
                    ablated = act - projection_einops(act, vec)
                    if isinstance(output, tuple):
                        return (ablated,) + output[1:]
                    else:
                        return ablated
                return hook

            handle = model.model.layers[layer_idx].register_forward_hook(create_hook(vector))

            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits[:, -1]
                scores = refusal_score_fn(logits, refusal_tokens)
                layer_scores.extend(scores.cpu().tolist())

            handle.remove()

        results['per_layer_scores'].append(layer_scores)
        results['per_layer_refusal_rates'].append(sum(1 for s in layer_scores if s > 0) / len(layer_scores))

    # Evaluate all layers together
    combined_scores = []
    for i in range(0, len(eval_prompts), batch_size):
        batch_prompts = eval_prompts[i:i+batch_size]

        inputs = tokenizer(batch_prompts, add_special_tokens=True, padding=True,
                         truncation=False, return_tensors='pt')
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Apply all ablations
        handles = []
        for layer_idx in range(layer_vectors.n_layers):
            vector = vectors[layer_idx].to(model.dtype)

            def create_hook(vec):
                def hook(module, input, output):
                    if isinstance(output, tuple):
                        act = output[0]
                    else:
                        act = output
                    ablated = act - projection_einops(act, vec)
                    if isinstance(output, tuple):
                        return (ablated,) + output[1:]
                    else:
                        return ablated
                return hook

            handle = model.model.layers[layer_idx].register_forward_hook(create_hook(vector))
            handles.append(handle)

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits[:, -1]
            scores = refusal_score_fn(logits, refusal_tokens)
            combined_scores.extend(scores.cpu().tolist())

        for handle in handles:
            handle.remove()

    results['combined_score'] = sum(combined_scores) / len(combined_scores)
    results['combined_refusal_rate'] = sum(1 for s in combined_scores if s > 0) / len(combined_scores)

    return results
